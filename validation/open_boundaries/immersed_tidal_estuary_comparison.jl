# The tidal estuary of `immersed_tidal_estuary.jl` run twice over the same fluid cells: with the
# river head on the west boundary of the domain, and behind a fictitious immersed wall whose faces
# are opened with a `NormalFlow` immersed boundary condition. The two runs must agree.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode
using Oceananigans.BoundaryConditions: NormalRadiation, GravityWaveRadiationBoundaryCondition, SurfaceWaveRadiationBoundaryCondition
using CUDA
using CairoMakie
using Printf

arch = GPU()
const Lx, Lz = 48kilometers, 20    # head to mouth
Nx, Nz = 480, 40
Nwall = 20                         # land columns behind the immersed head
const river_depth = 5              # [m]
const U_river = 0.3                # mean river velocity [m s⁻¹]
const U_tide = 0.25                # tidal modulation of the river velocity [m s⁻¹]
const ω = 2π / 12.42hours          # M2 frequency [s⁻¹]
const S_ocean = 35                 # ocean salinity [g kg⁻¹]
Δt = 10
stop_time = 10days

# The river velocity tapers to zero over the 2 m above the river bed
@inline river_profile(z) = clamp((z + river_depth) / 2, 0, 1)
@inline river_transport_velocity(t) = U_river + U_tide * sin(ω * t)
@inline river_velocity(z, t) = river_transport_velocity(t) * river_profile(z)
# Fresh water enters through the river faces; below them the head returns the interior salinity, a no-flux wall
@inline head_salinity(z, S) = ifelse(z > -river_depth, zero(S), S)

# Domain conditions take (j, k, …); immersed ones take (i, j, k, …) at the fluid cell next to the face
@inline domain_river(j, k, grid, clock, fields) = river_velocity(znode(k, grid, Center()), clock.time)
@inline domain_river_transport(j, k, grid, clock, fields, depth) = river_transport_velocity(clock.time) * depth
@inline domain_head_salinity(j, k, grid, clock, fields) = @inbounds head_salinity(znode(k, grid, Center()), fields.S[1, j, k])
@inline immersed_river(i, j, k, grid, clock, fields) = river_velocity(znode(k, grid, Center()), clock.time)
@inline immersed_head_salinity(i, j, k, grid, clock, fields) = @inbounds head_salinity(znode(k, grid, Center()), fields.S[i, j, k])

function estuary_simulation(immersed_head, filename)
    mouth_u = NormalFlowBoundaryCondition(0; scheme = NormalRadiation(; inflow_timescale = Inf, outflow_timescale = Inf))
    mouth_S = ValueBoundaryCondition(S_ocean; scheme = PerturbationAdvection(inflow_timescale = 1hour, outflow_timescale = Inf))
    mouth_U = GravityWaveRadiationBoundaryCondition((0, 0))
    η_bcs = FieldBoundaryConditions(east = SurfaceWaveRadiationBoundaryCondition())

    if immersed_head
        underlying_grid = RectilinearGrid(arch; size = (Nx + Nwall, Nz), halo = (4, 4), x = (-Nwall * Lx / Nx, Lx), z = (-Lz, 0),
                                          topology = (Bounded, Flat, Bounded))
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(x -> ifelse(x < 0, 0, -Lz)))
        head_u = ImmersedBoundaryCondition(west = NormalFlowBoundaryCondition(immersed_river, discrete_form = true))
        head_S = ImmersedBoundaryCondition(west = ValueBoundaryCondition(immersed_head_salinity, discrete_form = true))
        u_bcs = FieldBoundaryConditions(immersed = head_u, east = mouth_u)
        S_bcs = FieldBoundaryConditions(immersed = head_S, east = mouth_S)
        U_bcs = FieldBoundaryConditions(east = mouth_U)
    else
        grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (4, 4), x = (0, Lx), z = (-Lz, 0), topology = (Bounded, Flat, Bounded))
        river_profile_depth = sum(river_profile, Array(znodes(grid, Center()))) * Lz / Nz
        head_U = NormalFlowBoundaryCondition(domain_river_transport; discrete_form = true, parameters = river_profile_depth)
        u_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(domain_river, discrete_form = true), east = mouth_u)
        S_bcs = FieldBoundaryConditions(west = ValueBoundaryCondition(domain_head_salinity, discrete_form = true), east = mouth_S)
        U_bcs = FieldBoundaryConditions(west = head_U, east = mouth_U)
    end

    closure = (VerticalScalarDiffusivity(ν = 1e-3, κ = 1e-4), HorizontalScalarDiffusivity(ν = 10, κ = 10))
    model = HydrostaticFreeSurfaceModel(grid; closure, tracers = :S,
                                        free_surface = SplitExplicitFreeSurface(grid; substeps = 30),
                                        buoyancy = SeawaterBuoyancy(constant_temperature = 10),
                                        momentum_advection = WENO(order = 5), tracer_advection = WENO(order = 5),
                                        boundary_conditions = (; u = u_bcs, U = U_bcs, η = η_bcs, S = S_bcs))
    set!(model, S = S_ocean)

    simulation = Simulation(model; Δt, stop_time)
    u, v, w = model.velocities
    S = model.tracers.S
    η = model.free_surface.displacement
    simulation.output_writers[:fields] = JLD2Writer(model, (; u, w, S, η); filename, schedule = TimeInterval(30minutes), overwrite_files = true)

    wall_clock = Ref(time_ns())
    function progress(sim)
        @info @sprintf("%s: iter %d, t = %s, min S = %.2f, max |u| = %.2f, wall %.1f s", filename, iteration(sim), prettytime(sim),
                       minimum(S), maximum(abs, u), (time_ns() - wall_clock[]) / 1e9)
        wall_clock[] = time_ns()
        return nothing
    end
    add_callback!(simulation, progress, TimeInterval(12hours))

    return simulation
end

run!(estuary_simulation(false, "estuary_domain_head"))
run!(estuary_simulation(true, "estuary_immersed_head"))

# Compare over the fluid cells
load(name) = (S = FieldTimeSeries(name * ".jld2", "S"), u = FieldTimeSeries(name * ".jld2", "u"), η = FieldTimeSeries(name * ".jld2", "η"))
domain = load("estuary_domain_head")
immersed = load("estuary_immersed_head")
times = domain.S.times
Nt = length(times)
fluid = Nwall+1:Nwall+Nx
faces = Nwall+1:Nwall+Nx+1
Δx, Δz = Lx / Nx, Lz / Nz
x, y, z = nodes(domain.S)
xu, yu, zu = nodes(domain.u)
xi, yi, zi = nodes(immersed.S)

S_domain(n) = interior(domain.S[n], :, 1, :)
S_immersed(n) = interior(immersed.S[n], fluid, 1, :)
u_domain(n) = interior(domain.u[n], :, 1, :)
u_immersed(n) = interior(immersed.u[n], faces, 1, :)
η_domain(n) = interior(domain.η[n], :, 1, 1)
η_immersed(n) = interior(immersed.η[n], fluid, 1, 1)

ΔS = [maximum(abs, S_immersed(n) .- S_domain(n)) for n in 1:Nt]
Δu = [maximum(abs, u_immersed(n) .- u_domain(n)) for n in 1:Nt]
Δη = [maximum(abs, η_immersed(n) .- η_domain(n)) for n in 1:Nt]
volume_domain = [sum(η_domain(n)) * Δx for n in 1:Nt]
volume_immersed = [sum(η_immersed(n)) * Δx for n in 1:Nt]
mouth_domain = [sum(u_domain(n)[end, :]) * Δz for n in 1:Nt]
mouth_immersed = [sum(u_immersed(n)[end, :]) * Δz for n in 1:Nt]
head_domain = [sum(u_domain(n)[1, :]) * Δz for n in 1:Nt]
head_immersed = [sum(u_immersed(n)[1, :]) * Δz for n in 1:Nt]

@info @sprintf("Largest differences over %s: ΔS = %.2e g kg⁻¹, Δu = %.2e m s⁻¹, Δη = %.2e m", prettytime(times[end]), maximum(ΔS), maximum(Δu), maximum(Δη))

# Land columns are left blank
function S_immersed_with_land(n)
    Sn = Array(interior(immersed.S[n], :, 1, :))
    Sn[1:Nwall, :] .= NaN
    return Sn
end

fig = Figure(size = (1200, 1100))
n = Observable(Nt)
title = @lift @sprintf("Tidal estuary with the river head on the domain boundary and behind an immersed wall, t = %s", prettytime(times[$n]))
Label(fig[0, 1:2], title; fontsize = 18)

axS = Axis(fig[1, 1]; ylabel = "z (m)", title = "salinity (g kg⁻¹), head on the domain boundary")
hm = heatmap!(axS, x / 1e3, z, @lift(S_domain($n)); colormap = :haline, colorrange = (0, S_ocean))
Colorbar(fig[1, 2], hm)

axI = Axis(fig[2, 1]; ylabel = "z (m)", title = "salinity (g kg⁻¹), head behind an immersed wall")
hm = heatmap!(axI, xi / 1e3, zi, @lift(S_immersed_with_land($n)); colormap = :haline, colorrange = (0, S_ocean))
Colorbar(fig[2, 2], hm)

ΔS_range = maximum(ΔS)
axD = Axis(fig[3, 1]; ylabel = "z (m)", title = "salinity difference, immersed wall − domain boundary (g kg⁻¹)")
hm = heatmap!(axD, x / 1e3, z, @lift(S_immersed($n) .- S_domain($n)); colormap = :balance, colorrange = (-ΔS_range, ΔS_range))
Colorbar(fig[3, 2], hm)

axη = Axis(fig[4, 1]; xlabel = "x (km)", ylabel = "η (m)", title = "free surface")
lines!(axη, x / 1e3, @lift(η_domain($n)); label = "domain boundary")
lines!(axη, x / 1e3, @lift(η_immersed($n)); linestyle = :dash, label = "immersed wall")
axislegend(axη; position = :rt)
ylims!(axη, -0.5, 0.5)

for ax in (axS, axI, axD, axη)
    xlims!(ax, -Nwall * Δx / 1e3, Lx / 1e3)
end

filename = "immersed_tidal_estuary_comparison"
save(filename * ".png", fig)
CairoMakie.record(fig, filename * ".mp4", 1:Nt; framerate = 12) do i
    n[] = i
end

fig = Figure(size = (1000, 900))
time_in_days = times / 1day
ax = Axis(fig[1, 1]; ylabel = "∫η dx (m²)", title = "free-surface volume")
lines!(ax, time_in_days, volume_domain; label = "domain boundary")
lines!(ax, time_in_days, volume_immersed; linestyle = :dash, label = "immersed wall")
axislegend(ax; position = :lt)

ax = Axis(fig[2, 1]; ylabel = "transport (m² s⁻¹)", title = "transport through the head and the mouth")
lines!(ax, time_in_days, head_domain; label = "head, domain boundary")
lines!(ax, time_in_days, head_immersed; linestyle = :dash, label = "head, immersed wall")
lines!(ax, time_in_days, mouth_domain; label = "mouth, domain boundary")
lines!(ax, time_in_days, mouth_immersed; linestyle = :dash, label = "mouth, immersed wall")
axislegend(ax; position = :rt)

ax = Axis(fig[3, 1]; xlabel = "time (days)", ylabel = "largest difference", yscale = log10, title = "immersed wall − domain boundary")
lines!(ax, time_in_days, max.(ΔS, 1e-16); label = "salinity (g kg⁻¹)")
lines!(ax, time_in_days, max.(Δu, 1e-16); label = "u (m s⁻¹)")
lines!(ax, time_in_days, max.(Δη, 1e-16); label = "η (m)")
axislegend(ax; position = :rb)

save(filename * "_time_series.png", fig)
