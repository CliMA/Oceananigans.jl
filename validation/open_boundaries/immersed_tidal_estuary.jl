# A salt-wedge estuary in two dimensions: a tidally modulated river enters through the immersed
# head of the estuary (a time-dependent `NormalFlow` immersed boundary condition on `u`), fresh
# water flows out over a salt wedge, and the mouth radiates into the open ocean.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode, znode
using CairoMakie
using Printf
using Statistics: mean

arch = GPU()
Lx, Lz = 50kilometers, 20
Nx, Nz = 500, 40

head = 2kilometers        # the estuary head (land) occupies x < head
river_depth = 5           # [m]
U_river = 0.3             # mean river velocity [m s⁻¹]
U_tide = 0.25             # tidal modulation of the river velocity [m s⁻¹]
ω = 2π / 12.42hours       # M2 frequency [s⁻¹]
S_ocean = 35              # ocean salinity [g kg⁻¹]
stop_time = 10days

grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5), x = (0, Lx), z = (-Lz, 0), topology = (Bounded, Flat, Bounded))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(x -> ifelse(x < head, 0, -Lz)))

@inline river_velocity(i, j, k, grid, clock, fields) =
    ifelse(znode(k, grid, Center()) > -river_depth, U_river + U_tide * sin(ω * clock.time), zero(U_river))

radiation = PerturbationAdvection(inflow_timescale = 1hour, outflow_timescale = 6hours)
u_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(west = NormalFlowBoundaryCondition(river_velocity, discrete_form = true)),
                                east = NormalFlowBoundaryCondition(0; scheme = radiation))
S_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(west = ValueBoundaryCondition(0)),
                                east = ValueBoundaryCondition(S_ocean; scheme = PerturbationAdvection(inflow_timescale = 1hour, outflow_timescale = Inf)))

closure = VerticalScalarDiffusivity(ν = 1e-3, κ = 1e-4)
free_surface = SplitExplicitFreeSurface(grid; substeps = 30)
buoyancy = SeawaterBuoyancy(constant_temperature = 10)

model = HydrostaticFreeSurfaceModel(grid; free_surface, closure, buoyancy, tracers = :S,
                                    momentum_advection = WENO(order = 9), tracer_advection = WENO(order = 9),
                                    boundary_conditions = (; u = u_bcs, S = S_bcs))

set!(model, S = S_ocean)

simulation = Simulation(model; Δt = 30, stop_time)

u, v, w = model.velocities
S = model.tracers.S
η = model.free_surface.displacement
∫η = Field(Integral(η, dims = 1))
river_transport(t) = (U_river + U_tide * sin(ω * t)) * river_depth
∫u_mouth = Field(Integral(view(u, Nx + 1, :, :), dims = 3))

wall_clock = Ref(time_ns())
function progress(sim)
    compute!(∫η)
    compute!(∫u_mouth)
    elapsed = (time_ns() - wall_clock[]) / 1e9
    @info @sprintf("iter %d, t = %s, river = %.2f m² s⁻¹, mouth = %.2f m² s⁻¹, ∫η dx = %.1f m², min S = %.2f, wall %.1f s",
                   iteration(sim), prettytime(sim), river_transport(time(sim)), first(∫u_mouth), first(∫η), minimum(S), elapsed)
    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, TimeInterval(6hours))

filename = "immersed_tidal_estuary"
simulation.output_writers[:fields] = JLD2Writer(model, (; u, w, S, η);
                                                filename, schedule = TimeInterval(30minutes), overwrite_existing = true)
run!(simulation)

# Plot
St = FieldTimeSeries(filename * ".jld2", "S")
ut = FieldTimeSeries(filename * ".jld2", "u")
ηt = FieldTimeSeries(filename * ".jld2", "η")
times = St.times
Nt = length(times)
x, y, z = nodes(St)
xu, yu, zu = nodes(ut)
xη, yη, zη = nodes(ηt)

fig = Figure(size = (1200, 800))
n = Observable(Nt)
title = @lift @sprintf("Tidal estuary with an immersed head, t = %s", prettytime(times[$n]))
Label(fig[0, 1:2], title; fontsize = 20)

axS = Axis(fig[1, 1]; xlabel = "x (km)", ylabel = "z (m)", title = "salinity")
Sn = @lift interior(St[$n], :, 1, :)
hm = heatmap!(axS, x / 1e3, z, Sn; colormap = :haline, colorrange = (0, S_ocean))
Colorbar(fig[1, 2], hm)

axu = Axis(fig[2, 1]; xlabel = "x (km)", ylabel = "z (m)", title = "u (m s⁻¹)")
un = @lift interior(ut[$n], :, 1, :)
hm = heatmap!(axu, xu / 1e3, zu, un; colormap = :balance, colorrange = (-0.5, 0.5))
Colorbar(fig[2, 2], hm)

axη = Axis(fig[3, 1]; xlabel = "x (km)", ylabel = "η (m)", title = "free surface")
ηn = @lift interior(ηt[$n], :, 1, 1)
lines!(axη, xη / 1e3, ηn)
ylims!(axη, -0.5, 0.5)

save(filename * ".png", fig)
record(fig, filename * ".mp4", 1:Nt; framerate = 12) do i
    n[] = i
end
