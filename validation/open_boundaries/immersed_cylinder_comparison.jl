# Pulsating flow past a cylinder, run twice over the same fluid cells: with the inflow on the west
# boundary of the domain, and through a fictitious immersed wall whose faces are opened with a
# `NormalFlow` immersed boundary condition. The two runs must agree.

using Oceananigans
using Oceananigans.Grids: xnode, inactive_cell
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using CUDA
using CairoMakie
using Printf

arch = GPU()
const D = 1                        # cylinder diameter
const U = 1                        # mean inflow speed
const T = 8                        # pulsation period
Lx, Ly = 24D, 8D
Nx, Ny = 384, 128
Nwall = 16                         # land columns behind the immersed inlet
ν = U * D / 100                    # Re = 100
Δt = 0.01
stop_time = 6T

@inline inflow(t) = U * (1 + sin(2π * t / T) / 2)
@inline domain_inflow(j, k, grid, clock, fields) = inflow(clock.time)
# The immersed condition applies to every west-facing immersed face: the wall faces at x = 0 are open, the cylinder stays closed
@inline immersed_inflow(i, j, k, grid, clock, fields) = ifelse(xnode(i, grid, Face()) < D, inflow(clock.time), zero(grid))
# The cylinder sits off the channel axis so that the wake sheds from the start
cylinder(x, y) = (x - 6D)^2 + (y - D / 4)^2 < (D / 2)^2

function cylinder_simulation(immersed_inlet, filename)
    outflow = NormalFlowBoundaryCondition(U; scheme = PerturbationAdvection(inflow_timescale = 1, outflow_timescale = Inf))
    # The lee of the cylinder faces west too: both runs open it with zero velocity, so that only the inlet differs
    inlet = ImmersedBoundaryCondition(west = NormalFlowBoundaryCondition(immersed_inflow, discrete_form = true))

    if immersed_inlet
        underlying_grid = RectilinearGrid(arch; size = (Nx + Nwall, Ny), halo = (5, 5), x = (-Nwall * Lx / Nx, Lx), y = (-Ly / 2, Ly / 2),
                                          topology = (Bounded, Bounded, Flat))
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary((x, y) -> (x < 0) | cylinder(x, y)))
        u_bcs = FieldBoundaryConditions(immersed = inlet, east = outflow)
    else
        underlying_grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (5, 5), x = (0, Lx), y = (-Ly / 2, Ly / 2), topology = (Bounded, Bounded, Flat))
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(cylinder))
        u_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(domain_inflow, discrete_form = true), immersed = inlet, east = outflow)
    end

    pressure_solver = ConjugateGradientPoissonSolver(grid; reltol = 1e-9, abstol = 0, maxiter = 1000)
    model = NonhydrostaticModel(grid; advection = WENO(order = 5), closure = ScalarDiffusivity(; ν), pressure_solver,
                                boundary_conditions = (; u = u_bcs))
    set!(model, u = U)

    simulation = Simulation(model; Δt, stop_time)
    u, v, w = model.velocities
    ζ = Field(∂x(v) - ∂y(u))
    simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, ζ); filename, schedule = TimeInterval(T / 20), overwrite_files = true)

    wall_clock = Ref(time_ns())
    function progress(sim)
        @info @sprintf("%s: iter %d, t = %.2f, max |u| = %.3f, wall %.1f s", filename, iteration(sim), time(sim),
                       maximum(abs, u), (time_ns() - wall_clock[]) / 1e9)
        wall_clock[] = time_ns()
        return nothing
    end
    add_callback!(simulation, progress, TimeInterval(T / 2))

    return simulation
end

run!(cylinder_simulation(false, "cylinder_domain_inlet"))
run!(cylinder_simulation(true, "cylinder_immersed_inlet"))

# Compare over the fluid nodes: cells inside the cylinder are masked in the domain run but hold
# extrapolated values behind the open faces of the immersed run
load(name) = (u = FieldTimeSeries(name * ".jld2", "u"), v = FieldTimeSeries(name * ".jld2", "v"), ζ = FieldTimeSeries(name * ".jld2", "ζ"))
domain = load("cylinder_domain_inlet")
immersed = load("cylinder_immersed_inlet")
times = domain.u.times
Nt = length(times)
fluid = Nwall+1:Nwall+Nx
faces = Nwall+1:Nwall+Nx+1
Δx = Lx / Nx
xζ, yζ, zζ = nodes(domain.ζ)
xi, yi, zi = nodes(immersed.ζ)

active = [!inactive_cell(i, j, 1, domain.u.grid) for i in 1:Nx, j in 1:Ny]
active_cell(i, j) = (1 ≤ i ≤ Nx) & (1 ≤ j ≤ Ny) && active[i, j]
u_fluid = [(i == 1 || active_cell(i-1, j)) && (i == Nx + 1 || active_cell(i, j)) for i in 1:Nx+1, j in 1:Ny]
v_fluid = [active_cell(i, j-1) && active_cell(i, j) for i in 1:Nx, j in 1:Ny+1]
ζ_fluid = [all(active_cell(i′, j′) for i′ in i-1:i, j′ in j-1:j) for i in 1:Nx+1, j in 1:Ny+1]

u_domain(n) = interior(domain.u[n], :, :, 1)
u_immersed(n) = interior(immersed.u[n], faces, :, 1)
v_domain(n) = interior(domain.v[n], :, :, 1)
v_immersed(n) = interior(immersed.v[n], fluid, :, 1)
ζ_domain(n) = interior(domain.ζ[n], :, :, 1)
ζ_immersed(n) = interior(immersed.ζ[n], faces, :, 1)
ζ_difference(n) = ifelse.(ζ_fluid, ζ_immersed(n) .- ζ_domain(n), NaN)

Δu = [maximum(abs, (u_immersed(n) .- u_domain(n))[u_fluid]) for n in 1:Nt]
Δv = [maximum(abs, (v_immersed(n) .- v_domain(n))[v_fluid]) for n in 1:Nt]
Δζ = [maximum(abs, (ζ_immersed(n) .- ζ_domain(n))[ζ_fluid]) for n in 1:Nt]
outflow_domain = [sum(u_domain(n)[end, :]) * Ly / Ny for n in 1:Nt]
outflow_immersed = [sum(u_immersed(n)[end, :]) * Ly / Ny for n in 1:Nt]
inflow_domain = [sum(u_domain(n)[1, :]) * Ly / Ny for n in 1:Nt]
inflow_immersed = [sum(u_immersed(n)[1, :]) * Ly / Ny for n in 1:Nt]

@info @sprintf("Largest differences over t = %.1f: Δu = %.2e, Δv = %.2e, Δζ = %.2e", times[end], maximum(Δu), maximum(Δv), maximum(Δζ))

# Land columns are left blank
function ζ_immersed_with_land(n)
    ζn = Array(interior(immersed.ζ[n], :, :, 1))
    ζn[1:Nwall, :] .= NaN
    return ζn
end

fig = Figure(size = (1000, 1250))
n = Observable(Nt)
title = @lift @sprintf("Pulsating flow past a cylinder, inflow on the domain boundary and through an immersed wall, t = %.1f", times[$n])
Label(fig[0, 1:2], title; fontsize = 18)

axD = Axis(fig[1, 1]; ylabel = "y", title = "vorticity, inflow on the domain boundary", aspect = DataAspect())
hm = heatmap!(axD, xζ, yζ, @lift(ζ_domain($n)); colormap = :curl, colorrange = (-4, 4))
Colorbar(fig[1, 2], hm)

axI = Axis(fig[2, 1]; ylabel = "y", title = "vorticity, inflow through an immersed wall", aspect = DataAspect())
hm = heatmap!(axI, xi, yi, @lift(ζ_immersed_with_land($n)); colormap = :curl, colorrange = (-4, 4))
Colorbar(fig[2, 2], hm)

Δζ_range = maximum(Δζ)
axΔ = Axis(fig[3, 1]; xlabel = "x", ylabel = "y", title = "vorticity difference, immersed wall − domain boundary", aspect = DataAspect())
hm = heatmap!(axΔ, xζ, yζ, @lift(ζ_difference($n)); colormap = :balance, colorrange = (-Δζ_range, Δζ_range))
Colorbar(fig[3, 2], hm)

for ax in (axD, axI, axΔ)
    xlims!(ax, -Nwall * Δx, Lx)
end

filename = "immersed_cylinder_comparison"
save(filename * ".png", fig)
CairoMakie.record(fig, filename * ".mp4", 1:Nt; framerate = 12) do i
    n[] = i
end

fig = Figure(size = (1000, 600))
ax = Axis(fig[1, 1]; ylabel = "transport", title = "transport through the inlet and the outlet")
lines!(ax, times, inflow_domain; label = "inlet, domain boundary")
lines!(ax, times, inflow_immersed; linestyle = :dash, label = "inlet, immersed wall")
lines!(ax, times, outflow_domain; label = "outlet, domain boundary")
lines!(ax, times, outflow_immersed; linestyle = :dash, label = "outlet, immersed wall")
axislegend(ax; position = :rt)

ax = Axis(fig[2, 1]; xlabel = "time", ylabel = "largest difference", yscale = log10, title = "immersed wall − domain boundary")
lines!(ax, times, max.(Δu, 1e-16); label = "u")
lines!(ax, times, max.(Δv, 1e-16); label = "v")
lines!(ax, times, max.(Δζ, 1e-16); label = "vorticity")
axislegend(ax; position = :rb)

save(filename * "_time_series.png", fig)
