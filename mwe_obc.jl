using Oceananigans
using Oceananigans.Grids: inactive_node
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Fields: @compute
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, immersed_inactive_node, peripheral_node
using CairoMakie

Lx, Lz = 10, 3
Nx = Nz = 8

grid_base = RectilinearGrid(topology = (Bounded, Flat, Bounded), size = (Nx, Nz), x = (0, Lx), z = (0, Lz))
flat_bottom(x) = 1
grid = ImmersedBoundaryGrid(grid_base, PartialCellBottom(flat_bottom))

squeeze(f) = interior(f, :, 1, :)

pn  = compute!(Field(KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center()))) |> squeeze
ipn = compute!(Field(KernelFunctionOperation{Face, Center, Center}(immersed_peripheral_node, grid, Face(), Center(), Center()))) |> squeeze
iin = compute!(Field(KernelFunctionOperation{Face, Center, Center}(immersed_inactive_node, grid, Face(), Center(), Center()))) |> squeeze

U = 1
inflow_timescale = 1e-4
outflow_timescale = Inf
u_boundaries = FieldBoundaryConditions(
    west   = PerturbationAdvectionOpenBoundaryCondition(U; inflow_timescale, outflow_timescale),
    east   = PerturbationAdvectionOpenBoundaryCondition(U; inflow_timescale, outflow_timescale),)

model = NonhydrostaticModel(; grid,
                              boundary_conditions = (; u = u_boundaries),
                              pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=5),
                              advection = WENO(; grid, order=5)
                             )
set!(model, u=U, enforce_incompressibility=true)

Δt = 0.5 * minimum_zspacing(grid) / abs(U)
simulation = Simulation(model; Δt = Δt, stop_time=10)

conjure_time_step_wizard!(simulation, IterationInterval(1), cfl = 0.1)

run!(simulation)

using CairoMakie

fig = Figure();
ax = Axis(fig[1, 1])
hm = heatmap!(ax, interior(model.velocities.u, :, 1, :), colorrange=(-U, U), colormap=:balance)
Colorbar(fig[1, 2], hm)
