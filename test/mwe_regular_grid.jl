using Oceananigans
using Oceananigans.Solvers: FourierTridiagonalPoissonSolver, ConjugateGradientPoissonSolver
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using LinearAlgebra: norm
using Logging

N = 4
size = (N, N, N); extent = (1, 1, 1)
Δz = 1/N
periodic_grid = RectilinearGrid(topology = (Bounded, Periodic, Bounded); size, extent)
bounded_grid  = RectilinearGrid(topology = (Bounded, Bounded, Bounded);  size, extent)

function test_conjugate_gradient_with_immersed_boundary_grid_and_open_boundaries(underlying_grid)
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(-0.3))
    cg_solver = ConjugateGradientPoissonSolver(grid, preconditioner=FourierTridiagonalPoissonSolver(underlying_grid), maxiter=10)

    U = average_mass_flux = 1
    inflow_timescale = 1e-4
    outflow_timescale = Inf
    u_boundaries = FieldBoundaryConditions(west = PerturbationAdvectionOpenBoundaryCondition(U; inflow_timescale, outflow_timescale, average_mass_flux),
                                           east = PerturbationAdvectionOpenBoundaryCondition(U; inflow_timescale, outflow_timescale, average_mass_flux))

    model = NonhydrostaticModel(grid = grid,
                                boundary_conditions = (u = u_boundaries,),
                                pressure_solver = cg_solver,
                                advection = WENO(; grid, order=5))
    u₀(x, y, z) = U + 1e-2 * rand()
    set!(model, u=u₀)

    # Test that pressure correction works with immersed boundaries
    Δt = 0.1 * minimum_zspacing(grid) / abs(U)
    simulation = Simulation(model; Δt, stop_time=5, verbose=false)
    conjure_time_step_wizard!(simulation, IterationInterval(1), cfl = 0.1)
    run!(simulation)

    return norm(interior(model.velocities.u)) / grid.Nx < 1e2 # Test that u didn't blow up
end

with_logger(NullLogger()) do
    @show znodes(periodic_grid, Face()) == znodes(bounded_grid, Face())
    @show test_conjugate_gradient_with_immersed_boundary_grid_and_open_boundaries(bounded_grid)
    @show test_conjugate_gradient_with_immersed_boundary_grid_and_open_boundaries(periodic_grid)
end
