include("dependencies_for_runtests.jl")
using Oceananigans.Solvers: fft_poisson_solver, ConjugateGradientPoissonSolver, DiagonallyDominantPreconditioner
using Oceananigans.Models.NonhydrostaticModels
using Oceananigans.TimeSteppers: calculate_pressure_correction!
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using LinearAlgebra: norm
using Random: seed!

function test_conjugate_gradient_basic_functionality(arch, preconditioner_name, grid, preconditioner)
    @testset "Conjugate gradient Poisson solver unit tests [$arch, $preconditioner_name]" begin
        @info "Unit testing Conjugate gradient poisson solver with $preconditioner_name..."

        # Test the preconditioner constructor
        x = y = (0, 1)
        z = (0, 1)
        grid = RectilinearGrid(arch, size=(2, 2, 2); x, y, z)
        solver = ConjugateGradientPoissonSolver(grid, preconditioner=preconditioner)
        pressure = CenterField(grid)
        solve!(pressure, solver.conjugate_gradient_solver, solver.right_hand_side)
        @test solver isa ConjugateGradientPoissonSolver

        z = [0, 0.2, 1]
        grid = RectilinearGrid(arch, size=(2, 2, 2); x, y, z)
        solver = ConjugateGradientPoissonSolver(grid, preconditioner=preconditioner)
        pressure = CenterField(grid)
        solve!(pressure, solver.conjugate_gradient_solver, solver.right_hand_side)
        @test solver isa ConjugateGradientPoissonSolver
    end
end

function test_conjugate_gradient_default_constructor(arch)
    @testset "Conjugate gradient solver default constructor behavior [$arch]" begin
        @info "Testing CG solver default constructor and hybrid behavior..."

        # Test default constructor with regular grid
        grid = RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1))
        solver_default = ConjugateGradientPoissonSolver(grid)
        @test solver_default isa ConjugateGradientPoissonSolver

        # Test with immersed boundary grid (should use hybrid preconditioner by default)
        flat_bottom(x, y) = -0.8
        grid_immersed = ImmersedBoundaryGrid(grid, PartialCellBottom(flat_bottom))

        solver_immersed = ConjugateGradientPoissonSolver(grid_immersed)
        @test solver_immersed isa ConjugateGradientPoissonSolver

        # Test that both solvers can solve a simple problem
        for (solver, test_grid) in [(solver_default, grid), (solver_immersed, grid_immersed)]
            pressure = CenterField(test_grid)
            rhs = solver.right_hand_side

            fill!(pressure, 0.0)
            fill!(rhs, 0.0)

            # Set a simple source
            rhs[4, 4, 4] = 1.0

            # Should solve without errors
            @test_nowarn solve!(pressure, solver.conjugate_gradient_solver, rhs)

            # Should converge
            @test iteration(solver.conjugate_gradient_solver) > 0
            @test iteration(solver.conjugate_gradient_solver) <= solver.conjugate_gradient_solver.maxiter
        end
    end
end

function test_conjugate_gradient_with_nonhydrostatic_model(arch, preconditioner_name, grid, preconditioner)
    @testset "Conjugate gradient solver with NonhydrostaticModel [$arch, $preconditioner_name]" begin
        @info "Testing CG solver integration with NonhydrostaticModel using $preconditioner_name..."

        # Create a small model for testing
        grid = RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1))

        # Create model with CG pressure solver using specified preconditioner
        model = NonhydrostaticModel(
            grid = grid,
            pressure_solver = ConjugateGradientPoissonSolver(grid, preconditioner=preconditioner, maxiter=50)
        )

        @test model.pressure_solver isa ConjugateGradientPoissonSolver

        # Set up some initial conditions to create pressure solve need
        model.velocities.u[4, 4, 4] = 0.1
        model.velocities.v[4, 4, 4] = 0.05
        model.velocities.w[4, 4, 4] = -0.15

        # Store initial pressure (should be zero)
        initial_pressure = copy(Array(interior(model.pressures.pNHS)))
        initial_norm = norm(initial_pressure)
        @test initial_norm ≈ 0.0 atol=1e-15

        # Perform pressure correction (calls CG solver)
        Δt = 0.01
        calculate_pressure_correction!(model, Δt)

        # Check results
        final_pressure = Array(interior(model.pressures.pNHS))
        final_norm = norm(final_pressure)
        iterations_first = iteration(model.pressure_solver.conjugate_gradient_solver)

        @test final_norm > 0  # Should have computed non-trivial pressure
        @test iterations_first > 0
        @test iterations_first <= 50  # Should converge within max iterations

        # Test second pressure solve (uses previous pressure as initial guess)
        model.velocities.u[4, 4, 4] = 0.12
        model.velocities.v[4, 4, 4] = 0.07
        model.velocities.w[4, 4, 4] = -0.19

        pressure_before_second = copy(Array(interior(model.pressures.pNHS)))
        pressure_before_norm = norm(pressure_before_second)
        @test pressure_before_norm > 0  # Should start with non-zero pressure from first solve

        # Perform second pressure correction
        calculate_pressure_correction!(model, Δt)

        final_pressure_2 = Array(interior(model.pressures.pNHS))
        final_norm_2 = norm(final_pressure_2)
        iterations_second = iteration(model.pressure_solver.conjugate_gradient_solver)

        @test final_norm_2 > 0
        @test iterations_second > 0
        @test iterations_second <= 50

        @info "  $preconditioner_name convergence: first solve = $iterations_first iterations, second solve = $iterations_second iterations"
    end
end

function test_conjugate_gradient_with_immersed_boundary_grid_and_open_boundaries(arch, preconditioner_name, grid_base, preconditioner)
    @testset "Conjugate gradient solver with ImmersedBoundaryGrid [$arch, $preconditioner_name]" begin
        @info "Testing CG solver with ImmersedBoundaryGrid using $preconditioner_name..."
        seed!(198)  # For reproducible results

        # Create immersed boundary grid
        flat_bottom(x) = 1
        grid = ImmersedBoundaryGrid(grid_base, PartialCellBottom(flat_bottom))

        # Test that CG solver can be created for immersed boundary grid with specified preconditioner
        cg_solver = ConjugateGradientPoissonSolver(grid, preconditioner=preconditioner, maxiter=10)
        @test cg_solver isa ConjugateGradientPoissonSolver

        # Test with open boundary conditions
        U = 1
        inflow_timescale = 1e-4
        outflow_timescale = Inf
        u_boundaries = FieldBoundaryConditions(
            west = PerturbationAdvectionOpenBoundaryCondition(U; inflow_timescale, outflow_timescale),
            east = PerturbationAdvectionOpenBoundaryCondition(U; inflow_timescale, outflow_timescale),
        )

        # Create model with CG solver and boundary conditions
        model = NonhydrostaticModel(
            grid = grid,
            boundary_conditions = (u = u_boundaries,),
            pressure_solver = cg_solver,
            advection = WENO(; grid, order=5)
        )

        @test model.pressure_solver isa ConjugateGradientPoissonSolver

        # Set initial conditions
        u₀(x, z) = U + 1e-2*rand()
        set!(model, u=u₀)
        @test norm(interior(model.velocities.u)) / grid.Nx < 1e2 # Test that u didn't blow up
        @test all(interior(model.pressures.pNHS, :, 1, 1:8÷3) .== 0) # Pressure is zero inside immersed boundary

        # Test that pressure correction works with immersed boundaries
        Δt = 0.5 * minimum_zspacing(grid) / abs(U)
        calculate_pressure_correction!(model, Δt)

        final_pressure_norm = norm(Array(interior(model.pressures.pNHS)))
        iterations = iteration(model.pressure_solver.conjugate_gradient_solver)

        @test final_pressure_norm >= 0  # Norm should be non-negative
        @test iterations > 0
        @test iterations <= 10  # Should converge within max iterations

        @info "  $preconditioner_name with immersed boundaries: $iterations iterations"

        # Test that model can advance in time without blowing up
        @test_nowarn time_step!(model, Δt)
        simulation = Simulation(model; Δt = Δt, stop_time=10, verbose=false)
        conjure_time_step_wizard!(simulation, IterationInterval(1), cfl = 0.1)
        run!(simulation)
        @test norm(interior(model.velocities.u)) / grid.Nx < 1e2 # Test that u didn't blow up

        @info "  $preconditioner_name immersed boundary test completed successfully"
    end
end


@testset "Conjugate gradient Poisson solver" begin
    @info "Testing Conjugate gradient poisson solver..."
    for arch in archs
        grid_base_3d = RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1))
        grid_base_2d = RectilinearGrid(arch, topology = (Bounded, Flat, Bounded),
                                       size = (8, 8), x = (0, 10), z = (0, 3))

        # Test basic functionality with both preconditioners
        test_conjugate_gradient_basic_functionality(arch, "DiagonallyDominant", grid_base_3d, DiagonallyDominantPreconditioner())
        test_conjugate_gradient_basic_functionality(arch, "FFT",                grid_base_3d, fft_poisson_solver(grid_base_3d))

        # Test default constructor behavior
        test_conjugate_gradient_default_constructor(arch)

        # Test with NonhydrostaticModel using both preconditioners
        test_conjugate_gradient_with_nonhydrostatic_model(arch, "DiagonallyDominant", grid_base_3d, DiagonallyDominantPreconditioner())
        test_conjugate_gradient_with_nonhydrostatic_model(arch, "FFT",                grid_base_3d, fft_poisson_solver(grid_base_3d))

        # Test immersed boundary grids with both preconditioners
        test_conjugate_gradient_with_immersed_boundary_grid_and_open_boundaries(arch, "DiagonallyDominant", grid_base_2d, DiagonallyDominantPreconditioner())
        test_conjugate_gradient_with_immersed_boundary_grid_and_open_boundaries(arch, "FFT",                grid_base_2d, fft_poisson_solver(grid_base_2d))
    end
end

