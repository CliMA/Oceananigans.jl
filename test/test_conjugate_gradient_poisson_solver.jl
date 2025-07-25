include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")

using Oceananigans.Solvers: fft_poisson_solver, ConjugateGradientPoissonSolver, DiagonallyDominantPreconditioner
using Oceananigans.TimeSteppers: compute_pressure_correction!
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using Oceananigans.Grids: XYZRegularRG
using LinearAlgebra: norm
using Random: seed!

topos_3d = [(Periodic, Bounded,  Bounded),
            (Bounded,  Periodic, Bounded),
            (Bounded,  Bounded,  Bounded),
            (Periodic, Periodic, Bounded)]

topos_2d = [(Flat,     Bounded,  Bounded),
            (Bounded,  Flat,     Bounded),
            (Flat,     Periodic, Bounded),
            (Periodic, Flat,     Bounded)]

topos = vcat(topos_3d, topos_2d)

function make_random_immersed_grid(grid)
    seed!(536)
    Lz = grid.Lz
    z_top = grid.z.cᵃᵃᶠ[grid.Nz+1]
    z_bottom = grid.z.cᵃᵃᶠ[1]

    random_bottom_topography(args...) = z_bottom + rand() * abs((z_top + z_bottom) / 2)
    return ImmersedBoundaryGrid(grid, GridFittedBottom(random_bottom_topography))
end

function compute_pressure_solution(grid)
    arch = architecture(grid)
    reltol = abstol = eps(eltype(grid))
    solver = ConjugateGradientPoissonSolver(grid; reltol, abstol, maxiter=Int(1e10))
    R, U = random_divergent_source_term(grid)

    p_bcs = FieldBoundaryConditions(grid, (Center, Center, Center))
    ϕ   = CenterField(grid, boundary_conditions=p_bcs)  # "kinematic pressure"
    ∇²ϕ = CenterField(grid, boundary_conditions=p_bcs)

    # Using Δt = 1 to avoid pressure rescaling
    solve_for_pressure!(ϕ, solver, 1, U)
    compute_∇²!(∇²ϕ, ϕ, arch, grid)
    return ϕ, ∇²ϕ, R
end

function test_conjugate_gradient_basic_functionality(grid, preconditioner)
    preconditioner_name = typeof(preconditioner).name.wrapper
    @info "  Unit testing ConjugateGradientPoissonSolver with $preconditioner_name..."

    # Test the preconditioner constructor
    solver = ConjugateGradientPoissonSolver(grid, preconditioner=preconditioner)
    pressure = CenterField(grid)
    @test_nowarn solve!(pressure, solver.conjugate_gradient_solver, solver.right_hand_side)

    # Should converge
    @test iteration(solver.conjugate_gradient_solver) <= solver.conjugate_gradient_solver.maxiter
end

function test_conjugate_gradient_default_constructor(arch)
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

function test_conjugate_gradient_with_nonhydrostatic_model(grid, preconditioner)
    preconditioner_name = typeof(preconditioner).name.wrapper
    @info "  Testing CG solver integration with NonhydrostaticModel using $preconditioner_name..."
    seed!(198)  # For reproducible results

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
    compute_pressure_correction!(model, Δt)

    # Check results
    final_pressure = Array(interior(model.pressures.pNHS))
    final_norm = norm(final_pressure)
    iterations_first = iteration(model.pressure_solver.conjugate_gradient_solver)

    @test final_norm > 0  # Should have computed non-trivial pressure
    @test iterations_first > 0
    @test iterations_first <= 50  # Should converge within max iterations

    # Test second pressure solve (uses previous pressure as initial guess)
    model.velocities.u[2, 2, 2] = 0.12
    model.velocities.v[2, 2, 2] = 0.07
    model.velocities.w[2, 2, 2] = -0.19

    pressure_before_second = copy(Array(interior(model.pressures.pNHS)))
    pressure_before_norm = norm(pressure_before_second)
    @test pressure_before_norm > 0  # Should start with non-zero pressure from first solve

    # Perform second pressure correction
    compute_pressure_correction!(model, Δt)

    final_pressure_2 = Array(interior(model.pressures.pNHS))
    final_norm_2 = norm(final_pressure_2)
    iterations_second = iteration(model.pressure_solver.conjugate_gradient_solver)

    @test final_norm_2 > 0
    @test iterations_second > 0
    @test iterations_second <= 50

    @info "    $preconditioner_name convergence: first solve = $iterations_first iterations, second solve = $iterations_second iterations"
end

function test_conjugate_gradient_with_immersed_boundary_grid_and_flux_boundary_condition(underlying_grid, preconditioner)
    # See https://github.com/CliMA/Oceananigans.jl/issues/4603
    preconditioner_name = typeof(preconditioner).name.wrapper
    @info "  Testing CGSolver with ImmersedBoundaryGrid with a flux boundary condition using $preconditioner_name..."

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(-0.5))
    model = NonhydrostaticModel(; grid,
                                  pressure_solver = ConjugateGradientPoissonSolver(grid),
                                  boundary_conditions = (; u = FieldBoundaryConditions(top=FluxBoundaryCondition(1.0))))
    @test model.pressure_solver isa ConjugateGradientPoissonSolver

    time_step!(model, 1)
    @test norm(interior(model.velocities.u)) / grid.Nx < 1e2 # Test that u didn't blow up

    return nothing
end

function test_CGSolver_with_immersed_boundary_and_open_boundaries(underlying_grid, preconditioner, immersed_bottom)
    preconditioner_name = typeof(preconditioner).name.wrapper
    immersed_bottom_name = typeof(immersed_bottom).name.wrapper
    underlying_grid_name = underlying_grid isa XYZRegularRG ? "regular grid" : "stretched grid"
    @info "  Testing ConjugateGradientPoissonSolver + $preconditioner_name, on $underlying_grid_name with $immersed_bottom_name"
    seed!(198)  # For reproducible results

    grid = ImmersedBoundaryGrid(underlying_grid, immersed_bottom)
    # Main.@infiltrate
    cg_solver = ConjugateGradientPoissonSolver(grid, preconditioner=preconditioner, maxiter=100)
    @test cg_solver isa ConjugateGradientPoissonSolver

    # Test with open boundary conditions
    U = 1
    inflow_timescale = 1e-4
    outflow_timescale = Inf
    u_boundaries = FieldBoundaryConditions(west = PerturbationAdvectionOpenBoundaryCondition(U; inflow_timescale, outflow_timescale),
                                           east = PerturbationAdvectionOpenBoundaryCondition(U; inflow_timescale, outflow_timescale))

    model = NonhydrostaticModel(grid = grid,
                                boundary_conditions = (u = u_boundaries,),
                                pressure_solver = cg_solver,
                                advection = WENO(order=5))

    @test model.pressure_solver isa ConjugateGradientPoissonSolver

    u₀(x, y, z) = U + 1e-2 * rand()
    set!(model, u=u₀)

    @test norm(interior(model.velocities.u)) / grid.Nx < 1e2 # Test that u didn't blow up
    @test all(interior(model.pressures.pNHS, :, :, 1) .== 0) # Pressure is zero inside immersed boundary (assumes the bottommost Center is immersed)

    # Test that pressure correction works with immersed boundaries
    Δt = 0.1 * minimum_zspacing(grid) / abs(U)
    compute_pressure_correction!(model, Δt)

    final_pressure_norm = norm(Array(interior(model.pressures.pNHS)))
    iterations = iteration(model.pressure_solver.conjugate_gradient_solver)

    @test final_pressure_norm >= 0  # Norm should be non-negative
    @test iterations > 0
    @test iterations <= model.pressure_solver.conjugate_gradient_solver.maxiter  # Should converge within max iterations

    # Test that model can advance in time without blowing up
    @test_nowarn time_step!(model, Δt)
    @test norm(interior(model.velocities.u)) / grid.Nx < 1e2 # Test that u didn't blow up
    simulation = Simulation(model; Δt, stop_time=5, verbose=false)
    conjure_time_step_wizard!(simulation, IterationInterval(1), cfl = 0.1)
    run!(simulation)

    return norm(interior(model.velocities.u)) / grid.Nx < 1e2 # Test that u didn't blow up
end

function size_and_extent_from_topo(N, topo)
    contains_flat = any(t -> t == Flat, topo)
    if contains_flat
        return (; size=(N, N), extent=(1, 1))
    else
        return (; size=(N, N, N), extent=(1, 1, 1))
    end
end


function test_divergence_free_solution(arch, float_type, topos)
    for topo in topos
        @info "    Testing $topo topology on square grids [$(typeof(arch)), $float_type]..."
        for N in [7, 16]
            grid = make_random_immersed_grid(RectilinearGrid(arch, float_type, topology=topo; size_and_extent_from_topo(N, topo)...))
            ϕ, ∇²ϕ, R = compute_pressure_solution(grid)
            @test @allowscalar interior(∇²ϕ) ≈ interior(R)
            @test isapprox(mean(ϕ), 0, atol=eps(eltype(grid)))
        end
    end
end

function test_divergence_free_solution_on_rectangular_grids(arch, topos)
    Ns = [11, 16]
    for topo in topos
        @info "    Testing $topo topology on rectangular grids with even and prime sizes [$(typeof(arch))]..."
        for Nx in Ns, Ny in Ns, Nz in Ns
            grid = make_random_immersed_grid(RectilinearGrid(arch, topology=topo, size=(Nx, Ny, Nz), extent=(1, 1, 1)))
            ϕ, ∇²ϕ, R = compute_pressure_solution(grid)
            @test @allowscalar interior(∇²ϕ) ≈ interior(R)
            @test isapprox(mean(ϕ), 0, atol=eps(eltype(grid)))
        end
    end
end


@testset "Conjugate gradient Poisson solver" begin
    @info "Testing ConjugateGradientPoissonSolver..."
    for arch in archs
        @testset "Conjugate gradient solver default constructor behavior on [$arch]" begin
            @info "  Testing CG solver default constructor and hybrid behavior..."
            test_conjugate_gradient_default_constructor(arch)
        end

        for float_type in float_types
            @testset "Divergence-free solution on [$(typeof(arch)), $float_type]" begin
                @info "  Testing divergence-free solution [$(typeof(arch)), $float_type]..."
                test_divergence_free_solution(arch, float_type, topos)
                test_divergence_free_solution_on_rectangular_grids(arch, topos_3d)
            end
        end

        # Test more than one underlying_grid
        underlying_grids = Dict(#"regular grid"   => RectilinearGrid(arch, topology = (Bounded, Periodic, Bounded), size=(8, 4, 8), halo=(4, 4, 4), extent = (1, 1, 1)),
                                "stretched grid" => RectilinearGrid(arch, topology = (Bounded, Bounded, Bounded), size=(8, 4, 8), halo=(4, 4, 4),
                                                                    x = (0, 1), y = (0, 1), z = -1:0.125:0))

        for (underlying_grid_name, underlying_grid) in underlying_grids
            @testset "Conjugate gradient Poisson solver unit tests on a $underlying_grid_name [$(typeof(arch))]" begin
                test_conjugate_gradient_basic_functionality(underlying_grid, DiagonallyDominantPreconditioner())
                test_conjugate_gradient_basic_functionality(underlying_grid, fft_poisson_solver(underlying_grid))
            end

            @testset "Conjugate gradient solver with NonhydrostaticModel on a $underlying_grid_name [$(typeof(arch))]" begin
                test_conjugate_gradient_with_nonhydrostatic_model(underlying_grid, DiagonallyDominantPreconditioner())
                test_conjugate_gradient_with_nonhydrostatic_model(underlying_grid, fft_poisson_solver(underlying_grid))
            end

            @testset "Conjugate gradient solver with ImmersedBoundaryGrid on a $underlying_grid_name and flux boundary condition [$(typeof(arch))]" begin
                test_conjugate_gradient_with_immersed_boundary_grid_and_flux_boundary_condition(underlying_grid, DiagonallyDominantPreconditioner())
                test_conjugate_gradient_with_immersed_boundary_grid_and_flux_boundary_condition(underlying_grid, fft_poisson_solver(underlying_grid))
            end
        end

        @testset "Conjugate gradient solver with ImmersedBoundaryGrid, a GridFittedBottom and open boundary conditions [$(typeof(arch))]" begin
            bottom = GridFittedBottom(-0.6)
            for (underlying_grid_name, underlying_grid) in underlying_grids
                @info "  Testing $underlying_grid_name with different bottom types and open boundary conditions [$(typeof(arch))]..."
                @test test_CGSolver_with_immersed_boundary_and_open_boundaries(underlying_grid, DiagonallyDominantPreconditioner(), bottom)
                @test test_CGSolver_with_immersed_boundary_and_open_boundaries(underlying_grid, fft_poisson_solver(underlying_grid), bottom)
            end
        end

        @testset "Conjugate gradient solver with ImmersedBoundaryGrid, a PartialCellBottom and open boundary conditions [$(typeof(arch))]" begin
            bottom = PartialCellBottom(-0.5)
            for (underlying_grid_name, underlying_grid) in underlying_grids
                @test test_CGSolver_with_immersed_boundary_and_open_boundaries(underlying_grid, DiagonallyDominantPreconditioner(), bottom)
                @test test_CGSolver_with_immersed_boundary_and_open_boundaries(underlying_grid, fft_poisson_solver(underlying_grid), bottom)
            end
        end

        @testset "Conjugate gradient solver with ImmersedBoundaryGrid, a sinusoidal GridFittedBottom and open boundary conditions [$(typeof(arch))]" begin
            sinusoidal_bottom(x, y) = -0.8 + 0.1 * sin(2π * x) * cos(2π * y)
            bottom = GridFittedBottom(sinusoidal_bottom)
            for (underlying_grid_name, underlying_grid) in underlying_grids
                @test test_CGSolver_with_immersed_boundary_and_open_boundaries(underlying_grid, DiagonallyDominantPreconditioner(), bottom)
                @test test_CGSolver_with_immersed_boundary_and_open_boundaries(underlying_grid, fft_poisson_solver(underlying_grid), bottom)
            end
        end
    end
end
