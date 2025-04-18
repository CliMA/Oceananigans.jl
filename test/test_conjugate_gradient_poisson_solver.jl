include("dependencies_for_runtests.jl")
using Oceananigans.Solvers: fft_poisson_solver, KrylovPoissonSolver

@testset "Conjugate gradient Poisson solver" begin
    @info "Testing Conjugate gradient poisson solver..."
    for arch in archs
        @testset "Conjugate gradient Poisson solver unit tests [$arch]" begin
            @info "Unit testing Conjugate gradient poisson solver..."

            # Test the generic fft_poisson_solver constructor
            x = y = (0, 1)
            z = (0, 1)
            grid = RectilinearGrid(arch, size=(2, 2, 2); x, y, z)
            solver = KrylovPoissonSolver(grid, preconditioner=fft_poisson_solver(grid))
            pressure = CenterField(grid)
            solve!(pressure, solver.conjugate_gradient_solver, solver.right_hand_side)
            @test solver isa KrylovPoissonSolver

            z = [0, 0.2, 1]
            grid = RectilinearGrid(arch, size=(2, 2, 2); x, y, z)
            solver = KrylovPoissonSolver(grid, preconditioner=fft_poisson_solver(grid))
            pressure = CenterField(grid)
            solve!(pressure, solver.conjugate_gradient_solver, solver.right_hand_side)
            @test solver isa KrylovPoissonSolver
        end
    end
end

