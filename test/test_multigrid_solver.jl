include("dependencies_for_runtests.jl")

using Oceananigans.Solvers: solve!
using Statistics

function run_poisson_equation_test(grid)
    arch = architecture(grid)
    # Solve ∇²ϕ = r
    ϕ_truth = CenterField(grid)

    # Initialize zero-mean "truth" solution with random numbers
    set!(ϕ_truth, (x, y, z) -> rand())
    parent(ϕ_truth) .-= mean(ϕ_truth)
    fill_halo_regions!(ϕ_truth)

    # Calculate Laplacian of "ϕ_truth"
    ∇²ϕ = r = CenterField(grid)
    compute_∇²!(∇²ϕ, ϕ_truth, arch, grid)

    solver = MultigridSolver(compute_∇²!, arch, grid; template_field = ϕ_truth, reltol=eps(eltype(grid)))

    # Solve Poisson equation
    ϕ_solution = CenterField(grid)
    solve!(ϕ_solution, solver, r)
    ϕ_solution .-= mean(ϕ_solution)
 
    # Diagnose Laplacian of solution
    ∇²ϕ_solution = CenterField(grid)
    compute_∇²!(∇²ϕ_solution, ϕ_solution, arch, grid)

    # Test
    extrema_tolerance = 1e-12
    std_tolerance = 1e-13

    CUDA.@allowscalar begin
        @test minimum(abs, interior(∇²ϕ_solution) .- interior(∇²ϕ)) < extrema_tolerance
        @test maximum(abs, interior(∇²ϕ_solution) .- interior(∇²ϕ)) < extrema_tolerance
        @test          std(interior(∇²ϕ_solution) .- interior(∇²ϕ)) < std_tolerance

        @test   minimum(abs, interior(ϕ_solution) .- interior(ϕ_truth)) < extrema_tolerance
        @test   maximum(abs, interior(ϕ_solution) .- interior(ϕ_truth)) < extrema_tolerance
        @test            std(interior(ϕ_solution) .- interior(ϕ_truth)) < std_tolerance
    end

    return nothing
end

@testset "MultigridSolverPoisson" begin
    @info "Testing MultigridSolver with the poisson equation..."
    arch = CPU()
    grid = RectilinearGrid(arch, size=(4, 8, 4), extent=(1, 3, 1))
    run_poisson_equation_test(grid)
end
