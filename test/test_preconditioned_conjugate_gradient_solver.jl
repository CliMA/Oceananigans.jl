include("dependencies_for_runtests.jl")

using Oceananigans.Solvers: solve!
using Statistics

function identity_operator!(b, x)
    parent(b) .= parent(x)
    return nothing
end

function run_identity_operator_test(grid)
    b = CenterField(grid)

    solver = PreconditionedConjugateGradientSolver(identity_operator!, template_field = b)

    initial_guess = solution = similar(b)
    set!(initial_guess, (x, y, z) -> rand())

    solve!(initial_guess, solver, b)

    @test norm(solution) .< solver.tolerance
end

function run_poisson_equation_test(grid)
    arch = architecture(grid)
    # Solve ∇²ϕ = r
    ϕ_truth = CenterField(grid)

    # Initialize zero-mean "truth" solution with random numbers
    set!(ϕ_truth, (x, y, z) -> rand())
    parent(ϕ_truth) .-= mean(ϕ_truth)
    fill_halo_regions!(ϕ_truth)

    # Calculate Laplacian of "truth"
    ∇²ϕ = r = CenterField(grid)
    compute_∇²!(∇²ϕ, ϕ_truth, arch, grid)

    solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=ϕ_truth)

    # Solve Poisson equation
    ϕ_solution = CenterField(grid)
    solve!(ϕ_solution, solver, r, arch, grid)

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

@testset "PreconditionedConjugateGradientSolver" begin
    for arch in archs
        @info "Testing PreconditionedConjugateGradientSolver [$(typeof(arch))]..."
        grid = RectilinearGrid(arch, size=(4, 8, 4), extent=(1, 3, 1))
        run_identity_operator_test(grid)
        run_poisson_equation_test(grid)
    end
end
