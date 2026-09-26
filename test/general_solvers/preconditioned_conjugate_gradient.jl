include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Solvers: solve!
using Statistics

function identity_operator!(b, x)
    parent(b) .= parent(x)
    return nothing
end

function run_identity_operator_test(grid)
    b = CenterField(grid)
    solver = ConjugateGradientSolver(identity_operator!, template_field = b, reltol=0, abstol=10*sqrt(eps(eltype(grid))))
    initial_guess = solution = similar(b)
    set!(initial_guess, (x, y, z) -> rand())

    solve!(initial_guess, solver, b)

    @test norm(solution) .< solver.abstol
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

    solver = ConjugateGradientSolver(compute_∇²!, template_field=ϕ_truth, reltol=eps(eltype(grid)), maxiter=Int(1e10))

    # Solve Poisson equation
    ϕ_solution = CenterField(grid)
    solve!(ϕ_solution, solver, r, arch, grid)
    fill_halo_regions!(ϕ_solution)

    # Diagnose Laplacian of solution
    ∇²ϕ_solution = CenterField(grid)
    compute_∇²!(∇²ϕ_solution, ϕ_solution, arch, grid)

    # Test
    extrema_tolerance = 1e-12
    std_tolerance = 1e-13

    @allowscalar begin
        @test minimum(abs, interior(∇²ϕ_solution) .- interior(∇²ϕ)) < extrema_tolerance
        @test maximum(abs, interior(∇²ϕ_solution) .- interior(∇²ϕ)) < extrema_tolerance
        @test          std(interior(∇²ϕ_solution) .- interior(∇²ϕ)) < std_tolerance

        @test   minimum(abs, interior(ϕ_solution) .- interior(ϕ_truth)) < extrema_tolerance
        @test   maximum(abs, interior(ϕ_solution) .- interior(ϕ_truth)) < extrema_tolerance
        @test            std(interior(ϕ_solution) .- interior(ϕ_truth)) < std_tolerance
    end

    return nothing
end

function run_operator_count_test(grid)
    arch = architecture(grid)
    applications = Ref(0)

    function counting_∇²!(∇²ϕ, ϕ, arch, grid)
        applications[] += 1
        return compute_∇²!(∇²ϕ, ϕ, arch, grid)
    end

    ϕ_truth = CenterField(grid)
    set!(ϕ_truth, (x, y, z) -> rand())
    parent(ϕ_truth) .-= mean(ϕ_truth)
    r = CenterField(grid)
    compute_∇²!(r, ϕ_truth, arch, grid)

    solver = ConjugateGradientSolver(counting_∇²!, template_field=ϕ_truth, reltol=eps(eltype(grid)), maxiter=Int(1e10))
    solve!(CenterField(grid), solver, r, arch, grid)

    # one application per iteration plus one for the initial residual
    @test solver.iteration > 1
    @test applications[] == solver.iteration + 1

    return nothing
end

# The unpreconditioned conjugate gradient iteration with the scalars ρ, α, and β computed on the host
function reference_conjugate_gradient!(x, linear_operation!, b, iterations, args...)
    r = similar(b)
    p = similar(b)
    q = similar(b)

    linear_operation!(q, x, args...)
    parent(r) .= parent(b) .- parent(q)

    ρⁱ⁻¹ = zero(eltype(b))

    for iteration in 0:iterations-1
        ρ = dot(r, r)

        if iteration == 0
            parent(p) .= parent(r)
        else
            β = ρ / ρⁱ⁻¹
            parent(p) .= parent(r) .+ β .* parent(p)
        end

        linear_operation!(q, p, args...)
        α = ρ / dot(p, q)

        parent(x) .+= α .* parent(p)
        parent(r) .-= α .* parent(q)

        ρⁱ⁻¹ = ρ
    end

    return x
end

function run_reference_iterates_test(grid)
    arch = architecture(grid)

    ϕ_truth = CenterField(grid)
    set!(ϕ_truth, (x, y, z) -> rand())
    parent(ϕ_truth) .-= mean(ϕ_truth)
    b = CenterField(grid)
    compute_∇²!(b, ϕ_truth, arch, grid)

    # With zero tolerances the solver performs exactly `maxiter` iterations
    iterations = 5
    solver = ConjugateGradientSolver(compute_∇²!, template_field=b, reltol=0, abstol=0, maxiter=iterations)
    x = CenterField(grid)
    solve!(x, solver, b, arch, grid)

    x_reference = CenterField(grid)
    reference_conjugate_gradient!(x_reference, compute_∇²!, b, iterations, arch, grid)

    @test solver.iteration == iterations
    @test Array(interior(x)) ≈ Array(interior(x_reference))

    return nothing
end

@testset "ConjugateGradientSolver" begin
    for arch in archs
        @info "Testing ConjugateGradientSolver [$(typeof(arch))]..."
        grid = RectilinearGrid(arch, size=(4, 8, 4), extent=(1, 3, 1))
        run_identity_operator_test(grid)
        run_poisson_equation_test(grid)
        run_operator_count_test(grid)
        run_reference_iterates_test(grid)
    end
end
