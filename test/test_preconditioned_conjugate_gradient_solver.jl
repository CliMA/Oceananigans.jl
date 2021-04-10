using Oceananigans.Solvers: solve!
using Statistics

function identity_operator!(b, x)
    parent(b) .= parent(x)
end

function run_identity_operator_test(arch, grid)
    b = Field(Center, Center, Center, arch, grid)

    solver = PreconditionedConjugateGradientSolver(identity_operator!, template_field = b)

    initial_guess = solution = similar(b)
    set!(initial_guess, (x, y, z) -> rand())

    solve!(initial_guess, solver, b)

    @test norm(solution) .< solver.tolerance
end

@kernel function ∇²!(∇²ϕ, ϕ, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²ϕ[i, j, k] = ∇²(i, j, k, grid, ϕ)
end

function laplacian!(∇²ϕ, ϕ)
    arch = architecture(ϕ)
    grid = ϕ.grid
    fill_halo_regions!(ϕ, arch)
    event = launch!(arch, grid, :xyz, ∇²!, ∇²ϕ, ϕ, grid, dependencies=Event(device(arch)))
    wait(device(arch), event)
    return nothing
end

function run_poisson_equation_test(arch, grid)
    # Solve ∇²ϕ = r
    ϕ_truth = Field(Center, Center, Center, arch, grid)

    # Initialize zero-mean "truth" solution with random numbers
    set!(ϕ_truth, (x, y, z) -> rand())
    parent(ϕ_truth) .-= mean(ϕ_truth)
    fill_halo_regions!(ϕ_truth, arch)

    # Calculate Laplacian of "truth"
    ∇²ϕ = r = Field(Center, Center, Center, arch, grid)
    laplacian!(∇²ϕ, ϕ_truth)

    solver = PreconditionedConjugateGradientSolver(laplacian!, template_field=ϕ_truth)

    # Solve Poisson equation
    ϕ_solution = Field(Center, Center, Center, arch, grid)
    solve!(ϕ_solution, solver, r)

    # Diagnose Laplacian of solution
    ∇²ϕ_solution = Field(Center, Center, Center, arch, grid)
    laplacian!(∇²ϕ_solution, ϕ_solution)

    # Test
    extrema_tolerance = 1e-12
    std_tolerance = 1e-14

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
        grid = RegularRectilinearGrid(size=(4, 8, 4), extent=(1, 3, 1))
        run_identity_operator_test(arch, grid)
        run_poisson_equation_test(arch, grid)
    end
end

#=
"""
    ### Not sure what to call this
    ### it is for left hand side operator in
    ### (-g∇ₕ² + 1/Δt) ϕⁿ⁺¹ = ϕⁿ / Δt + ∇ₕHU★
    #
"""
@kernel function implicit_η!(grid, f, implicit_η_f)

    #
    # g= model.free_surface.gravitational_acceleration
    #
    g = 9.81

    #
    # Δt= simulation.Δt
    #
    Δt = 9.81

    i, j, k = @index(Global, NTuple)
    @inbounds implicit_η_f[i, j] = -g * ∇²(i, j, grid, f) + f[i,j]/Δt

    # need this for 2d vertically integrated ∇²hᶜᶜᵃ
end
=#
