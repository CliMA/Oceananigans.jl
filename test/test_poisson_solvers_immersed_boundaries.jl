include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")

using Oceananigans.Solvers: ConjugateGradientPoissonSolver

#####
##### Run pressure solver tests
#####

topos = [(Periodic, Bounded,  Bounded),
         (Bounded,  Periodic, Bounded),
         (Bounded,  Bounded,  Bounded),
         (Periodic, Periodic, Bounded),
         (Flat,     Bounded,  Bounded),
         (Bounded,  Flat,     Bounded),
         (Flat,     Periodic, Bounded),
         (Periodic, Flat,     Bounded)]

function make_random_immersed_grid(grid)
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

default_Ns() = [7, 16]
default_rect_Ns() = [11, 16]

function size_and_extent_from_topo(N, topo)
    contains_flat = any(t -> t == Flat, topo)
    # Assumes we'll have at most one instance of Flat (i.e., 2D or 3D grids only)
    if contains_flat
        return (; size=(N, N), extent=(1, 1))
    else
        return (; size=(N, N, N), extent=(1, 1, 1))
    end
end

function test_divergence_free_solution(arch, float_type, topos)
    for topo in topos
        @info "    Testing $topo topology on square grids [$(typeof(arch)), $float_type]..."
        for N in default_Ns()
            grid = make_random_immersed_grid(RectilinearGrid(arch, float_type, topology=topo; size_and_extent_from_topo(N, topo)...))

            ϕ, ∇²ϕ, R = compute_pressure_solution(grid)
            @test CUDA.@allowscalar interior(∇²ϕ) ≈ interior(R)
            @test isapprox(mean(ϕ), 0, atol=eps(eltype(grid)))
        end
    end
end

function test_divergence_free_solutions_on_rectangular_grids(arch, topos)
    Ns = default_rect_Ns()
    for topo in topos
        @info "    Testing $topo topology on rectangular grids with even and prime sizes [$(typeof(arch))]..."
        for Nx in Ns, Ny in Ns, Nz in Ns
            grid = make_random_immersed_grid(RectilinearGrid(arch, topology=topo, size=(Nx, Ny, Nz), extent=(1, 1, 1)))
            ϕ, ∇²ϕ, R = compute_pressure_solution(grid)
            @test CUDA.@allowscalar interior(∇²ϕ) ≈ interior(R)
            @test isapprox(mean(ϕ), 0, atol=eps(eltype(grid)))
        end
    end
end

@testset "Poisson solvers immersed" begin
    @info "Testing immersed Poisson solvers..."
    for arch in archs, float_type in float_types
        @testset "Divergence-free solution [$(typeof(arch)), $float_type]" begin
            @info "  Testing divergence-free solution [$(typeof(arch)), $float_type]..."
            test_divergence_free_solution(arch, float_type, topos)
            test_divergence_free_solutions_on_rectangular_grids(arch, topos)
        end
    end
end
