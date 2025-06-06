include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")

using Oceananigans.Solvers: fft_poisson_solver

#####
##### Run pressure solver tests 1
#####

topos = [(Periodic, Bounded,  Bounded),
         (Bounded,  Periodic, Bounded),
         (Bounded,  Bounded,  Bounded),
         (Periodic, Periodic, Bounded)]

two_dimensional_topologies = [(Flat,     Bounded,  Bounded),
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

@testset "Poisson solvers immersed" begin
    @info "Testing immersed Poisson solvers..."

    for arch in archs
        @testset "Divergence-free solution [$(typeof(arch))]" begin
            @info "  Testing divergence-free solution [$(typeof(arch))]..."

            for topo in topos
                for N in [7, 16]

                    grids_3d = [make_random_immersed_grid(RectilinearGrid(arch, topology=topo, size=(N, N, N), extent=(1, 1, 1))),
                                make_random_immersed_grid(RectilinearGrid(arch, topology=topo, size=(1, N, N), extent=(1, 1, 1))),
                                make_random_immersed_grid(RectilinearGrid(arch, topology=topo, size=(N, 1, N), extent=(1, 1, 1))),
                                make_random_immersed_grid(RectilinearGrid(arch, topology=topo, size=(N, N, 1), extent=(1, 1, 1)))]

                    grids_2d = [make_random_immersed_grid(RectilinearGrid(arch, size=(N, N), extent=(1, 1), topology=topo))
                                for topo in two_dimensional_topologies]

                    grids = []
                    push!(grids, grids_3d..., grids_2d...)

                    for grid in grids
                        N == 7 && @info "    Testing $(topology(grid)) topology on square grids [$(typeof(arch))]..."

                        ϕ, ∇²ϕ, R = compute_pressure_solution(grid)
                        @test CUDA.@allowscalar interior(∇²ϕ) ≈ interior(R)
                        @test isapprox(mean(ϕ), 0, atol=eps(eltype(grid)))
                    end
                end
            end

            Ns = [11, 16]
            for topo in topos
                @info "    Testing $topo topology on rectangular grids with even and prime sizes [$(typeof(arch))]..."
                for Nx in Ns, Ny in Ns, Nz in Ns
                    grid = make_random_immersed_grid(RectilinearGrid(arch, topology=topo, size=(Nx, Ny, Nz), extent=(1, 1, 1)))

                    ϕ, ∇²ϕ, R = compute_pressure_solution(grid)
                    @test CUDA.@allowscalar interior(∇²ϕ) ≈ interior(R)
                    @test isapprox(mean(ϕ), 0, atol=eps(eltype(grid)))
                end
            end

            # Do a couple at Float32 (since its too expensive to repeat all tests...)
            Float32_grids = [make_random_immersed_grid(RectilinearGrid(arch, Float32, topology=(Periodic, Bounded, Bounded), size=(16, 16, 16), extent=(1, 1, 1))),
                             make_random_immersed_grid(RectilinearGrid(arch, Float32, topology=(Bounded, Bounded, Bounded), size=(7, 11, 13), extent=(1, 1, 1)))]

            for grid in Float32_grids
                ϕ, ∇²ϕ, R = compute_pressure_solution(grid)
                @test CUDA.@allowscalar interior(∇²ϕ) ≈ interior(R)
                @test isapprox(mean(ϕ), 0, atol=eps(eltype(grid)))
            end
        end
    end
end