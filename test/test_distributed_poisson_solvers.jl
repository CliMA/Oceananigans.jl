using MPI
MPI.Init()

# Make sure results are 
# reproducible
using Random
Random.seed!(1234)

include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")

# # Distributed Poisson Solver tests
#
# These tests are meant to be run on 4 ranks. This script may be run
# stand-alone (outside the test environment) via
#
# mpiexec -n 4 julia --project test_distributed_poisson_solver.jl
#
# provided that a few packages (like TimesDates.jl) are in your global environment.
#
# Another possibility is to use tmpi ():
#
# tmpi 4 julia --project
#
# then later:
# 
# julia> include("test_distributed_poisson_solver.jl")
#
# When running the tests this way, uncomment the following line

# to initialize MPI.

using Oceananigans.DistributedComputations: reconstruct_global_grid, DistributedGrid, Partition
using Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

function random_divergent_source_term(grid::DistributedGrid)
    arch = architecture(grid)
    default_bcs = FieldBoundaryConditions()

    u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
    v_bcs = regularize_field_boundary_conditions(default_bcs, grid, :v)
    w_bcs = regularize_field_boundary_conditions(default_bcs, grid, :w)

    u_bcs = inject_halo_communication_boundary_conditions(u_bcs, arch.local_rank, arch.connectivity, topology(grid))
    v_bcs = inject_halo_communication_boundary_conditions(v_bcs, arch.local_rank, arch.connectivity, topology(grid))
    w_bcs = inject_halo_communication_boundary_conditions(w_bcs, arch.local_rank, arch.connectivity, topology(grid))

    Ru = XFaceField(grid, boundary_conditions=u_bcs)
    Rv = YFaceField(grid, boundary_conditions=v_bcs)
    Rw = ZFaceField(grid, boundary_conditions=w_bcs)
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(size(Ru)...))
    set!(Rv, rand(size(Rv)...))
    set!(Rw, rand(size(Rw)...))

    fill_halo_regions!(Ru)
    fill_halo_regions!(Rv)
    fill_halo_regions!(Rw)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R)
    
    return R, U
end

function divergence_free_poisson_solution(grid_points, ranks, topo)
    arch = Distributed(CPU(), partition=Partition(ranks...))
    local_grid = RectilinearGrid(arch, topology=topo, size=grid_points, extent=(2π, 2π, 2π))

    # The test will solve for ϕ, then compare R to ∇²ϕ.
    ϕ   = CenterField(local_grid)
    ∇²ϕ = CenterField(local_grid)
    R, U = random_divergent_source_term(local_grid)

    global_grid = reconstruct_global_grid(local_grid)
    solver = DistributedFFTBasedPoissonSolver(global_grid, local_grid)
    
    # Using Δt = 1.
    solve_for_pressure!(ϕ, solver, 1, U)

    # "Recompute" ∇²ϕ
    compute_∇²!(∇²ϕ, ϕ, arch, local_grid)

    return Array(interior(∇²ϕ)) ≈ Array(R)
end

@testset "Distributed FFT-based Poisson solver" begin
    for topology in ((Periodic, Periodic, Periodic), 
                     (Periodic, Periodic, Bounded),
                     (Periodic, Bounded, Bounded),
                     (Bounded, Bounded, Bounded))
                    
        @info "  Testing 3D distributed FFT-based Poisson solver with topology $topology..."
        @show @test divergence_free_poisson_solution((44, 44, 8), (4, 1, 1), topology)
        @show @test divergence_free_poisson_solution((16, 44, 8), (4, 1, 1), topology)
        @show @test divergence_free_poisson_solution((44, 44, 8), (1, 4, 1), topology)
        @show @test divergence_free_poisson_solution((44, 16, 8), (1, 4, 1), topology)
        @show @test divergence_free_poisson_solution((16, 44, 8), (1, 4, 1), topology)
        @show @test divergence_free_poisson_solution((22, 44, 8), (2, 2, 1), topology)
        @show @test divergence_free_poisson_solution((44, 22, 8), (2, 2, 1), topology)

        @info "  Testing 2D distributed FFT-based Poisson solver with topology $topology..."
        @show @test divergence_free_poisson_solution((44, 16, 1), (4, 1, 1), topology)
        @show @test divergence_free_poisson_solution((16, 44, 1), (4, 1, 1), topology)
    end
    # for topology in ((Periodic, Periodic, Bounded),
    #                  (Periodic, Bounded, Bounded),
    #                  (Bounded, Bounded, Bounded))
    #     @info "  Testing 3D distributed Fourier Tridiagonal Poisson solver with topology $topology..."
    #     @test divergence_free_poisson_tridiagonal_solution((44, 11, 8), (1, 4, 1), topology)
    #     @test divergence_free_poisson_tridiagonal_solution((44,  4, 8), (1, 4, 1), topology)
    #     @test divergence_free_poisson_tridiagonal_solution((16, 11, 8), (1, 4, 1), topology)
    #     @test divergence_free_poisson_tridiagonal_solution((22,  8, 8), (2, 2, 1), topology)
    #     @test divergence_free_poisson_tridiagonal_solution(( 8, 22, 8), (2, 2, 1), topology)
    #     @test divergence_free_poisson_tridiagonal_solution((44, 11, 8), (1, 4, 1), topology)
    #     @test divergence_free_poisson_tridiagonal_solution((44,  4, 8), (1, 4, 1), topology)
    #     @test divergence_free_poisson_tridiagonal_solution((16, 11, 8), (1, 4, 1), topology)
    #     @test divergence_free_poisson_tridiagonal_solution((22,  8, 8), (2, 2, 1), topology)
    #     @test divergence_free_poisson_tridiagonal_solution(( 8, 22, 8), (2, 2, 1), topology)
    # end
end
