include("dependencies_for_runtests.jl")

using MPI

# # Distributed model tests
#
# These tests are meant to be run on 4 ranks. This script may be run
# stand-alone (outside the test environment) via
#
# mpiexec -n 4 julia --project test_distributed_models.jl
#
# provided that a few packages (like TimesDates.jl) are in your global environment.
#
# Another possibility is to use tmpi ():
#
# tmpi 4 julia --project
#
# then later:
# 
# julia> include("test_distributed_models.jl")
#
# When running the tests this way, uncomment the following line

MPI.Init()

# to initialize MPI.

using Oceananigans.DistributedComputations: reconstruct_global_grid, Partition
using Oceananigans.DistributedComputations: ZXYPermutation, ZYXPermutation

@kernel function set_distributed_solver_input!(permuted_ϕ, ϕ, ::ZYXPermutation)
    i, j, k = @index(Global, NTuple)
    @inbounds permuted_ϕ[k, j, i] = ϕ[i, j, k]
end

@kernel function set_distributed_solver_input!(permuted_ϕ, ϕ, ::ZXYPermutation)
    i, j, k = @index(Global, NTuple)
    @inbounds permuted_ϕ[k, i, j] = ϕ[i, j, k]
end

function random_divergent_source_term(grid)
    # Generate right hand side from a random (divergent) velocity field.
    Ru = XFaceField(grid)
    Rv = YFaceField(grid)
    Rw = ZFaceField(grid)
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, (x, y, z) -> rand())
    set!(Rv, (x, y, z) -> rand())
    set!(Rw, (x, y, z) -> rand())

    arch = architecture(grid)
    fill_halo_regions!(Ru)
    fill_halo_regions!(Rv)
    fill_halo_regions!(Rw)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R)
    
    return R
end

function divergence_free_poisson_solution_triply_periodic(grid_points, ranks)
    arch = Distributed(CPU(), partition=Partition(ranks...))
    local_grid = RectilinearGrid(arch, topology=(Periodic, Periodic, Periodic), size=grid_points, extent=(1, 2, 3))

    bcs = FieldBoundaryConditions(local_grid, (Center, Center, Center))
    bcs = inject_halo_communication_boundary_conditions(bcs, arch.local_rank, arch.connectivity, (Periodic, Periodic, Periodic))

    # The test will solve for ϕ, then compare R to ∇²ϕ.
    ϕ   = CenterField(local_grid, boundary_conditions=bcs)
    ∇²ϕ = CenterField(local_grid, boundary_conditions=bcs)
    R   = random_divergent_source_term(local_grid)
    
    global_grid = reconstruct_global_grid(local_grid)
    solver = DistributedFFTBasedPoissonSolver(global_grid, local_grid)

    # Solve it
    ϕc = first(solver.storage)

    launch!(arch, local_grid, :xyz, set_distributed_solver_input!, ϕc, R, solver.input_permutation)

    solve!(ϕ, solver)

    # "Recompute" ∇²ϕ
    compute_∇²!(∇²ϕ, ϕ, arch, local_grid)

    return R ≈ interior(∇²ϕ)
end

@testset "Distributed FFT-based Poisson solver" begin
    @info "  Testing 3D distributed FFT-based Poisson solver..."
    @test divergence_free_poisson_solution_triply_periodic((44, 44, 8), (1, 4, 1))
    @test divergence_free_poisson_solution_triply_periodic((44, 16, 8), (1, 4, 1))
    @test divergence_free_poisson_solution_triply_periodic((16, 44, 8), (1, 4, 1))
    @test divergence_free_poisson_solution_triply_periodic((44, 16, 8), (2, 2, 1))
    @test divergence_free_poisson_solution_triply_periodic((16, 44, 8), (2, 2, 1))

    @info "  Testing 2D distributed FFT-based Poisson solver..."
    @test divergence_free_poisson_solution_triply_periodic((44, 16, 1), (1, 4, 1))
    @test divergence_free_poisson_solution_triply_periodic((44, 16, 1), (4, 1, 1))
    @test divergence_free_poisson_solution_triply_periodic((16, 44, 1), (1, 4, 1))
    @test divergence_free_poisson_solution_triply_periodic((16, 44, 1), (4, 1, 1))

    @test_throws ArgumentError divergence_free_poisson_solution_triply_periodic((16, 44, 1), (2, 2, 1))
    @test_throws ArgumentError divergence_free_poisson_solution_triply_periodic((44, 16, 1), (2, 2, 1))
end

