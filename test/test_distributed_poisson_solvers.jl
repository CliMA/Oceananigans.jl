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

using Oceananigans.Distributed: reconstruct_global_grid
using Oceananigans.Distributed: ZXYPermutation, ZYXPermutation

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
    event = launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    return R
end

function divergence_free_poisson_solution_triply_periodic(grid_points, ranks)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(CPU(), ranks=ranks, topology=topo)
    local_grid = RectilinearGrid(arch, topology=topo, size=grid_points, extent=(1, 2, 3))
    bcs = FieldBoundaryConditions(local_grid, (Center, Center, Center))
    bcs = inject_halo_communication_boundary_conditions(bcs, arch.local_rank, arch.connectivity)

    # The test will solve for ϕ, then compare R to ∇²ϕ.
    ϕ   = CenterField(local_grid, boundary_conditions=bcs)
    ∇²ϕ = CenterField(local_grid, boundary_conditions=bcs)
    R   = random_divergent_source_term(local_grid)
    
    global_grid = reconstruct_global_grid(local_grid)
    solver = DistributedFFTBasedPoissonSolver(local_grid)

    # Solve it
    solver_rhs = solver.unpermuted_right_hand_side

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @show summary(solver_rhs)    
        @show size(solver_rhs)    
        @show size(R)    
    end

    solver_rhs .= R
    solve!(ϕ, solver)

    # "Recompute" ∇²ϕ
    compute_∇²!(∇²ϕ, ϕ, arch, local_grid)

    return R ≈ interior(∇²ϕ)
end

@testset "Distributed FFT-based Poisson solver" begin

    test_topologies = [(Periodic, Periodic, Periodic),
                       (Periodic, Periodic, Bounded),
                       (Periodic, Bounded, Bounded)]

    @info "  Testing 3D distributed FFT-based Poisson solver..."
    for topology in test_topologies
        for size in [(44, 32, 8), (24, 44, 16)]
            test_ranks = [(4, 1, 1)]
            topology[2] === Periodic && push!(test_ranks, (1, 4, 1), (2, 2, 1))
            for ranks in test_ranks
                @info "    Testing $topology with layout $ranks and size $size..."

                # Regular grid
                arch = MultiArch(CPU(); ranks, topology)
                local_grid = RectilinearGrid(arch; topology, size, extent=(1, 2, 3))
                @test divergence_free_poisson_solution(local_grid)

                # Vertically-stretched grid
                if topology[3] != Periodic
                    Δζ = 1 / size[3]
                    ζ = 0:Δζ:1
                    z = ζ.^2
                    local_grid = RectilinearGrid(arch; topology, size, x=(0, 1), y=(0, 2), z)
                    @test divergence_free_poisson_solution(local_grid)
                end
            end
        end
    end

    @info "  Testing 2D distributed FFT-based Poisson solver..."
    for topology in test_topologies
        test_ranks = [(4, 1, 1)]
        topology[2] === Periodic && push!(test_ranks, (1, 4, 1))
        for ranks in test_ranks
            @info "    Testing $topology with layout $ranks and size (44, 32, 1)..."
            arch = MultiArch(CPU(); ranks, topology)
            local_grid = RectilinearGrid(arch; topology, size=(44, 32, 1), extent=(1, 2, 3))
            @test divergence_free_poisson_solution(local_grid)
        end
    end

    # Test that we throw an error when attempting (x, y) decomposition of 2D problem
    topology = (Periodic, Periodic, Bounded)
    arch = MultiArch(CPU(); ranks=(2, 2, 1), topology)
    local_grid = RectilinearGrid(arch; topology, size=(44, 32, 1), extent=(1, 2, 3))
    @test_throws ArgumentError divergence_free_poisson_solution(local_grid)
    @test_throws ArgumentError divergence_free_poisson_solution(local_grid)
end
