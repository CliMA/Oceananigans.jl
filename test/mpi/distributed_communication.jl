using MPI

MPI.Init(threadlevel=:multiple)

include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.DistributedComputations: TransposableField, transpose_z_to_y!, transpose_y_to_z!,
                                            transpose_y_to_x!, transpose_x_to_y!

# Every cell of a field on a grid with unit spacing holds a value unique to its global (i, j),
# so a halo filled from the wrong neighbor, or the wrong corner of the right neighbor, is detected.
global_position_value(x, y) = x + 1000y

function halo_matches_global_positions(c)
    grid = c.grid
    arch = architecture(grid)
    Nx, Ny = size(grid, 1) * arch.ranks[1], size(grid, 2) * arch.ranks[2]
    Hx, Hy, Hz = halo_size(grid)
    ox = (arch.local_index[1] - 1) * size(grid, 1)
    oy = (arch.local_index[2] - 1) * size(grid, 2)
    P = Array(parent(c))
    kᵢ = Hz + 1
    correct = true
    for j in axes(P, 2), i in axes(P, 1)
        gi = mod1(ox + i - Hx, Nx)
        gj = mod1(oy + j - Hy, Ny)
        correct &= P[i, j, kᵢ] == global_position_value(gi - 1/2, gj - 1/2)
    end
    return correct
end

function filled_halos_are_correct(partition; N = 16, Nz = 4)
    arch = Distributed(child_arch; partition)
    grid = RectilinearGrid(arch; size = (N, N, Nz), extent = (N, N, Nz), halo = (3, 3, 3),
                           topology = (Periodic, Periodic, Bounded))
    c = CenterField(grid)
    set!(c, (x, y, z) -> global_position_value(x, y))
    fill_halo_regions!(c)
    return halo_matches_global_positions(c)
end

function reduction_is_global(partition; N = 16, Nz = 4)
    arch = Distributed(child_arch; partition)
    grid = RectilinearGrid(arch; size = (N, N, Nz), extent = (N, N, Nz), topology = (Periodic, Periodic, Bounded))
    c = CenterField(grid)
    set!(c, (x, y, z) -> global_position_value(x, y))
    return maximum(c) == global_position_value(N - 1/2, N - 1/2) && minimum(c) == global_position_value(1/2, 1/2)
end

staging_options = child_arch isa GPU ? (true, false) : (false,)
staging_options = MPI.has_cuda() || !(child_arch isa GPU) ? staging_options : (true,)

@testset "Distributed halo values and reductions [$(summary(child_arch)), host staging = $staged]" for staged in staging_options
    ENV["OCEANANIGANS_HOST_STAGED_COMMUNICATION"] = staged
    for partition in (Partition(4, 1), Partition(1, 4), Partition(2, 2))
        @test filled_halos_are_correct(partition)
        @test reduction_is_global(partition)
    end
    delete!(ENV, "OCEANANIGANS_HOST_STAGED_COMMUNICATION")
end

# Two disjoint grids, each on a 2-rank sub-communicator of the 4 ranks. The transposes must split
# each grid's own communicator; splitting MPI.COMM_WORLD would mix ranks of the two grids.
@testset "Transposes on a sub-communicator [$(summary(child_arch))]" begin
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    communicator = MPI.Comm_split(MPI.COMM_WORLD, world_rank ÷ 2, world_rank)
    arch = Distributed(child_arch; partition = Partition(2, 1), communicator)
    N, Nz = 8, 4
    grid = RectilinearGrid(arch; size = (N, N, Nz), extent = (1, 1, 1), topology = (Periodic, Periodic, Bounded))
    ϕ = Field((Center(), Center(), Center()), grid, ComplexF64)
    set!(ϕ, (x, y, z) -> x + 10y + 100z + world_rank ÷ 2)  # differs between the two grids
    T = TransposableField(ϕ)
    set!(T.zfield, ϕ)
    transpose_z_to_y!(T)
    transpose_y_to_x!(T)
    transpose_x_to_y!(T)
    transpose_y_to_z!(T)
    @test Array(interior(T.zfield)) == Array(interior(ϕ))
    MPI.free(communicator)
end
