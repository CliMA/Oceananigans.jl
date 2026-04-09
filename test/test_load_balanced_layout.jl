include("dependencies_for_runtests.jl")

using Test
using Oceananigans
using Oceananigans.DistributedComputations: RankLayout, n_active_tiles, tile_shape, is_active_tile
using Oceananigans.DistributedComputations: balanced_1d_partition
using Oceananigans.ImmersedBoundaries: GridFittedBottom
using Oceananigans.DistributedComputations: compute_active_cell_weights
using Oceananigans.DistributedComputations: compute_tile_loads
using Oceananigans.DistributedComputations: build_rank_layout
using Oceananigans.DistributedComputations: load_balanced_layout
using Oceananigans.DistributedComputations: inspect_tile_occupancy

# Test helpers for balanced_1d_partition
function _bin_sums(c, sizes)
    sums = Int[]
    idx = 1
    for n in sizes
        push!(sums, sum(c[idx:idx+n-1]))
        idx += n
    end
    return sums
end

function _brute_force_optimal_max(c, R)
    N = length(c)
    R == 1 && return sum(c)
    R == N && return maximum(c)
    best = sum(c)   # upper bound
    # Enumerate all ways to choose R-1 split points in 1:N-1
    function enumerate_splits(start, picked)
        if length(picked) == R - 1
            sizes = Int[]
            prev = 0
            for p in picked
                push!(sizes, p - prev)
                prev = p
            end
            push!(sizes, N - prev)
            m = maximum(_bin_sums(c, sizes))
            best = min(best, m)
            return
        end
        for k in start:(N - (R - 1 - length(picked)))
            enumerate_splits(k + 1, vcat(picked, k))
        end
    end
    enumerate_splits(1, Int[])
    return best
end

@testset "RankLayout struct" begin
    # 2x2 layout, top-right tile is empty
    partition    = Partition(2, 2, 1)
    tile_to_rank = [0 1; 2 -1]            # rows = ix, cols = iy
    rank_to_tile = [(1, 1), (1, 2), (2, 1)]
    layout = RankLayout(partition, tile_to_rank, rank_to_tile)

    @test n_active_tiles(layout) == 3
    @test tile_shape(layout) == (2, 2)
    @test is_active_tile(layout, 1, 1) == true
    @test is_active_tile(layout, 2, 2) == false
    @test layout.tile_to_rank[2, 2] == -1

    # Off-diagonal active tiles — guards against row/col swap in is_active_tile
    @test is_active_tile(layout, 1, 2) == true
    @test is_active_tile(layout, 2, 1) == true

    # rank_to_tile inverse — guards the "1-based index = rank+1" convention
    @test layout.rank_to_tile[1] == (1, 1)   # rank 0 → (1, 1)
    @test layout.rank_to_tile[2] == (1, 2)   # rank 1 → (1, 2)
    @test layout.rank_to_tile[3] == (2, 1)   # rank 2 → (2, 1)
end

@testset "balanced_1d_partition" begin
    # Uniform load → equal sizes
    @test balanced_1d_partition([10, 10, 10, 10, 10, 10], 3) == [2, 2, 2]

    # Skewed load: front-loaded
    sizes = balanced_1d_partition([100, 1, 1, 1, 1, 1, 1, 100], 2)
    @test sum(sizes) == 8
    @test length(sizes) == 2
    # First and last bin should each contain one of the heavy elements
    @test sizes[1] >= 1 && sizes[2] >= 1

    # Single-bin degenerate
    @test balanced_1d_partition([1, 2, 3, 4], 1) == [4]

    # As many bins as elements
    @test balanced_1d_partition([1, 2, 3, 4], 4) == [1, 1, 1, 1]

    # All-zero input → split evenly
    @test sum(balanced_1d_partition([0, 0, 0, 0, 0, 0], 3)) == 6

    # Single-element input
    @test balanced_1d_partition([42], 1) == [1]

    # Strong front-loaded check: assert the optimal split [4, 4]
    @test balanced_1d_partition([100, 1, 1, 1, 1, 1, 1, 100], 2) == [4, 4]

    # Regression for the reconstruction bug: bin sums must be ≤ L_opt
    @test balanced_1d_partition([5, 1, 1, 1], 3) == [1, 1, 2] ||
          balanced_1d_partition([5, 1, 1, 1], 3) == [1, 2, 1]   # both are optimal max=5
    @test maximum(_bin_sums([5, 1, 1, 1], balanced_1d_partition([5, 1, 1, 1], 3))) == 5
    @test maximum(_bin_sums([100, 1, 1, 1], balanced_1d_partition([100, 1, 1, 1], 3))) == 100

    # Brute-force oracle: for several small random inputs, the returned
    # partition's max bin sum must equal the optimum.
    for trial in 1:20
        N = rand(2:8)
        R = rand(1:N)
        c = rand(0:9, N)
        sizes = balanced_1d_partition(c, R)
        @test length(sizes) == R
        @test sum(sizes) == N
        got_max = maximum(_bin_sums(c, sizes))
        opt_max = _brute_force_optimal_max(c, R)
        @test got_max == opt_max
    end
end

@testset "compute_active_cell_weights — GridFittedBottom" begin
    Nx, Ny, Nz = 8, 4, 5
    grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz),
                           x=(0, 1), y=(0, 1), z=(-1, 0),
                           topology=(Bounded, Bounded, Bounded))

    # Bottom dips below the domain only on the rightmost x-column
    bottom_height(x, y) = x > 0.85 ? 1.0 : -2.0   # active everywhere except last x-column
    ib = GridFittedBottom(bottom_height)

    cx, cy, total = compute_active_cell_weights(grid, ib)

    @test length(cx) == Nx
    @test length(cy) == Ny
    @test cx[Nx] == 0                       # rightmost x-column fully immersed
    @test all(cx[1:Nx-1] .== Ny * Nz)       # everywhere else: full column
    @test all(cy .== (Nx - 1) * Nz)
    @test total == (Nx - 1) * Ny * Nz
end

@testset "compute_active_cell_weights — LatitudeLongitudeGrid" begin
    grid = LatitudeLongitudeGrid(CPU(); size=(12, 6, 3),
                                 longitude=(-180, 180), latitude=(-60, 60),
                                 z=(-1000, 0), topology=(Periodic, Bounded, Bounded))
    ib = GridFittedBottom((λ, φ) -> -500.0)   # half the depth submerged
    cx, cy, total = compute_active_cell_weights(grid, ib)
    @test total > 0
    @test sum(cx) == total
    @test sum(cy) == total
end

@testset "compute_tile_loads — empty tile detection" begin
    Nx, Ny, Nz = 8, 8, 2
    grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz),
                           x=(0, 1), y=(0, 1), z=(-1, 0),
                           topology=(Bounded, Bounded, Bounded))
    # Bottom carves out the upper-right quadrant entirely
    bottom_height(x, y) = (x > 0.5 && y > 0.5) ? 1.0 : -2.0
    ib = GridFittedBottom(bottom_height)

    sizes_x = [4, 4]
    sizes_y = [4, 4]
    tile_loads = compute_tile_loads(grid, ib, sizes_x, sizes_y)

    @test size(tile_loads) == (2, 2)
    @test tile_loads[2, 2] == 0           # upper-right tile is fully immersed
    @test tile_loads[1, 1] > 0
    @test tile_loads[1, 2] > 0
    @test tile_loads[2, 1] > 0
    # Total active cells across all tiles should equal compute_active_cell_weights total
    _, _, total = compute_active_cell_weights(grid, ib)
    @test sum(tile_loads) == total

    # Uneven partition — exercises _cell_to_tile at non-midpoint boundaries
    tile_loads_uneven = compute_tile_loads(grid, ib, [3, 5], [5, 3])
    @test size(tile_loads_uneven) == (2, 2)
    @test sum(tile_loads_uneven) == total
end

@testset "build_rank_layout — iy-fast active assignment" begin
    # 3x2 layout, the (2, 2) tile is empty
    sizes_x = [2, 2, 2]
    sizes_y = [3, 3]
    tile_loads = [10 20; 5 0; 7 9]   # (Rx, Ry) = (3, 2)
    layout = build_rank_layout(sizes_x, sizes_y, tile_loads)

    @test tile_shape(layout) == (3, 2)
    @test n_active_tiles(layout) == 5
    @test layout.tile_to_rank[2, 2] == -1

    # Pin the full iy-fast sequence so a column-major regression is caught
    @test layout.rank_to_tile == [(1, 1), (1, 2), (2, 1), (3, 1), (3, 2)]
    @test layout.tile_to_rank == [0 1; 2 -1; 3 4]

    # The partition embedded in the layout should be Sizes-based
    @test layout.partition.x isa Oceananigans.DistributedComputations.Sizes
    @test layout.partition.y isa Oceananigans.DistributedComputations.Sizes
end


@testset "load_balanced_layout — end-to-end on a synthetic basin" begin
    Nx, Ny, Nz = 16, 16, 4
    grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz),
                           x=(0, 1), y=(0, 1), z=(-1, 0),
                           topology=(Bounded, Bounded, Bounded))
    # Carve a continent in the upper-right quadrant
    bottom_height(x, y) = (x > 0.6 && y > 0.6) ? 1.0 : -2.0
    ib = GridFittedBottom(bottom_height)

    layout = load_balanced_layout(grid, ib, (4, 4); report=false)

    @test tile_shape(layout) == (4, 4)
    @test n_active_tiles(layout) <= 16
    @test n_active_tiles(layout) > 0
    # The Sizes in the resulting partition must sum to Nx, Ny
    @test sum(layout.partition.x.sizes) == Nx
    @test sum(layout.partition.y.sizes) == Ny
    # Distributed grid should be rejected
    # (Skipped here — exercised in a later MPI test, since constructing a Distributed
    # grid in this non-MPI test file would require initializing MPI.)

    # Verify the balancer actually balanced (max load within 2× of mean)
    tile_loads = compute_tile_loads(grid, ib,
                                    collect(layout.partition.x.sizes),
                                    collect(layout.partition.y.sizes))
    active_loads = filter(>(0), vec(tile_loads))
    @test maximum(active_loads) / (sum(active_loads) / length(active_loads)) < 2.0

    # After refinement, the balance should be better than 1.5:1
    @test maximum(active_loads) / (sum(active_loads) / length(active_loads)) < 1.5
end

@testset "load_balanced_layout — 2D optimization on L-shape" begin
    Nx, Ny, Nz = 16, 16, 2
    grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz),
                           x=(0, 1), y=(0, 1), z=(-1, 0),
                           topology=(Bounded, Bounded, Bounded))
    ib = GridFittedBottom((x, y) -> (x > 0.5 && y > 0.5) ? 1.0 : -2.0)

    layout = load_balanced_layout(grid, ib, (4, 4); report=false)

    sx = collect(layout.partition.x.sizes)
    sy = collect(layout.partition.y.sizes)

    # The uniform partition [4,4,4,4] is optimal for this L-shape geometry:
    # max tile load = 32 with perfect balance among active tiles.
    @test sx == [4, 4, 4, 4]
    @test sy == [4, 4, 4, 4]

    tile_loads = compute_tile_loads(grid, ib, sx, sy)
    active_loads = filter(>(0), vec(tile_loads))
    @test maximum(active_loads) == 32
    @test all(active_loads .== 32)   # perfect balance

    # 4 empty tiles in the immersed quadrant
    @test count(==(0), tile_loads) == 4
    @test n_active_tiles(layout) == 12
end

@testset "load_balanced_layout — error paths" begin
    Nx, Ny, Nz = 8, 8, 2
    grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz),
                           x=(0, 1), y=(0, 1), z=(-1, 0),
                           topology=(Bounded, Bounded, Bounded))
    ib = GridFittedBottom((x, y) -> -2.0)   # everywhere active

    # Rx > Nx
    @test_throws ArgumentError load_balanced_layout(grid, ib, (Nx + 1, 2); report=false)
    # Ry > Ny
    @test_throws ArgumentError load_balanced_layout(grid, ib, (2, Ny + 1); report=false)
    # Unsupported weight kwarg
    @test_throws ArgumentError load_balanced_layout(grid, ib, (2, 2); weight=:unknown, report=false)
    # All-immersed mask → ArgumentError
    ib_all_immersed = GridFittedBottom((x, y) -> 100.0)
    @test_throws ArgumentError load_balanced_layout(grid, ib_all_immersed, (2, 2); report=false)
end

@testset "load_balanced_layout — report banner" begin
    Nx, Ny, Nz = 16, 16, 4
    grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz),
                           x=(0, 1), y=(0, 1), z=(-1, 0),
                           topology=(Bounded, Bounded, Bounded))
    ib = GridFittedBottom((x, y) -> (x > 0.6 && y > 0.6) ? 1.0 : -2.0)

    pipe = Pipe()
    redirect_stdout(pipe) do
        load_balanced_layout(grid, ib, (4, 4); report=true)
        Base.Libc.flush_cstdio()
        flush(stdout)
    end
    close(pipe.in)
    s = read(pipe, String)
    @test occursin("load_balanced_layout", s)
    @test occursin("tile shape", s)
    @test occursin("active tiles", s)
    @test occursin("recommended launch", s)
end

using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid

@testset "load_balanced_layout — tripolar constraints" begin
    grid = TripolarGrid(CPU(); size=(16, 16, 2), z=(-1000, 0))
    ib = GridFittedBottom((λ, φ) -> -500.0)

    # odd Rx → reject
    @test_throws ArgumentError load_balanced_layout(grid, ib, (3, 2); report=false)

    # uniform x must be respected: returned partition.x must be Int, not Sizes
    layout = load_balanced_layout(grid, ib, (4, 2); report=false)
    @test layout.partition.x isa Int
    @test layout.partition.x == 4
    # y must be Sizes (balanced)
    @test layout.partition.y isa Oceananigans.DistributedComputations.Sizes
    @test sum(layout.partition.y.sizes) == 16

    # all-land north row → reject. Concentrate all active cells in the southernmost
    # latitude band so the y-balancer assigns the entire northern half to a single
    # empty bin; the tripolar specialization must reject that because the fold
    # communication cannot operate on an empty north tile row.
    bottom_antarctic_only(λ, φ) = φ > -75 ? 1.0 : -500.0
    ib_polar = GridFittedBottom(bottom_antarctic_only)
    @test_throws ArgumentError load_balanced_layout(grid, ib_polar, (4, 2); report=false)

    # Nx % Rx ≠ 0 → reject (uniform x is not exact)
    @test_throws ArgumentError load_balanced_layout(grid, ib, (6, 2); report=false)
end

@testset "inspect_tile_occupancy" begin
    grid = RectilinearGrid(CPU(); size=(16, 16, 2),
                           x=(0, 1), y=(0, 1), z=(-1, 0),
                           topology=(Bounded, Bounded, Bounded))
    ib = GridFittedBottom((x, y) -> (x > 0.5 && y > 0.5) ? 1.0 : -2.0)

    s = inspect_tile_occupancy(grid, ib, (4, 4))
    @test s isa AbstractString
    @test occursin("active tiles", s)
    @test occursin("empty tiles", s)
    @test occursin("tile shape", s)
    @test occursin("tile loads", s)

    # Pin numeric content: the L-shape with (4,4) should find [4,4,4,4] uniform
    # → 12 active tiles, 4 empty tiles.
    @test occursin("active tiles  : 12", s)
    @test occursin("empty tiles   : 4", s)
    @test occursin("tile shape    : 4 × 4", s)

    # Unsupported weight kwarg
    @test_throws ArgumentError inspect_tile_occupancy(grid, ib, (4, 4); weight=:unknown)
end

using Oceananigans.DistributedComputations: save_rank_layout, load_rank_layout

@testset "save_rank_layout / load_rank_layout — JLD2 roundtrip" begin
    grid = RectilinearGrid(CPU(); size=(8, 8, 2),
                           x=(0, 1), y=(0, 1), z=(-1, 0),
                           topology=(Bounded, Bounded, Bounded))
    ib = GridFittedBottom((x, y) -> (x > 0.5 && y > 0.5) ? 1.0 : -2.0)
    layout = load_balanced_layout(grid, ib, (4, 4); report=false)

    mktempdir() do dir
        path = joinpath(dir, "layout.jld2")
        save_rank_layout(path, layout)
        @test isfile(path)

        roundtripped = load_rank_layout(path)

        @test n_active_tiles(roundtripped) == n_active_tiles(layout)
        @test tile_shape(roundtripped) == tile_shape(layout)
        @test roundtripped.tile_to_rank == layout.tile_to_rank
        @test roundtripped.rank_to_tile == layout.rank_to_tile
        @test roundtripped.partition.x.sizes == layout.partition.x.sizes
        @test roundtripped.partition.y.sizes == layout.partition.y.sizes
    end
end

@testset "save_rank_layout / load_rank_layout — tripolar (Int x partition)" begin
    grid = TripolarGrid(CPU(); size=(16, 16, 2), z=(-1000, 0))
    ib = GridFittedBottom((λ, φ) -> -500.0)
    layout = load_balanced_layout(grid, ib, (4, 2); report=false)
    @test layout.partition.x isa Int   # sanity: the tripolar specialization

    mktempdir() do dir
        path = joinpath(dir, "tripolar_layout.jld2")
        save_rank_layout(path, layout)
        roundtripped = load_rank_layout(path)

        # Critical: the tripolar partition has a bare Int x and Sizes y,
        # which is a different concrete type than the generic Sizes/Sizes form.
        @test roundtripped.partition.x isa Int
        @test roundtripped.partition.x == layout.partition.x
        @test roundtripped.partition.y.sizes == layout.partition.y.sizes
        @test roundtripped.tile_to_rank == layout.tile_to_rank
        @test roundtripped.rank_to_tile == layout.rank_to_tile
    end
end

using Oceananigans.DistributedComputations: Distributed, NeighboringRanks, build_rank_layout
using Oceananigans.Grids: Bounded, Periodic

@testset "Distributed constructor accepts rank_layout (single-rank smoke)" begin
    if MPI.Comm_size(MPI.COMM_WORLD) == 1
        sizes_x = [8]
        sizes_y = [8]
        tile_loads = reshape([1], 1, 1)
        layout = build_rank_layout(sizes_x, sizes_y, tile_loads)

        arch = Distributed(CPU(); rank_layout=layout)
        @test arch.local_index == (1, 1, 1)
        @test arch.partition == layout.partition
        @test n_active_tiles(layout) == 1
    else
        @info "Skipping single-rank smoke test (Comm_size != 1)"
    end
end

@testset "NeighboringRanks from RankLayout — non-MPI structural" begin
    # 2x2 layout, the (2, 2) tile is empty:
    # tile_to_rank (rows = ix, cols = iy):
    #   (1,1) → 0    (1,2) → 1
    #   (2,1) → 2    (2,2) → -1
    sizes_x = [4, 4]
    sizes_y = [4, 4]
    tile_loads = [10 5; 8 0]
    layout = build_rank_layout(sizes_x, sizes_y, tile_loads)

    # Bounded topology — no periodic wraparound
    bounded_topo = (Bounded, Bounded, Bounded)

    # Rank 0 lives at (1, 1). East = (2, 1) → rank 2, North = (1, 2) → rank 1,
    # NE = (2, 2) → empty → nothing. West/South are at the global edge → nothing.
    nr00 = NeighboringRanks((1, 1, 1), layout, bounded_topo)
    @test nr00.east == 2
    @test nr00.north == 1
    @test nr00.northeast === nothing
    @test nr00.west === nothing
    @test nr00.south === nothing
    @test nr00.southwest === nothing
    @test nr00.southeast === nothing
    @test nr00.northwest === nothing

    # Rank at (2, 1): West = (1, 1) → 0, North = (2, 2) → empty → nothing,
    # East = global edge → nothing, NW = (1, 2) → 1.
    nr20 = NeighboringRanks((2, 1, 1), layout, bounded_topo)
    @test nr20.west == 0
    @test nr20.northwest == 1
    @test nr20.north === nothing
    @test nr20.east === nothing
    @test nr20.south === nothing
end

@testset "NeighboringRanks from RankLayout — periodic x wrap" begin
    # 2x2 layout, all active. Periodic x means east of (2, *) wraps to (1, *).
    sizes_x = [4, 4]
    sizes_y = [4, 4]
    tile_loads = [10 5; 8 7]    # all active
    layout = build_rank_layout(sizes_x, sizes_y, tile_loads)

    periodic_x = (Periodic, Bounded, Bounded)

    # rank at (2, 1): east wraps to (1, 1) → 0
    nr20 = NeighboringRanks((2, 1, 1), layout, periodic_x)
    @test nr20.east == 0     # wrapped
    @test nr20.west == 0     # adjacent
    # Quirk of Rx=2 + periodic: east wraps to (1,1)=0 and west is also (1,1)=0.
    # Both east and west point to the same rank.
end

@testset "NeighboringRanks from RankLayout — periodic wrap onto empty tile" begin
    # 2x2 layout with both top tiles empty.
    # Periodic x: east of (2, 1) wraps to (1, 1) → 0 (good),
    # north of (2, 1) is (2, 2) → empty → nothing,
    # northeast wraps to (1, 2) → empty → nothing,
    # northwest adjacent (1, 2) → empty → nothing.
    sizes_x = [4, 4]
    sizes_y = [4, 4]
    tile_loads = [10 0; 8 0]   # both top tiles empty
    layout = build_rank_layout(sizes_x, sizes_y, tile_loads)

    periodic_x = (Periodic, Bounded, Bounded)

    nr20 = NeighboringRanks((2, 1, 1), layout, periodic_x)
    @test nr20.east == 0              # wraps to (1, 1) → 0
    @test nr20.west == 0              # adjacent (1, 1) → 0
    @test nr20.north === nothing      # (2, 2) is empty
    @test nr20.northeast === nothing  # wraps to (1, 2) → empty
    @test nr20.northwest === nothing  # adjacent (1, 2) → empty
end

@testset "RectilinearGrid built from Distributed(rank_layout=...) (single-rank)" begin
    if MPI.Comm_size(MPI.COMM_WORLD) == 1
        sizes_x = [8]
        sizes_y = [8]
        tile_loads = reshape([1], 1, 1)
        layout = build_rank_layout(sizes_x, sizes_y, tile_loads)
        arch = Distributed(CPU(); rank_layout=layout)
        grid = RectilinearGrid(arch; size=(8, 8, 4),
                               x=(0, 1), y=(0, 1), z=(-1, 0),
                               topology=(Bounded, Bounded, Bounded))
        @test size(grid) == (8, 8, 4)
    else
        @info "Skipping single-rank grid smoke test (Comm_size != 1)"
    end
end

@testset "placeholder — removed is_fold_aware_topology (no longer needed)" begin
    @test true
end
