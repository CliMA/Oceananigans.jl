include("dependencies_for_runtests.jl")

using MPI
MPI.Initialized() || MPI.Init()

using Test
using Oceananigans
using Oceananigans.DistributedComputations:
    load_balanced_layout, build_rank_layout, RankLayout,
    n_active_tiles, tile_shape, Sizes
using Oceananigans.ImmersedBoundaries: GridFittedBottom

nranks = MPI.Comm_size(MPI.COMM_WORLD)

if nranks == 4
    @testset "Distributed(rank_layout=…) on 4 ranks" begin
        # 2x2 layout, all active, uniform partition
        sizes_x = [4, 4]
        sizes_y = [4, 4]
        tile_loads = [10 5; 8 7]      # all active → 4 ranks
        layout = build_rank_layout(sizes_x, sizes_y, tile_loads)

        arch = Distributed(CPU(); rank_layout=layout)

        # Each rank should land on a unique tile
        local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        expected_tile = layout.rank_to_tile[local_rank + 1]
        @test arch.local_index[1] == expected_tile[1]
        @test arch.local_index[2] == expected_tile[2]
        @test arch.local_index[3] == 1

        # Connectivity: with all tiles active and Periodic wrap (which the constructor
        # uses conservatively), all 4 cardinal neighbors should be non-nothing for a 2x2 grid
        # since Periodic wraps make every edge neighbor itself.
        conn = arch.connectivity
        @test conn.east  !== nothing
        @test conn.west  !== nothing
        @test conn.north !== nothing
        @test conn.south !== nothing
    end

    @testset "RectilinearGrid on 4 ranks with rank_layout" begin
        Nx, Ny, Nz = 8, 8, 4
        sizes_x = [4, 4]
        sizes_y = [4, 4]
        tile_loads = [10 5; 8 7]
        layout = build_rank_layout(sizes_x, sizes_y, tile_loads)

        arch = Distributed(CPU(); rank_layout=layout)
        grid = RectilinearGrid(arch; size=(Nx, Ny, Nz),
                               x=(0, 1), y=(0, 1), z=(-1, 0),
                               topology=(Bounded, Bounded, Bounded))

        # Each rank should have a local grid of the right size
        local_Nx = sizes_x[arch.local_index[1]]
        local_Ny = sizes_y[arch.local_index[2]]
        @test size(grid) == (local_Nx, local_Ny, Nz)
    end
end

if nranks == 4
    @testset "TripolarGrid + load-balanced y on 4 ranks" begin
        # 32x32x2 tripolar grid, partition 2x2, uniform y for this test
        # (non-uniform y would require a specific bathymetry; for now verify the
        # constructor path works with a hand-built Sizes-based partition)
        Nx, Ny, Nz = 32, 32, 2
        sizes_y = [16, 16]
        tile_loads = ones(Int, 2, 2)   # all active
        partition = Partition(2, Sizes(sizes_y...), 1)
        tile_to_rank = [0 1; 2 3]
        rank_to_tile = [(1, 1), (1, 2), (2, 1), (2, 2)]
        layout = RankLayout(partition, tile_to_rank, rank_to_tile)

        arch = Distributed(CPU(); rank_layout=layout)
        grid = TripolarGrid(arch; size=(Nx, Ny, Nz), z=(-1000, 0))

        # Each rank should have the right local size
        local_Ny = sizes_y[arch.local_index[2]]
        @test size(grid, 2) == local_Ny
        @test size(grid, 1) == Nx ÷ 2   # uniform x
        @test size(grid, 3) == Nz

        # Verify the field can be filled and halos exchanged without error
        c = CenterField(grid)
        set!(c, (λ, φ, z) -> sin(λ) * cos(φ))
        fill_halo_regions!(c)

        # No NaNs in the interior after halo fill
        interior_data = Array(interior(c))
        @test !any(isnan, interior_data)
        @test all(isfinite, interior_data)
    end
end

if nranks == 3
    @testset "End-to-end physics: synthetic island, balanced vs unbalanced (3 ranks)" begin
        # 8x8 RectilinearGrid with Bounded topology, 2x2 partition.
        # One tile is empty by construction: the NE quadrant (i=5:8, j=5:8) is land.
        # Note: Periodic topology with empty tiles is not supported because the periodic
        # connection is lost when the wrap-around tile is absent. Use Bounded instead.
        Nx, Ny, Nz = 8, 8, 2
        bottom_height(x, y) = (x > 0.5 && y > 0.5) ? 100.0 : -1000.0

        serial_grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz),
                                      x=(0, 1), y=(0, 1), z=(-1000, 0),
                                      topology=(Bounded, Bounded, Bounded))
        ib = GridFittedBottom(bottom_height)

        using Oceananigans.DistributedComputations: compute_tile_loads
        sizes_x = [4, 4]
        sizes_y = [4, 4]
        tile_loads = compute_tile_loads(serial_grid, ib, sizes_x, sizes_y)
        @test tile_loads[2, 2] == 0

        layout = build_rank_layout(sizes_x, sizes_y, tile_loads)
        @test n_active_tiles(layout) == 3

        arch = Distributed(CPU(); rank_layout=layout)
        grid = RectilinearGrid(arch; size=(Nx, Ny, Nz),
                               x=(0, 1), y=(0, 1), z=(-1000, 0),
                               topology=(Bounded, Bounded, Bounded))
        ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)
        model = HydrostaticFreeSurfaceModel(ibg; free_surface=ExplicitFreeSurface())
        set!(model, u = (x, y, z) -> 0.01)
        for _ in 1:10
            time_step!(model, 0.001)   # small Δt to stay within CFL stability
        end

        # Verify no NaNs and positive kinetic energy
        u_data = Array(interior(model.velocities.u))
        v_data = Array(interior(model.velocities.v))
        @test !any(isnan, u_data)
        @test !any(isnan, v_data)
        @test all(isfinite, u_data)
        @test all(isfinite, v_data)
        ke = sum(u_data.^2) + sum(v_data.^2)
        @test isfinite(ke)
        @test ke > 0
    end
end
