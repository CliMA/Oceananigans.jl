include("dependencies_for_runtests.jl")

using Oceananigans.Utils: get_active_cells_map
using Oceananigans.ImmersedBoundaries: active_cells_per_column
using Oceananigans.DistributedComputations: partition_1d, ends_to_sizes

sizes = [ (6, 6, 3) ]
halos = [ (4, 4, 4) ]
longitudes = [ (0, 360) ]
latitudes = [ (-80, 85) ]
xs = [ (-10, 10) ]
ys = [ (-10, 10) ]
zs = [ (-10, 0) ]
topology_types = (Bounded, Periodic, Flat)
topologies = [ (Bounded, Periodic, Bounded) ]

latlong_constructors = [
  arch -> LatitudeLongitudeGrid(arch;
                                size,
                                halo,
                                longitude,
                                latitude,
                                z
                                )
  for (size, halo, longitude, latitude, z) in
    Iterators.product(sizes, halos, longitudes, latitudes, zs)
]

rectilinear_constructors = [
  arch -> RectilinearGrid(arch;
                          size,
                          x,
                          y,
                          z,
                          halo,
                          topology
                          )
  for (size, halo, x, y, z, topology) in
    Iterators.product(sizes, halos, xs, ys, zs, topologies)
]

ib_constructors = [
  bottom_height -> GridFittedBottom(bottom_height),
  bottom_height -> PartialCellBottom(bottom_height)
]

strategies = [nothing, SimplifiedGeneralisedBlockDistribution(), GeneralisedBlockDistribution()]

partitions = [Partition(x, y) for (x,y) in Iterators.product([1,2,4],[1,2,4])]

grid_constructors = Iterators.flatten([latlong_constructors, rectilinear_constructors])

@testset "Total active cells consistent" for (arch, grid_constructor, ib_constructor) in
    Iterators.product(archs, grid_constructors, ib_constructors)

  underlying_grid = grid_constructor(arch)

  Nx, Ny, Nz = size(underlying_grid)

  bottom_height = -30.0 .* rand(Float64, (Nx, Ny)) .+ 15.0

  ib = ib_constructor(bottom_height)
  immersed_grid = ImmersedBoundaryGrid(underlying_grid, ib; active_cells_map = true)

  active_cells_map = immersed_grid.interior_active_cells
  active_cells_count = isnothing(active_cells_map) ? Nx*Ny*Nz : length(active_cells_map)

  @test sum(active_cells_per_column(underlying_grid, ib)) == active_cells_count

end

@testset "Partitioning consistent" begin

  @testset "1d partitioning consistent" for (len, ranks) in Iterators.product((10, 100, 1000), (2,4,8))
    weights = rand(len)

    partitions = partition_1d(weights, ranks)

    sizes = ends_to_sizes(partitions)

    @test sum(sizes) == len
  end

  @testset "Grid partitioning consistent" for (arch, gridc, ibc, strategy, partition) in
      Iterators.product(archs, grid_constructors, ib_constructors, strategies, partitions)

    grid = gridc(arch)

    Nx, Ny, Nz = size(underlying_grid)

    bottom_height = -30.0 .* rand(Float64, (Nx, Ny)) .+ 15.0
    ib = ibc(bottom_height)
    weight_map = create_weight_map(grid, ib)
    balanced_partition = create_balanced_partition(strategy, partition, weight_map)

    @test sum(balanced_partition.x) == Nx
    @test sum(balanced_partition.y) == Ny

    @test length(balanced_partition.x) == partition.x
    @test length(balanced_partition.y) == partition.y

  end
end
