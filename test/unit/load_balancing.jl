include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Utils: get_active_cells_map
using Oceananigans.DistributedComputations: partition_1d, ends_to_sizes, create_cost_map, Sizes,
                                            GeneralizedBlockDistribution, SimplifiedGeneralizedBlockDistribution

sizes = [ (60, 60, 30) ]
halos = [ (4, 4, 4) ]
longitudes = [ (0, 360) ]
latitudes = [ (-40, 45) ]
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

tripolar_constructors = [
  arch -> TripolarGrid(arch;
                       size,
                       z,
                       halo)
  for (size, halo, x, y, z) in
    Iterators.product(sizes, halos, xs, ys, zs)
]

ib_constructors = [
  bottom_height -> GridFittedBottom(bottom_height),
  bottom_height -> PartialCellBottom(bottom_height)
]

strategies = [SimplifiedGeneralizedBlockDistribution(), GeneralizedBlockDistribution()]

partitions = [Partition(x, y) for (x,y) in Iterators.product([1,2,4],[1,2,4])]

grid_constructors = Iterators.flatten([latlong_constructors, rectilinear_constructors, tripolar_constructors])

@testset "Total active cells consistent" begin
    for (arch, grid_constructor, ib_constructor) in Iterators.product(archs, grid_constructors, ib_constructors)

        underlying_grid = grid_constructor(arch)

        Nx, Ny, Nz = size(underlying_grid)

        bottom_height = -30.0 .* rand(Float64, (Nx, Ny)) .+ 15.0

        ib = ib_constructor(bottom_height)
        immersed_grid = ImmersedBoundaryGrid(underlying_grid, ib; active_cells_map = true)

        active_cells_map = immersed_grid.interior_active_cells
        active_cells_count = isnothing(active_cells_map) ? Nx*Ny*Nz : length(active_cells_map)
        @testset "$arch, $(nameof(typeof(underlying_grid))), $(nameof(typeof(ib))), size=($Nx, $Ny, $Nz)" begin
          @test sum(create_cost_map(underlying_grid, ib)) == active_cells_count
        end
    end
end

@testset "Partitioning consistent" begin

  @testset "1d - len:$len, ranks:$ranks" for (len, ranks) in Iterators.product((10, 100, 1000), (2,4,8))
    costs = rand(len)

    partitions = partition_1d(costs, ranks)

    sizes = ends_to_sizes(partitions)

    @test sum(sizes) == len
  end

  @testset "2d - $arch, $gridc, $ibc, $strategy, $partition" for (arch, gridc, ibc, strategy, partition) in
      Iterators.product(archs, grid_constructors, ib_constructors, strategies, partitions)

    underlying_grid = gridc(arch)

    Nx, Ny, Nz = size(underlying_grid)

    bottom_height = -30.0 .* rand(Float64, (Nx, Ny)) .+ 15.0
    ib = ibc(bottom_height)
    cost_map = create_cost_map(underlying_grid, ib)
    balanced_partition = create_balanced_partition(strategy, partition, cost_map)

    # If partitioning in x
    if !isnothing(partition.x)
      @test balanced_partition.x isa Sizes
      @test length(balanced_partition.x.sizes) == partition.x
      @test sum(balanced_partition.x.sizes) == Nx
    end

    # If partitioning in y
    if !isnothing(partition.y)
      @test balanced_partition.y isa Sizes
      @test length(balanced_partition.y.sizes) == partition.y
      @test sum(balanced_partition.y.sizes) == Ny
    end

  end
end
