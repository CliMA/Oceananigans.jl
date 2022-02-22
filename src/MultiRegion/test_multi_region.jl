using Oceananigans
using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: getregion, getdevice
using Oceananigans.BoundaryConditions
using Oceananigans.Fields: set!
using BenchmarkTools

grid = RectilinearGrid(GPU(), size = (32, 32, 1), topology = (Periodic, Periodic, Bounded), x = (0, 1), y = (0, 1), z = (0, 1))

mrg = MultiRegionGrid(grid, partition = XPartition([3, 3, 3, 3]), devices = (0, 1, 2, 3))

field = Field((Center, Center, Center), mrg);
set!(field, (x, y ,z) -> x)

for i in 1:4
    @show getdevice(field, i)
    @show interior(getregion(field, i))
end

@time fill_halo_regions!(field)

field_glob = Field((Center, Center, Center), grid)

@time fill_halo_regions!(field_glob, GPU())
