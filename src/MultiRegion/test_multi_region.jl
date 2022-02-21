using Oceananigans
using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: getregion, getdevice
using Oceananigans.BoundaryConditions
using Oceananigans.Fields: set!

grid = RectilinearGrid(CPU(), size = (12, 1, 1), topology = (Periodic, Periodic, Bounded), x = collect(0:12), y = (0, 1), z = (0, 1))

mrg = MultiRegionGrid(grid, partition = XPartition([3, 3, 3, 3]), devices = (0, 1))

field = Field((Center, Center, Center), mrg);
set!(field, (x, y ,z) -> x)

for i in 1:4
    @show getdevice(field, i)
    @show interior(getregion(field, i))
end

fill_halo_regions!(field)