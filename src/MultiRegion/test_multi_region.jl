using Oceananigans
using Oceananigans.MultiRegion
using Oceananigans.Fields: set!

grid = RectilinearGrid(CPU(), size = (12, 1), topology = (Periodic, Periodic, Flat), x = collect(0:12), y = (0, 1))

mrg = MultiRegionGrid(grid, partition = XPartition([3, 3, 3, 3]), devices = (0, 1))

field = Field((Center, Center, Center), mrg);
set!(field, (x, y ,z) -> x)

for i in 1:4
    @show getdevice(field, i)
    @show interior(getregion(field, i))
end