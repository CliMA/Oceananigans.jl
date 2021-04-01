using Revise

using Oceananigans
using Oceananigans.Grids: short_show, long_show, show

grid1 = ZonallyStretchedRectilinearGrid(size=(8, 8, 8), x=(0,8), y=(0,8), z=(0,8), topology=(Bounded, Bounded, Bounded))
print(short_show(grid1),"\n")
#print(long_show(grid1),"\n")
#print(show(grid1),"\n")

grid2 = ZonallyStretchedRectilinearGrid(size=(8, 8), x=(0,8), y=(0,8), topology=(Bounded, Bounded, Flat))
print(short_show(grid2), "\n")

grid3 = ZonallyStretchedRectilinearGrid(size=(8), x=(0,8), topology=(Bounded, Flat, Flat))
print(short_show(grid3), "\n")