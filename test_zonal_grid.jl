using Revise

using Oceananigans
using Oceananigans.Grids: short_show, long_show, show

grid1 = ZonallyStretchedRectilinearGrid(size=(8, 8, 8), x=(0,8), y=(0,8), z=(0,8), topology=(Bounded, Bounded, Bounded))
show(grid1)

grid2 = ZonallyStretchedRectilinearGrid(size=(8, 8), x=(0,8), y=(0,8), topology=(Bounded, Bounded, Flat))
show(grid2)

grid3 = ZonallyStretchedRectilinearGrid(size=(8), x=(0,8), topology=(Bounded, Flat, Flat))
show(grid3)

#grid0 = RegularRectilinearGrid(size=(8, 8, 8), x=(0, 8), y=(0, 8), z=(0, 8), topology=(Bounded, Bounded, Bounded))
