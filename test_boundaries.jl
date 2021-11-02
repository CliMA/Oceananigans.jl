using Oceananigans
using Oceananigans.BoundaryConditions

bcs_x_u = [:periodic, :open, :flux, :grad]


grid = RegularRectilinearGrid(size = (10,), halo = (2,), x = (0, 1), topology = (Periodic, Flat, Flat))

u = Field(Face, Center, Center, grid)

u .= 1

