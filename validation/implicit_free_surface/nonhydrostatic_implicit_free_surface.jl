using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
# using GLMakie

grid = RectilinearGrid(size=(128, 32), halo=(4, 4), x=(-5, 5), z=(0, 1), topology=(Bounded, Flat, Bounded))
free_surface = ImplicitFreeSurface(gravitational_acceleration=10)
model = NonhydrostaticModel(; grid, free_surface)

# mountain(x) = (x - 3) / 2
# grid = ImmersedBoundaryGrid(grid, GridFittedBottom(mountain))
# Fu(x, z, t) = sin(t)
# model = NonhydrostaticModel(; grid, free_surface, advection=WENO(order=5), forcing=(; u=Fu))
