using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
# using GLMakie

Nz = 32
Δz = 1 / Nz
z = 0:Δz:1
grid = RectilinearGrid(size=(128, Nz); halo=(4, 4), x=(-5, 5), z, topology=(Bounded, Flat, Bounded))
free_surface = ImplicitFreeSurface(gravitational_acceleration=10)
model = NonhydrostaticModel(; grid, free_surface)
time_step!(model, 1)

# mountain(x) = (x - 3) / 2
# grid = ImmersedBoundaryGrid(grid, GridFittedBottom(mountain))
# Fu(x, z, t) = sin(t)
# model = NonhydrostaticModel(; grid, free_surface, advection=WENO(order=5), forcing=(; u=Fu))
