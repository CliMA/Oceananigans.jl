using Oceananigans

grid = RectilinearGrid(size = (20, 20, 20), extent = (1, 1, 1), halo = (7, 7, 7))
btm(x, y) = -0.5 * rand() - 0.5
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(btm))

model = HydrostaticFreeSurfaceModel(; grid, momentum_advection = WENOVectorInvariant())

time_step!(model, 1)