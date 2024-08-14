using Oceananigans
using Oceananigans.Fields: OneField

grid = RectilinearGrid(size = (100, 100), halo = (7, 7), extent = (10, 10), topology = (Periodic, Periodic, Flat))

tracer_advection = WENO(; order = 9)

u = XFaceField(grid)
v = YFaceField(grid)

fill!(u, 1)
fill!(v, 1)

velocities = PrescribedVelocityFields(; u, v)

model = HydrostaticFreeSurfaceModel(; grid, 
                                      tracer_advection, 
                                      tracers = :c,
                                      velocities,
                                      buoyancy = nothing
                                    )

set!(model.tracers.c, (x, y) -> (4 < x < 6) && (4 < y < 6))
Δt = 0.02

for step in 1:20 / Δt
  time_step!(model, Δt)
end