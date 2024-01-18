using Oceananigans

grid = RectilinearGrid(size = (64, 64), 
                       halo = (9, 9), 
                   topology = (Periodic, Periodic, Flat),
                          x = (-1, 1), 
                          y = (-1, 1))

tracer_advection = Oceananigans.MPData(grid; iterations = 1) #WENO(; order = 9)

U = 1
V = 1

model = HydrostaticFreeSurfaceModel(; grid, 
                                      velocities = PrescribedVelocityFields(; u = (x, y, z) -> U, v = (x, y, z) -> V),
                                      tracers = :c,
                                      buoyancy = nothing,
                                      tracer_advection)

cᵢ(x, y) = x^2 + y^2 < 0.25 ? 1 : 0
ci = CenterField(grid)
set!(model, c = cᵢ)
set!(ci, cᵢ)

CFL = 0.2

Δx = grid.Δxᶜᵃᵃ
Δy = grid.Δxᶜᵃᵃ

Δt = CFL / (U / Δx + V / Δy)

simulation = Simulation(model; Δt, stop_iteration = 1)

run!(simulation)