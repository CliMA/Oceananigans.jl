using Oceananigans
using Oceananigans.Units
using Printf

arch = CPU()

Nx, Ny, Nz = 64, 64, 4
Δ = 20kilometers
H = 200meters

grid = LambertConformalConicGrid(arch, Float64;
                                 size = (Nx, Ny, Nz),
                                 center = (0, 89),
                                 spacing = Δ,
                                 standard_parallels = (80, 85),
                                 central_longitude = 0,
                                 latitude_of_origin = 90,
                                 z = (-H, 0),
                                 halo = (3, 3, 3))

model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                    tracers = ())

η₀ = 0.01meters
Δφ = 1
ηᵢ(λ, φ, z) = η₀ * exp(-((90 - φ) / Δφ)^2)

set!(model, η = ηᵢ)

simulation = Simulation(model; Δt = 20seconds, stop_iteration = 20)

function progress(sim)
    u, v, w = sim.model.velocities
    max_u = maximum(abs, u)
    max_v = maximum(abs, v)

    @info @sprintf("iter %04d, time %s, max|u| %.3e, max|v| %.3e",
                   iteration(sim), prettytime(time(sim)), max_u, max_v)

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(5))

run!(simulation)

η = model.free_surface.displacement
u, v, w = model.velocities

max_η = maximum(abs, interior(η))
max_u = maximum(abs, interior(u))
max_v = maximum(abs, interior(v))

@info "Finished LCC polar hydrostatic splash" max_η max_u max_v
