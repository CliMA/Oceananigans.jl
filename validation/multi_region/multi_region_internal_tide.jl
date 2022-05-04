using Printf
using CUDA
using Oceananigans
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.MultiRegion
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using LinearAlgebra
using Adapt

function boundary_clustered(N, L, ini)
    Δz(k)   = k < N / 2 + 1 ? 2 / (N - 1) * (k - 1) + 1 : - 2 / (N - 1) * (k - N) + 1 
    z_faces = zeros(N+1) 
    for k = 2:N+1
        z_faces[k] = z_faces[k-1] + Δz(k-1)
    end
    z_faces = z_faces ./ z_faces[end] .* L .+ ini
    return z_faces
end

function center_clustered(N, L, ini)
    Δz(k)   = k < N / 2 + 1 ? 2 / (N - 1) * (k - 1) + 1 : - 2 / (N - 1) * (k - N) + 1 
    z_faces = zeros(N+1) 
    for k = 2:N+1
        z_faces[k] = z_faces[k-1] + 3 - Δz(k-1)
    end
    z_faces = z_faces ./ z_faces[end] .* L .+ ini
    return z_faces
end

grid = RectilinearGrid(CPU(), size=(512, 1, 256), 
                       x = (-10, 10), 
                       y = (0, 1),
                       z = (0, 5),
                topology = (Periodic, Periodic, Bounded))

# Gaussian bump of width "1"
bump(x, y) = exp(-x^2)

@inline show_name(t) = t() isa ExplicitFreeSurface ? "explicit" : "implicit"

grid_with_bump = ImmersedBoundaryGrid(grid, GridFittedBottom(bump))
mrg_with_bump  = MultiRegionGrid(grid_with_bump, partition=XPartition(2), devices=(0, 1))
# Tidal forcing
tidal_forcing(x, y, z, t) = 1e-4 * cos(t)

    
model = HydrostaticFreeSurfaceModel(grid = mrg_with_bump,
                                    momentum_advection = CenteredSecondOrder(),
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=10),
                                    closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1e-2, κ=1e-2),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=sqrt(0.5)),
                                    forcing = (u = tidal_forcing,))

# Linear stratification
set!(model, b = (x, y, z) -> 4 * z)

progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w))

gravity_wave_speed = sqrt(model.free_surface.gravitational_acceleration * grid.Lz)

Δt = 0.1 * minimum(grid.Δxᶜᵃᵃ) / gravity_wave_speed

simulation = Simulation(model, Δt = Δt, stop_time = 50000Δt)

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

run!(simulation)

@info """
    Simulation complete.
    Output: $(abspath(simulation.output_writers[:fields].filepath))
"""


    
model_ref = HydrostaticFreeSurfaceModel(grid = grid_with_bump,
                                        momentum_advection = CenteredSecondOrder(),
                                        free_surface = ExplicitFreeSurface(gravitational_acceleration=10),
                                        closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1e-2, κ=1e-2),
                                        tracers = :b,
                                        buoyancy = BuoyancyTracer(),
                                        coriolis = FPlane(f=sqrt(0.5)),
                                        forcing = (u = tidal_forcing,))

# Linear stratification
set!(model_ref, b = (x, y, z) -> 4 * z)

progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w))

simulation_ref = Simulation(model_ref, Δt = Δt, stop_time = 50000Δt)


simulation_ref.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

run!(simulation_ref)

@info """
    Simulation complete.
    Output: $(abspath(simulation.output_writers[:fields].filepath))
"""
