using Oceananigans
using Test
using Glob

using Oceananigans.Architectures: on_architecture
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: ForwardBackwardScheme
free_surface_timestepper = ForwardBackwardScheme()
timestepper = :SplitRungeKutta3
arch = CPU()

Nx, Ny, Nz = 16, 16, 16
Lx, Ly, Lz = 1000, 1000, 100
Δt = 0.1

bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z - Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))

build_model(substeps) = begin
    # grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Bounded, Bounded, Bounded))
    # grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz), longitude = (0, 360), latitude = (-75, 75), z = (-100, 0))
    H = 3000                   # domain depth [m]

    underlying_grid = TripolarGrid(Oceananigans.CPU(); size=(Nx, Ny, Nz), z=(-H, 0), halo=(5, 5, 4))

    σφ, σλ = 4, 8       # mountain extent in latitude and longitude (degrees)
    λ₀, φ₀ = 70, 55     # first pole location
    h = H + 1000        # mountain height above the bottom (m)

    gaussian(λ, φ) = exp(-((λ - λ₀)^2 / 2σλ^2 + (φ - φ₀)^2 / 2σφ^2))
    gaussian_mountains(λ, φ) = -H + h * (gaussian(λ, φ) + gaussian(λ - 180, φ) + gaussian(λ - 360, φ))

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(gaussian_mountains))
    @info "substeps = $substeps"
    free_surface = SplitExplicitFreeSurface(grid; substeps)
    HydrostaticFreeSurfaceModel(grid; timestepper, free_surface,
                                buoyancy = SeawaterBuoyancy(),
                                tracers = (:T, :S))
end

checkpoint_substeps = 12
restored_substeps = 18
fs_ts_name = nameof(typeof(free_surface_timestepper))
prefix = "SplitExplicit_changed_substeps_$(typeof(arch))_$(timestepper)"

model = build_model(checkpoint_substeps)
set!(model, T=bubble, S=bubble)
simulation = Simulation(model; Δt, stop_iteration=4)
simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = IterationInterval(1),
                                                        prefix = prefix,
                                                        cleanup = false)

run!(simulation)

checkpoint_state = Oceananigans.OutputWriters.load_checkpoint_state("$(prefix)_iteration1.jld2";
                                                                    base_path = "simulation/model")

@test checkpoint_state.free_surface_grid == on_architecture(CPU(), model.free_surface.displacement.grid)

restored_model = build_model(restored_substeps)
restored_simulation = Simulation(restored_model; Δt, stop_iteration=7)
restored_simulation.output_writers[:checkpointer] = Checkpointer(restored_model,
                                                                 schedule = IterationInterval(1),
                                                                 prefix = prefix,
                                                                 cleanup = false)

set!(restored_simulation; checkpoint=:latest)
# @test_logs (:warn, r"different halo size") set!(restored_simulation; checkpoint=:latest)
# @test_nowarn run!(restored_simulation)
run!(restored_simulation)

@test restored_simulation.model.clock.iteration == 7
@test all(isfinite, interior(restored_model.velocities.u))
@test all(isfinite, interior(restored_model.velocities.v))
@test all(isfinite, interior(restored_model.free_surface.displacement))

rm.(glob("$(prefix)_iteration*.jld2"), force=true)