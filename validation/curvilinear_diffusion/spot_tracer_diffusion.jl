using Oceananigans
using Oceananigans.TurbulenceClosures: Horizontal
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, PrescribedVelocityFields

using Statistics
using JLD2
using Printf
using GLMakie

Nx = 360
Ny = 120
latitude = (-60, 60)
longitude = (-180, 180)

# A spherical domain
grid = LatitudeLongitudeGrid(
    size = (Nx, Ny, 1),
    radius = 1,
    latitude = latitude,
    longitude = longitude,
    z = (-1, 0)
)

model = HydrostaticFreeSurfaceModel(
    grid = grid,
    tracers = :c,
    velocities = PrescribedVelocityFields(), # quiescent
    closure = HorizontalScalarDiffusivity(κ = 1),
    buoyancy = nothing
)

# Tracer patch for visualization
Gaussian(λ, φ, L) = exp(-(λ^2 + φ^2) / 2L^2)

# Tracer patch parameters
L = 12 # degree
φ₀ = 0 # degrees

cᵢ(λ, φ, z) = Gaussian(λ, φ - φ₀, L)

set!(model, c = cᵢ)

c = model.tracers.c

φᵃᶜᵃ_max = maximum(abs, φnodes(grid, Center(), Center(), Center()))
Δx_min = grid.radius * cosd(φᵃᶜᵃ_max) * deg2rad(grid.Δλᶜᵃᵃ)
Δy_min = grid.radius * deg2rad(grid.Δφᵃᶜᵃ)
Δ_min = min(Δx_min, Δy_min)

# Time-scale for gravity wave propagation across the smallest grid cell
cell_diffusion_time_scale = Δ_min^2

simulation = Simulation(
    model;
    Δt = 0.1cell_diffusion_time_scale,
    stop_time = 1000cell_diffusion_time_scale,
)

function progress_message(sim)
    max_c = maximum(sim.model.tracers.c)
    walltime = prettytime(sim.run_wall_time)

    return @info @sprintf("Iteration: %04d, time: %1.3f, Δt: %.2e, max(c) = %.1e, wall time: %s\n",
        iteration(sim), time(sim), sim.Δt, max_c, walltime)
end

add_callback!(simulation, progress_message, IterationInterval(100))

output_fields = model.tracers

output_prefix = "spot_tracer_diffusion_Nx$(grid.Nx)_Ny$(grid.Ny)"

simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields,
    schedule = TimeInterval(10cell_diffusion_time_scale),
    filename = output_prefix,
    overwrite_existing = true
)

run!(simulation)

λ = λnodes(grid, Center(), Center(), Center())
φ = φnodes(grid, Center(), Center(), Center())

λ = repeat(reshape(λ, Nx, 1), 1, Ny)
φ = repeat(reshape(φ, 1, Ny), Nx, 1)

c_timeseries = FieldTimeSeries(simulation.output_writers[:fields].filepath, "c")
times = c_timeseries.times

set_theme!(Theme(fontsize = 30))

fig = Figure(size = (1920, 1080))

n = Observable(1)
title = @lift "Tracer spot on a sphere, t = " * prettytime(times[$n])

c = @lift c_timeseries[$n]

fig = Figure(size = (1920, 1080))

ax = fig[1, 1] = Axis(
    fig,
    xlabel = "λ",
    ylabel = "φ",
    title = title,
)

heatmap!(ax, c)

frames = 1:length(times)

@info "Making an animation..."

Makie.record(fig, "spot_tracer_diffusion_Nx$(grid.Nx)_Ny$(grid.Ny).mp4", frames, framerate = 24) do i
    n[] = i
end
