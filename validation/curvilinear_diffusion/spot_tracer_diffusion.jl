# # Meridional diffusion

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
grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                             radius = 1,
                             latitude = latitude,
                             longitude = longitude,
                             z = (-1, 0))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = :c,
                                    velocities = PrescribedVelocityFields(), # quiescent
                                    closure = ScalarDiffusivity(κ=1, isotropy=Horizontal()),
                                    buoyancy = nothing)

# Tracer patch for visualization
Gaussian(λ, ϕ, L) = exp(-(λ^2 + ϕ^2) / 2L^2)

# Tracer patch parameters
L = 12 # degree
ϕ₀ = 0 # degrees

cᵢ(λ, ϕ, z) = Gaussian(λ, ϕ - ϕ₀, L)

set!(model, c=cᵢ)

c = model.tracers.c

function progress(s)
    c = s.model.tracers.c
    @info "Maximum(c) = $(maximum(c)), time = $(s.model.clock.time) / $(s.stop_time)"
    return nothing
end

ϕᵃᶜᵃ_max = maximum(abs, ynodes(Center, grid))
Δx_min = grid.radius * cosd(ϕᵃᶜᵃ_max) * deg2rad(grid.Δλ)
Δy_min = grid.radius * deg2rad(grid.Δϕ)
Δ_min = min(Δx_min, Δy_min)

# Time-scale for gravity wave propagation across the smallest grid cell
cell_diffusion_time_scale = Δ_min^2

simulation = Simulation(model,
                        Δt = 0.1cell_diffusion_time_scale,
                        stop_time = 1000cell_diffusion_time_scale,
                        iteration_interval = 100,
                        progress = progress)
                                                         
output_fields = model.tracers

output_prefix = "spot_tracer_diffusion_Nx$(grid.Nx)_Ny$(grid.Ny)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(10cell_diffusion_time_scale),
                                                      prefix = output_prefix,
                                                      force = true)

run!(simulation)

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

λ = xnodes(Center, grid)
ϕ = ynodes(Center, grid)

λ = repeat(reshape(λ, Nx, 1), 1, Ny)
ϕ = repeat(reshape(ϕ, 1, Ny), Nx, 1)

iter = Node(0)

plot_title = "hi"

c = @lift file["timeseries/c/" * string($iter)][:, :, 1]

set_theme!(Theme(fontsize = 30))

fig = Figure(resolution = (1920, 1080))

title = @lift "Tracer spot on a sphere, t = $(file["timeseries/t/" * string($iter)])"

ax = fig[1, 1] = Axis(fig,
                      xlabel = "λ",
                      ylabel = "ϕ",
                      title = title)

heatmap!(ax, c)

record(fig, "spot_tracer_diffusion_Nx$(grid.Nx)_Ny$(grid.Ny).mp4", iterations, framerate=30) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end
