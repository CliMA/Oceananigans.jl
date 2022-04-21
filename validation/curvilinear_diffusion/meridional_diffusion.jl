# # Meridional diffusion

using Oceananigans
lines!(cax, c, ϕ, color = :black)
using Oceananigans.TurbulenceClosures: Horizontal
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, VectorInvariant

using Statistics
using JLD2
using Printf
using GLMakie

include(joinpath(@__DIR__, "..", "solid_body_rotation", "hydrostatic_prescribed_velocity_fields.jl"))

Ny = 320

# A spherical domain
grid = LatitudeLongitudeGrid(size = (1, Ny, 1),
                             radius = 1,
                             latitude = (-80, 80),
                             longitude = (-180, 180),
                             z = (-1, 0))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    momentum_advection = VectorInvariant(),
                                    tracers = :c,
                                    coriolis = nothing,
                                    closure = HorizontalScalarDiffusivity(κ=1, ν=1),
                                    buoyancy = nothing)

# Tracer patch for visualization
Gaussian(λ, ϕ, L) = exp(-(λ^2 + ϕ^2) / 2L^2)

# Tracer patch parameters
L = 4 # degree
ϕ₀ = 0 # degrees

cᵢ(λ, ϕ, z) = Gaussian(0, ϕ - ϕ₀, L)

set!(model, c=cᵢ, u=cᵢ)

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
                        stop_time = 100cell_diffusion_time_scale,
                        iteration_interval = 100,
                        progress = progress)
                                                         
output_fields = merge(model.velocities, model.tracers)

output_prefix = "meridional_diffusion_Ny$(grid.Ny)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(cell_diffusion_time_scale),
                                                      prefix = output_prefix,
                                                      overwrite_existing = true)

run!(simulation)

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

ϕ = ynodes(Center, grid)

iter = Node(0)

plot_title = "hi"

u = @lift file["timeseries/u/" * string($iter)][1, :, 1]
c = @lift file["timeseries/c/" * string($iter)][1, :, 1]

set_theme!(Theme(fontsize = 30))

fig = Figure(resolution = (1920, 1080))

c_title = @lift @sprintf("Tracer diffusion on a meridian, t = %.2e", file["timeseries/t/" * string($iter)])
u_title = @lift @sprintf("Momentum diffusion on a meridian, t = %.2e", file["timeseries/t/" * string($iter)])

cax = fig[1, 1] = Axis(fig,
                       xlabel = "c(ϕ)",
                       ylabel = "ϕ",
                       title = c_title)

uax = fig[1, 2] = Axis(fig,
                       xlabel = "u(ϕ)",
                       ylabel = "ϕ",
                       title = u_title)

lines!(cax, c, ϕ, color = :black)
lines!(uax, u, ϕ, color = :black)

record(fig, "meridional_diffusion_Ny$(grid.Ny).mp4", iterations, framerate = 30) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end
