# # Meridional diffusion

using Oceananigans
using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel

using Statistics
using JLD2
using Printf
using GLMakie

include("../solid_body_rotation/hydrostatic_prescribed_velocity_fields.jl")

Ny = 320

# A spherical domain
grid = RegularLatitudeLongitudeGrid(size = (1, Ny, 1),
                                    radius = 1,
                                    latitude = (-80, 80),
                                    longitude = (-180, 180),
                                    z = (-1, 0))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = CPU(),
                                    tracers = :c,
                                    velocities = PrescribedVelocityFields(grid), # quiescent
                                    closure = HorizontallyCurvilinearAnisotropicDiffusivity(κh=1),
                                    buoyancy = nothing)

# Tracer patch for visualization
Gaussian(λ, ϕ, L) = exp(-(λ^2 + ϕ^2) / 2L^2)

# Tracer patch parameters
L = 4 # degree
ϕ₀ = 0 # degrees

cᵢ(λ, ϕ, z) = Gaussian(0, ϕ - ϕ₀, L)

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
                        Δt = 0.01cell_diffusion_time_scale,
                        stop_time = 100cell_diffusion_time_scale,
                        iteration_interval = 100,
                        progress = progress)
                                                         
#output_fields = merge(model.velocities, model.tracers, (η=model.free_surface.η,))
output_fields = model.tracers

output_prefix = "meridional_tracer_diffusion_Ny$(grid.Ny)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(cell_diffusion_time_scale / 10),
                                                      prefix = output_prefix,
                                                      force = true)

run!(simulation)

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

for iter in iterations
    c = file["timeseries/c/$iter"][1, :, 1]
    @show maximum(c)
end

ϕ = ynodes(Center, grid)

iter = Node(0)

plot_title = "hi"

c = @lift file["timeseries/c/" * string($iter)][1, :, 1]


fig = Figure(resolution = (540, 1080))

ax = fig[1, 1] = Axis(fig, xlabel = "U(ϕ)", ylabel = "ϕ")

lines!(ax, c, ϕ, color=:black)

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

record(fig, "meridional_tracer_diffusion_Ny$(grid.Ny).mp4", iterations, framerate=30) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end
