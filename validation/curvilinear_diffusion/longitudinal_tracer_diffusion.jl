# # Meridional diffusion

using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, PrescribedVelocityFields

using Oceananigans.TurbulenceClosures: Horizontal
using Statistics
using JLD2
using Printf
using GLMakie

Nx = 360

# A spherical domain
grid = LatitudeLongitudeGrid(size = (Nx, 1, 1),
                             radius = 1,
                             latitude = (-60, 60),
                             longitude = (-180, 180),
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

cᵢ(λ, ϕ, z) = Gaussian(λ, 0, L)

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
                        stop_time = 100cell_diffusion_time_scale,
                        iteration_interval = 100,
                        progress = progress)
                                                         
output_fields = model.tracers

output_prefix = "longitudinal_tracer_diffusion_Nx$(grid.Nx)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(cell_diffusion_time_scale),
                                                      prefix = output_prefix,
                                                      force = true)

run!(simulation)

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

for iter in iterations
    c = file["timeseries/c/$iter"][:, 1, 1]
    @show maximum(c)
end

λ = xnodes(Center, grid)

iter = Node(0)

title = @lift "Tracer diffusion on a parallel, t = $(file["timeseries/t/" * string($iter)])"

c = @lift file["timeseries/c/" * string($iter)][:, 1, 1]

fig = Figure(resolution = (1080, 540))

ax = fig[1, 1] = Axis(fig, ylabel = "c(λ)", xlabel = "λ")

lines!(ax, λ, c, color=:black)

record(fig, "longitudinal_tracer_diffusion_Nx$(grid.Nx).mp4", iterations, framerate=30) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end
