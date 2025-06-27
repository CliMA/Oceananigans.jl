#=
using Oceananigans
using Oceananigans.Grids: RightConnected
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
using Oceananigans.OrthogonalSphericalShellGrids: rotate_coordinates
using Oceananigans.Units
using OrthogonalSphericalShellGrids: TripolarGrid
using Printf
using GLMakie

Nx, Ny, Nz = 128, 128, 1

grid = RotatedLatitudeLongitudeGrid(size = (Nx, Ny, Nz),
                                    latitude = (-60, 60),
                                    longitude = (-60, 60),
                                    north_pole = (0, 0),
                                    halo = (7, 7, 3),
                                    z = (-1000, 0),
                                    topology = (Bounded, Bounded, Bounded))

model = HydrostaticFreeSurfaceModel(; grid,
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    momentum_advection = WENOVectorInvariant())

U = 1
ϵᵢ(λ, φ, z) = U * (2rand() - 1)
set!(model, u=ϵᵢ, v=ϵᵢ)

Δx = minimum_xspacing(grid)
Δt = min(20minutes, 0.1 * Δx / U)
simulation = Simulation(model; Δt, stop_time=360days)

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))

u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)
s = @at (Center, Center, Center) sqrt(u^2 + v^2)

simulation.output_writers[:jld2] = JLD2Writer(model, (; u, v, ζ, s),
                                              schedule = TimeInterval(1day),
                                              filename = "polar_turbulence",
                                              overwrite_existing = true)

@info "Run simulation..."

run!(simulation)

@info "Load output..."
=#

filepath = simulation.output_writers[:jld2].filepath

ζ_timeseries = FieldTimeSeries(filepath, "ζ")
u_timeseries = FieldTimeSeries(filepath, "u")
v_timeseries = FieldTimeSeries(filepath, "v")
s_timeseries = FieldTimeSeries(filepath, "s")

times = u_timeseries.times

@info "Make a movie of the splash..."

using GLMakie

n = Observable(1)

title = @lift @sprintf("t = %s", prettytime(times[$n]))

ζₙ = @lift interior(ζ_timeseries[$n], :, :, 1)
uₙ = @lift interior(u_timeseries[$n], :, :, 1)
vₙ = @lift interior(v_timeseries[$n], :, :, 1)
sₙ = @lift interior(s_timeseries[$n], :, :, 1)

s_lim = maximum(abs, interior(s_timeseries))
ζ₀ = maximum(abs, ζ_timeseries[end]) / 2

fig = Figure(size=(1200, 800))

ax_u = Axis(fig[1, 1], aspect=1)
ax_v = Axis(fig[1, 3], aspect=1)
ax_ζ = Axis(fig[2, 1], aspect=1)
ax_s = Axis(fig[2, 3], aspect=1)

hm_u = heatmap!(ax_u, uₙ; colormap = :balance, colorrange = (-s_lim, s_lim))
Colorbar(fig[1, 2], hm_u; label = "u (m s⁻¹)")

hm_v = heatmap!(ax_v, vₙ; colormap = :balance, colorrange = (-s_lim, s_lim))
Colorbar(fig[1, 4], hm_v; label = "v (m s⁻¹)")

hm_ζ = heatmap!(ax_ζ, ζₙ; colormap = :balance, colorrange = (-ζ₀, ζ₀))
Colorbar(fig[2, 2], hm_ζ; label = "ζ")

hm_s = heatmap!(ax_s, sₙ; colormap = :speed, colorrange = (0, s_lim))
Colorbar(fig[2, 4], hm_s; label = "√u²+v² (m s⁻¹)")

fig[0, :] = Label(fig, title, tellwidth=false)

fig

frames = 1:length(times)

record(fig, "polar_turbulence.mp4", frames, framerate = 12) do i
    n[] = i
end

