using Oceananigans
using.Oceananigans.Grids: conformal_cubed_sphere_panel
using Oceananigans.Units

using Printf, Rotations

Nx, Ny, Nz = 64, 64, 2

grid = conformal_cubed_sphere_panel(size = (Nx, Ny, Nz),
                                    z = (-1000, 0),
                                    topology=(Bounded, Bounded, Bounded),
                                    rotation = RotY(π/2))
                   
closure = ScalarDiffusivity(ν=2e-4, κ=2e-4)

model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = VectorInvariant(),
                                    closure,
                                    buoyancy = nothing,
                                    tracers=())

η₀ = 1
Δ = 10
ηᵢ(λ, φ, z) = η₀ * exp(-(λ^2 + φ^2) / 2Δ^2)
ϵᵢ(λ, φ, z) = 1e-6 * randn()

set!(model, η=ηᵢ, u=ϵᵢ, v=ϵᵢ)

Δt = 3minutes

simulation = Simulation(model, Δt=Δt, stop_time = 2days)

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))


u, v, w = model.velocities
η = model.free_surface.η

save_fields_interval = 18minutes

s = @at (Center, Center, Center) sqrt(u^2 + v^2)

simulation.output_writers[:splash] = JLD2OutputWriter(model, (; u, v, s, η),
                                                      schedule = TimeInterval(save_fields_interval),
                                                      filename = "ossg_splash",
                                                      with_halos = true,
                                                      overwrite_existing = true)

@info "Run simulation..."

run!(simulation)

@info "Load output..."

filepath = simulation.output_writers[:splash].filepath

η_timeseries = FieldTimeSeries(filepath, "η")
u_timeseries = FieldTimeSeries(filepath, "u")
v_timeseries = FieldTimeSeries(filepath, "v")
s_timeseries = FieldTimeSeries(filepath, "s")

times = u_timeseries.times

@info "Make a movie of the splash..."

using GLMakie

n = Observable(1)

title = @lift @sprintf("t = %s", prettytime(times[$n]))

ηₙ = @lift interior(η_timeseries[$n], :, :, 1)
uₙ = @lift interior(u_timeseries[$n], :, :, grid.Nz)
vₙ = @lift interior(v_timeseries[$n], :, :, grid.Nz)
sₙ = @lift interior(s_timeseries[$n], :, :, grid.Nz)

s_lim = maximum(abs, interior(s_timeseries))

fig = Figure(size=(800, 800))

ax_u = Axis(fig[1, 1])
ax_v = Axis(fig[1, 3])
ax_η = Axis(fig[2, 1])
ax_s = Axis(fig[2, 3])

hm_u = heatmap!(ax_u, uₙ; colormap = :balance, colorrange = (-s_lim, s_lim))
Colorbar(fig[1, 2], hm_u; label = "u (m s⁻¹)")

hm_v = heatmap!(ax_v, vₙ; colormap = :balance, colorrange = (-s_lim, s_lim))
Colorbar(fig[1, 4], hm_v; label = "v (m s⁻¹)")

hm_η = heatmap!(ax_η, ηₙ; colormap = :thermal, colorrange = (0, η₀))
Colorbar(fig[2, 2], hm_η; label = "η (m)")

hm_s = heatmap!(ax_s, sₙ; colormap = :speed, colorrange = (0, s_lim))
Colorbar(fig[2, 4], hm_s; label = "√u²+v² (m s⁻¹)")

fig[0, :] = Label(fig, title, tellwidth=false)

fig

frames = 1:length(times)

record(fig, "splash.mp4", frames, framerate = 12) do i
    n[] = i
end
