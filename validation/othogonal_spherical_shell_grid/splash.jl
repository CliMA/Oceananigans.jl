using Oceananigans
using Oceananigans.Units
using Rotations

Nx, Ny, Nz = 64, 64, 4

grid = OrthogonalSphericalShellGrid(size = (Nx, Ny, Nz),
                                    z = (-1000, 0),
                                    topology=(Bounded, Bounded, Bounded),
                                    rotation = RotY(π/2))
                   
closure = ScalarDiffusivity(ν=1e-3, κ=1e-3)

model = HydrostaticFreeSurfaceModel(; grid,
                                      momentum_advection = VectorInvariant(),
                                      closure,
                                      buoyancy = nothing,
                                      tracers=())

ηᵢ(λ, φ, z) = 1 * exp(-λ^2/(2*10^2) - φ^2/(2*10^2))
ϵᵢ(λ, φ, z) = 1e-6 * randn()

set!(model, η=ηᵢ, u=ϵᵢ, v=ϵᵢ)

Δt = 3minutes

simulation = Simulation(model, Δt=Δt, stop_time = 4days)

u, v, w = model.velocities
η = model.free_surface.η

save_fields_interval = 5minutes

s = @at (Center, Center, Center) sqrt(u^2 + v^2)

simulation.output_writers[:splash] = JLD2OutputWriter(model, (u=u, v=v, s=s),
                                                     schedule = TimeInterval(save_fields_interval),
                                                     filename = "ossg_splash",
                                                     overwrite_existing = true)

@info "Run simulation..."

run!(simulation)


@info "Load output..."

filepath = simulation.output_writers[:splash].filepath

u_timeseries = FieldTimeSeries(filepath, "u")
v_timeseries = FieldTimeSeries(filepath, "v")
s_timeseries = FieldTimeSeries(filepath, "s")

times = u_timeseries.times

xu, yu, zu = nodes(u_timeseries)
xc, yc, zc = nodes(s_timeseries)

@info "Making a movie of the splash..."

using GLMakie, Printf

n = Observable(1)

title = @lift @sprintf("t = %s", prettytime(times[$n]))

uₙ = @lift interior(u_timeseries[$n], :, :, grid.Nz)
vₙ = @lift interior(v_timeseries[$n], :, :, grid.Nz)
sₙ = @lift interior(s_timeseries[$n], :, :, grid.Nz)

s_lim = maximum(abs, interior(s_timeseries))

fig = Figure(resolution = (1200, 400))

ax_u = Axis(fig[1, 1])
ax_v = Axis(fig[1, 3])
ax_s = Axis(fig[1, 5])

hm_u = heatmap!(ax_u, uₙ; colormap = :balance, colorrange = (-s_lim, s_lim))
Colorbar(fig[1, 2], hm_u; label = "u (m s⁻¹)")

hm_v = heatmap!(ax_v, vₙ; colormap = :balance, colorrange = (-s_lim, s_lim))
Colorbar(fig[1, 4], hm_v; label = "v (m s⁻¹)")

hm_s = heatmap!(ax_s, sₙ; colormap = :speed, colorrange = (0, s_lim))
Colorbar(fig[1, 6], hm_s; label = "√u²+v² (m s⁻¹)")

fig[0, :] = Label(fig, title, tellwidth=false)

fig

frames = 1:length(times)

record(fig, "splash.mp4", frames, framerate=18) do i
    n[] = i
end
