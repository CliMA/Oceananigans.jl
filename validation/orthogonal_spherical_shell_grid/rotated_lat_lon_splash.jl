using Oceananigans
using Oceananigans.Grids: RightConnected
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
using Oceananigans.OrthogonalSphericalShellGrids: rotate_coordinates
using Oceananigans.Units
using OrthogonalSphericalShellGrids: TripolarGrid
using Printf
using GLMakie
using Random

size = (64, 64, 2)
latitude = (-60, 60)
longitude = (-60, 60)
z = (-1000, 0)
topology = (Bounded, Bounded, Bounded)

η₀ = 1
Δ = 10
ηᵢ(λ, φ, z) = η₀ * exp(-(λ^2 + φ^2) / 2Δ^2)

g1 = LatitudeLongitudeGrid(; size, latitude, longitude, z, topology)
g2 = RotatedLatitudeLongitudeGrid(; size, latitude, longitude, z, topology, north_pole=(0, 0))

momentum_advection = VectorInvariant()
closure = ScalarDiffusivity(ν=2e-4, κ=2e-4)
m1 = HydrostaticFreeSurfaceModel(grid=g1; closure, momentum_advection)
m2 = HydrostaticFreeSurfaceModel(grid=g2; closure, momentum_advection)

Random.seed!(123)
ϵᵢ(λ, φ, z) = 1e-6 * randn()
set!(m1, η=ηᵢ, u=ϵᵢ, v=ϵᵢ)

set!(m2, η = interior(m1.free_surface.η),
         u = interior(m1.velocities.u),
         v = interior(m1.velocities.v))

models = (unrotated=m1, rotated=m2)

for name in keys(models)
    model = models[name]
    Δt = 3minutes
    simulation = Simulation(model, Δt=Δt, stop_time=2days)

    progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, wall time: %s\n",
                                    iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                    prettytime(sim.run_wall_time))

    simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))

    u, v, w = model.velocities
    η = model.free_surface.η
    s = @at (Center, Center, Center) sqrt(u^2 + v^2)

    simulation.output_writers[:splash] = JLD2Writer(model, (; u, v, s, η),
                                                    schedule = IterationInterval(6),
                                                    filename = "splash_$name",
                                                    overwrite_existing = true)

    @info "Run simulation..."
    run!(simulation)
end

@show interior(m1.free_surface.η) == interior(m2.free_surface.η)
@show interior(m1.velocities.u)   == interior(m2.velocities.u)
@show interior(m1.velocities.v)   == interior(m2.velocities.v)

filepath1 = "splash_unrotated"
filepath2 = "splash_rotated"

η1 = FieldTimeSeries(filepath1, "η")
u1 = FieldTimeSeries(filepath1, "u")
v1 = FieldTimeSeries(filepath1, "v")
s1 = FieldTimeSeries(filepath1, "s")

η2 = FieldTimeSeries(filepath2, "η")
u2 = FieldTimeSeries(filepath2, "u")
v2 = FieldTimeSeries(filepath2, "v")
s2 = FieldTimeSeries(filepath2, "s")

n = Observable(1)

η1n = @lift interior(η1[$n], :, :, 1)
u1n = @lift interior(u1[$n], :, :, 1)
v1n = @lift interior(v1[$n], :, :, 1)
s1n = @lift interior(s1[$n], :, :, 1)

η2n = @lift interior(η2[$n], :, :, 1)
u2n = @lift interior(u2[$n], :, :, 1)
v2n = @lift interior(v2[$n], :, :, 1)
s2n = @lift interior(s2[$n], :, :, 1)

fig = Figure(size=(1400, 800))

ax1 = Axis(fig[2, 1])
ax2 = Axis(fig[2, 2])
ax3 = Axis(fig[2, 3])
ax4 = Axis(fig[2, 4])

ax5 = Axis(fig[3, 1])
ax6 = Axis(fig[3, 2])
ax7 = Axis(fig[3, 3])
ax8 = Axis(fig[3, 4])

hm = heatmap!(ax1, η1n)
Colorbar(fig[1, 1], hm; label = "η (m)", vertical=false)

hm = heatmap!(ax2, u1n)
Colorbar(fig[1, 2], hm; label = "u (m s⁻¹)", vertical=false)

hm = heatmap!(ax3, v1n)
Colorbar(fig[1, 3], hm; label = "v (m s⁻¹)", vertical=false)

hm = heatmap!(ax4, s1n)
Colorbar(fig[1, 4], hm; label = "√(u² + v²) (m s⁻¹)", vertical=false)

heatmap!(ax5, η2n)
heatmap!(ax6, u2n)
heatmap!(ax7, v2n)
heatmap!(ax8, s2n)

times = u1.times
title = @lift @sprintf("t = %s", prettytime(times[$n]))
Label(fig[0, 1:4], title, tellwidth=false)

display(fig)

frames = 1:length(times)

record(fig, "splash.mp4", frames, framerate = 12) do nn
    n[] = nn
end

