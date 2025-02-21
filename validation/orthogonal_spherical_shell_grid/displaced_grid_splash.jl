using Oceananigans
using Oceananigans.Grids: RightConnected
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
using Oceananigans.OrthogonalSphericalShellGrids: rotate_coordinates
using Oceananigans.Units
using OrthogonalSphericalShellGrids: TripolarGrid
using Printf
using GLMakie

Nx, Ny, Nz = 64, 64, 2

lat_lon_grid =
    LatitudeLongitudeGrid(size = (Nx, Ny, Nz),
                          latitude = (-60, 60),
                          longitude = (-60, 60),
                          z = (-1000, 0),
                          topology = (Bounded, Bounded, Bounded))

rotated_lat_lon_grid =
    RotatedLatitudeLongitudeGrid(size = (Nx, Ny, Nz),
                                 latitude = (-60, 60),
                                 longitude = (-60, 60),
                                 north_pole = (0, 0),
                                 z = (-1000, 0),
                                 topology = (Bounded, Bounded, Bounded))

λ = λnodes(rotated_lat_lon_grid, Center(), Center(), Center())
φ = φnodes(rotated_lat_lon_grid, Center(), Center(), Center())

@show maximum(φ)
@show minimum(φ)
@show maximum(λ)
@show minimum(λ)

blank_grid = OrthogonalSphericalShellGrid(size = (Nx, Ny, Nz),
                                          z = (-1000, 0),
                                          topology = (Periodic, Bounded, Bounded))

tripolar_grid = TripolarGrid(size=(Nx, Ny, Nz), z=(-1000, 0))

#=
properties = (:λᶜᶜᵃ,
              :λᶠᶜᵃ,
              :λᶜᶠᵃ,
              :λᶠᶠᵃ,
              :φᶜᶜᵃ,
              :φᶠᶜᵃ,
              :φᶜᶠᵃ,
              :φᶠᶠᵃ,
              # :z,
              :Δxᶜᶜᵃ,
              :Δxᶠᶜᵃ,
              :Δxᶜᶠᵃ,
              :Δxᶠᶠᵃ,
              :Δyᶜᶜᵃ,
              :Δyᶜᶠᵃ,
              :Δyᶠᶜᵃ,
              :Δyᶠᶠᵃ,
              :Azᶜᶜᵃ,
              :Azᶠᶜᵃ,
              :Azᶜᶠᵃ,
              :Azᶠᶠᵃ)

for p in properties
    @show p
    bp = getproperty(blank_grid, p)
    tp = getproperty(tripolar_grid, p)
    dp = getproperty(rotated_lat_lon_grid, p)
    @show size(bp) size(dp) size(tp)
    @show size(bp) == size(tp)
    @show size(dp) == size(tp)
end
=#
                   
closure = ScalarDiffusivity(ν=2e-4, κ=2e-4)

grid = rotated_lat_lon_grid
φ₀ = rotated_lat_lon_grid.conformal_mapping.north_pole[2]

# grid = lat_lon_grid
# φ₀ = 0

model = HydrostaticFreeSurfaceModel(grid=lat_lon_grid; closure,
                                    momentum_advection = VectorInvariant())

η₀ = 1
Δ = 10
ηᵢ(λ, φ, z) = η₀ * exp(-(λ^2 + φ^2) / 2Δ^2)
ϵᵢ(λ, φ, z) = 1e-6 * randn()
set!(model, η=ηᵢ, u=ϵᵢ, v=ϵᵢ)

rotated_model = HydrostaticFreeSurfaceModel(grid=rotated_lat_lon_grid; closure,
                                            momentum_advection = VectorInvariant())

set!(rotated_model,
     η = interior(model.free_surface.η),
     u = interior(model.velocities.u),
     v = interior(model.velocities.v))

@show rotated_model.velocities.u
@show rotated_model.velocities.v

fig = Figure(size=(1500, 500))
axλ = Axis(fig[1, 1])
axφ = Axis(fig[1, 2])
axA1 = Axis(fig[2, 1])
axA2 = Axis(fig[2, 2])
axA3 = Axis(fig[2, 3])
axA4 = Axis(fig[2, 4])
axx1 = Axis(fig[3, 1])
axx2 = Axis(fig[3, 2])
axx3 = Axis(fig[3, 3])
axx4 = Axis(fig[3, 4])
axy1 = Axis(fig[4, 1])
axy2 = Axis(fig[4, 2])
axy3 = Axis(fig[4, 3])
axy4 = Axis(fig[4, 4])
axη = Axis(fig[1, 3])

hm = heatmap!(axλ, rotated_lat_lon_grid.λᶜᶜᵃ)
Colorbar(fig[0, 1], hm, vertical=false)

hm = heatmap!(axφ, rotated_lat_lon_grid.φᶜᶜᵃ)
Colorbar(fig[0, 2], hm, vertical=false)

hm = heatmap!(axη, interior(rotated_model.free_surface.η, :, :, 1))
Colorbar(fig[0, 3], hm, vertical=false)

heatmap!(axA1, rotated_lat_lon_grid.Azᶜᶜᵃ)
heatmap!(axA2, rotated_lat_lon_grid.Azᶠᶜᵃ)
heatmap!(axA3, rotated_lat_lon_grid.Azᶜᶠᵃ)
heatmap!(axA4, rotated_lat_lon_grid.Azᶠᶠᵃ)

heatmap!(axx1, rotated_lat_lon_grid.Δxᶜᶜᵃ)
heatmap!(axx2, rotated_lat_lon_grid.Δxᶠᶜᵃ)
heatmap!(axx3, rotated_lat_lon_grid.Δxᶜᶠᵃ)
heatmap!(axx4, rotated_lat_lon_grid.Δxᶠᶠᵃ)

heatmap!(axy1, rotated_lat_lon_grid.Δyᶜᶜᵃ)
heatmap!(axy2, rotated_lat_lon_grid.Δyᶠᶜᵃ)
heatmap!(axy3, rotated_lat_lon_grid.Δyᶜᶠᵃ)
heatmap!(axy4, rotated_lat_lon_grid.Δyᶠᶠᵃ)

display(fig)

model = rotated_model
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
                                                      #schedule = IterationInterval(1),
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

