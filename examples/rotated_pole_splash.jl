# # Polar splash on a rotated latitude-longitude grid
#
# This example demonstrates a small hydrostatic free-surface simulation near the
# geographic North Pole by rotating the latitude-longitude grid pole away from the
# geographic pole.
#
# The rotated grid avoids the severe meridian convergence that an ordinary
# latitude-longitude grid experiences near the pole.

using Oceananigans
using CairoMakie

CairoMakie.activate!(type = "png")

size = (32, 32, 2)
latitude = (-60, 60)
longitude = (120, 240)
z = (-100, 0)
topology = (Bounded, Bounded, Bounded)

grid = RotatedLatitudeLongitudeGrid(size = size,
                                    latitude = latitude,
                                    longitude = longitude,
                                    north_pole = (0, 0),
                                    z = z,
                                    topology = topology)

model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    free_surface = SplitExplicitFreeSurface(grid; substeps = 5),
                                    tracers = ())

η₀ = 1e-2
Δφ = 8
ηᵢ(λ, φ, z) = η₀ * exp(-((90 - φ) / Δφ)^2)

set!(model, η = ηᵢ)

simulation = Simulation(model; Δt = 10, stop_iteration = 2)
run!(simulation)

η = model.free_surface.displacement

λ = Array(λnodes(grid, Center(), Center()))
φ = Array(φnodes(grid, Center(), Center()))
η_snapshot = Array(interior(η, :, :, 1))

fig = Figure(size = (700, 450))
ax = Axis(fig[1, 1],
          xlabel = "longitude",
          ylabel = "latitude",
          title = "free-surface displacement near the North Pole",
          aspect = DataAspect())

sc = scatter!(ax, vec(λ), vec(φ);
              color = vec(η_snapshot),
              colormap = :thermal,
              markersize = 9)

Colorbar(fig[1, 2], sc; label = "η (m)")

fig
