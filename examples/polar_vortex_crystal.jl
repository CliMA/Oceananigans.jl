# # Polar vortex crystal
#
# Self-organisation of cyclones into a stable ring around a central cyclone in
# rotating shallow water on a polar disk — a regime studied by
# [Siegelman, Young & Ingersoll (2022)](@cite SiegelmanYoungIngersoll2022)
# as a model for Jupiter's polar vortex clusters observed by Juno's JIRAM
# instrument.
#
# The grid is a [`LambertConformalConicGrid`](@ref Oceananigans.OrthogonalSphericalShellGrids.LambertConformalConicGrid)
# centred exactly on the North Pole with `standard_parallel = 90` — the
# polar stereographic limit, where the cone is tangent to the sphere at the
# pole and the projection has no antemeridian wedge.

using Oceananigans
using Oceananigans.OrthogonalSphericalShellGrids
using Oceananigans.Units
using Printf

# ## Grid
#
# A 128² horizontal × 1 vertical-cell grid covering a 3200 km box on the pole,
# with a circular wall (an [`ImmersedBoundaryGrid`](@ref) with `GridFittedBottom`
# whose bottom height reaches the top of the domain) at 1500 km from the pole
# forming the polar disk.

Nx = Ny = 128
Δ  = 25kilometers
H  = 1000meters

grid = LambertConformalConicGrid(; size = (Nx, Ny, 1),
                                   center = (0, 90),
                                   spacing = Δ,
                                   standard_parallel = 90,
                                   latitude_of_origin = 90,
                                   central_longitude = 0,
                                   z = (-H, 0),
                                   halo = (7, 7, 7))

R = grid.radius
disk_radius = 1500kilometers
bowl_bottom(λ, φ) = R * deg2rad(90 - φ) > disk_radius ? 0 : -H
ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(bowl_bottom))

# ## Model
#
# Single-layer barotropic shallow-water-style dynamics via a
# `HydrostaticFreeSurfaceModel` with one vertical level. We use
# `HydrostaticSphericalCoriolis` (so f ≈ 2Ω near the pole), a
# `SplitExplicitFreeSurface` whose number of substeps is set from the
# barotropic-wave CFL, and `WENOVectorInvariant` momentum advection.

Δt = 20minutes

model = HydrostaticFreeSurfaceModel(ibg;
                                    coriolis           = HydrostaticSphericalCoriolis(),
                                    free_surface       = SplitExplicitFreeSurface(ibg; cfl = 0.7, fixed_Δt = Δt),
                                    momentum_advection = WENOVectorInvariant())

# ## Initial condition
#
# Six Gaussian cyclones at a distance `ring_radius = 900 km` from the pole, evenly
# spaced in longitude, plus one central cyclone at the pole. The depression
# amplitude `η₀ = -13 m` and width `σ = 200 km` give an initial Rossby number
# `Ro ≈ -2gη₀/(f²σ²) ≈ 0.3` at each vortex centre.
#
# Working in the projected coordinates `(x, y)` of the polar stereographic
# limit lets us write `η` as a sum of Gaussians centred on `(xᵥ, yᵥ)`, and the
# geostrophic velocities `u = -(g/f) ∂η/∂y` and `v = (g/f) ∂η/∂x` in closed form.
# `set!` is told these velocities are in the grid's intrinsic frame.

ring_size   = 6
ring_radius = 900kilometers
σ  = 200kilometers
η₀ = -13

ring_latitude = 90 - rad2deg(ring_radius / R)
ring_longitudes = [360k / ring_size for k in 0:ring_size-1]
vortex_coordinates = [[(λ, ring_latitude) for λ in ring_longitudes]; (0, 90)]
vortex_centres = [lcc_forward(grid.conformal_mapping, λ, φ) for (λ, φ) in vortex_coordinates]

g = Oceananigans.defaults.gravitational_acceleration
f = 2 * Oceananigans.defaults.planet_rotation_rate

ηᵥ(x, y, xᵥ, yᵥ) = η₀ * exp(-((x - xᵥ)^2 + (y - yᵥ)^2) / 2σ^2)

function η_init(λ, φ, z)
    x, y = lcc_forward(grid.conformal_mapping, λ, φ)
    return sum(ηᵥ(x, y, xᵥ, yᵥ) for (xᵥ, yᵥ) in vortex_centres)
end

function u_init(λ, φ, z)
    x, y = lcc_forward(grid.conformal_mapping, λ, φ)
    return sum(g / f * (y - yᵥ) / σ^2 * ηᵥ(x, y, xᵥ, yᵥ) for (xᵥ, yᵥ) in vortex_centres)
end

function v_init(λ, φ, z)
    x, y = lcc_forward(grid.conformal_mapping, λ, φ)
    return sum(- g / f * (x - xᵥ) / σ^2 * ηᵥ(x, y, xᵥ, yᵥ) for (xᵥ, yᵥ) in vortex_centres)
end

set!(model, η = η_init, u = u_init, v = v_init; intrinsic_velocities = true)

# ## Simulation
#
# 120-day run at Δt = 20 minutes (outer timestep limited by the advective
# CFL; the split-explicit substeps handle the much faster barotropic
# gravity waves internally). The progress callback reports the advective
# CFL alongside the maximum velocity and free-surface displacement.
# Snapshots are saved every 12 hours.

simulation = Simulation(model; Δt, stop_time = 120days)

advective_cfl = AdvectiveCFL(simulation.Δt)

progress(sim) = @printf("iter %5d, t = %s, max|u| = %.3f m/s, max|η| = %.2f m, CFL = %.3f\n",
                        iteration(sim), prettytime(sim),
                        maximum(abs, sim.model.velocities.u),
                        maximum(abs, sim.model.free_surface.displacement),
                        advective_cfl(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

u, v, w = model.velocities
η = model.free_surface.displacement
ζ = ∂x(v) - ∂y(u)
s = sqrt(u^2 + v^2)

filename = "polar_vortex_crystal.jld2"
simulation.output_writers[:fields] = JLD2Writer(model, (; η, ζ, s);
                                                filename,
                                                schedule = TimeInterval(12hours),
                                                overwrite_files = true)

## Fail the docs build if this simulation produces NaNs #hide
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
run!(simulation)

# ## Visualization
#
# Animate η, ζ, and |u| over the 120-day evolution.

using CairoMakie

CairoMakie.activate!(type = "png")

ηts = FieldTimeSeries(filename, "η")
ζts = FieldTimeSeries(filename, "ζ")
sts = FieldTimeSeries(filename, "s")
times = ηts.times
Nt = length(times)

η_lim = maximum(abs, ηts)
ζ_lim = maximum(abs, ζts) / 2
s_lim = maximum(sts)

n = Observable(1)
title = @lift "Polar vortex crystal — t = " * prettytime(times[$n])
ηₙ = @lift ηts[$n]
ζₙ = @lift ζts[$n]
sₙ = @lift sts[$n]

fig = Figure(size = (1500, 540))
Label(fig[0, 1:6], title; fontsize = 18, tellwidth = false)

ax_η = Axis(fig[1, 1], aspect = 1, title = "η (m)")
hm_η = heatmap!(ax_η, ηₙ; colormap = :balance, colorrange = (-η_lim, η_lim))
Colorbar(fig[1, 2], hm_η)

ax_ζ = Axis(fig[1, 3], aspect = 1, title = "ζ (1/s)")
hm_ζ = heatmap!(ax_ζ, ζₙ; colormap = :balance, colorrange = (-ζ_lim, ζ_lim))
Colorbar(fig[1, 4], hm_ζ)

ax_s = Axis(fig[1, 5], aspect = 1, title = "|u| (m/s)")
hm_s = heatmap!(ax_s, sₙ; colormap = :speed, colorrange = (0, s_lim))
Colorbar(fig[1, 6], hm_s)

for ax in (ax_η, ax_ζ, ax_s)
    hidedecorations!(ax)
end

record(fig, "polar_vortex_crystal.mp4", 1:Nt; framerate = 12) do i
    n[] = i
end
nothing #hide

# ![](polar_vortex_crystal.mp4)
