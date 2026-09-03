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

grid = LambertConformalConicGrid(Float64;
                                 size = (Nx, Ny, 1),
                                 center = (0, 90),
                                 spacing = Δ,
                                 standard_parallel = 90,
                                 latitude_of_origin = 90,
                                 central_longitude = 0,
                                 z = (-H, 0),
                                 halo = (7, 7, 7))

R_earth = grid.radius
R_bowl  = 1500kilometers
bowl_bottom(λ, φ) = ifelse(R_earth * (π/2 - deg2rad(φ)) > R_bowl, zero(H), -H)
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
# Six Gaussian cyclones at radius `r_ring = 900 km` from the pole, evenly
# spaced in longitude, plus one central cyclone at the pole. The depression
# amplitude `η₀ = -13 m` and width `σ = 200 km` give an initial Rossby number
# `Ro ≈ -2gη₀/(f²σ²) ≈ 0.3` at each vortex centre.
#
# Working in projected (`xp`, `yp`) coordinates of the polar stereographic
# limit lets us write `η` and the geostrophic velocities `(u, v)` as plain
# Gaussians of distance from each vortex centre. `set!` is told these are in
# the grid's intrinsic frame.

N_ring   = 6
r_ring   = 900kilometers
σ_vortex = 200kilometers
η₀       = -13

ring_λ      = [360k/N_ring for k in 0:N_ring-1]
ring_φ      = fill(90 - rad2deg(r_ring/R_earth), N_ring)
vortex_λ    = vcat(ring_λ, [0])
vortex_φ    = vcat(ring_φ, [90])
vortex_xpyp = [lcc_forward(grid.conformal_mapping, λv, φv)
               for (λv, φv) in zip(vortex_λ, vortex_φ)]

g_const  = Oceananigans.defaults.gravitational_acceleration
f_pole   = 2 * Oceananigans.defaults.planet_rotation_rate
geo_coef = g_const * η₀ / (f_pole * σ_vortex^2)

gaussian(xp, yp, xv, yv) = exp(-((xp - xv)^2 + (yp - yv)^2) / (2σ_vortex^2))

function η_init(λ, φ, z)
    xp, yp = lcc_forward(grid.conformal_mapping, λ, φ)
    return sum(η₀ * gaussian(xp, yp, xv, yv) for (xv, yv) in vortex_xpyp)
end

function u_init(λ, φ, z)
    xp, yp = lcc_forward(grid.conformal_mapping, λ, φ)
    return sum( geo_coef * (yp - yv) * gaussian(xp, yp, xv, yv) for (xv, yv) in vortex_xpyp)
end

function v_init(λ, φ, z)
    xp, yp = lcc_forward(grid.conformal_mapping, λ, φ)
    return sum(-geo_coef * (xp - xv) * gaussian(xp, yp, xv, yv) for (xv, yv) in vortex_xpyp)
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
                        iteration(sim), prettytime(time(sim)),
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
                                                overwrite_existing = true)

## Fail the docs build if this simulation produces NaNs #hide
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
run!(simulation)

# ## Visualization
#
# Animate η, |u|, and ζ over the 5-day evolution.

using CairoMakie

CairoMakie.activate!(type = "png")

ηts = FieldTimeSeries(filename, "η")
ζts = FieldTimeSeries(filename, "ζ")
sts = FieldTimeSeries(filename, "s")
times = ηts.times
Nt = length(times)

η_lim = maximum(maximum(abs, interior(ηts[n])) for n in 1:Nt)
ζ_lim = maximum(maximum(abs, interior(ζts[n])) for n in 1:Nt) * 0.5
s_lim = maximum(maximum(abs, interior(sts[n])) for n in 1:Nt)

n = Observable(1)
title = @lift @sprintf("Polar vortex crystal — t = %.2f d", times[$n] / 86400)
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
