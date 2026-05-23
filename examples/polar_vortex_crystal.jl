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
# with a circular wall (an [`ImmersedBoundaryGrid`](@ref) with `GridFittedBoundary`)
# at 1500 km from the pole forming the polar disk.

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
                                 halo = (5, 5, 5))

R_earth = grid.radius
R_bowl  = 1500kilometers
bowl_mask(λ, φ, z) = R_earth * (π/2 - deg2rad(φ)) > R_bowl
ibg = ImmersedBoundaryGrid(grid, GridFittedBoundary(bowl_mask))

# ## Model
#
# Single-layer barotropic shallow-water-style dynamics via a
# `HydrostaticFreeSurfaceModel` with one vertical level, the spherical
# Coriolis term (so f ≈ 2Ω near the pole), and an `ImplicitFreeSurface`.

model = HydrostaticFreeSurfaceModel(ibg;
                                    coriolis     = HydrostaticSphericalCoriolis(),
                                    free_surface = ImplicitFreeSurface(),
                                    tracers      = ())

# ## Initial condition
#
# Six Gaussian cyclones at radius `r_ring = 900 km` from the pole, evenly
# spaced in longitude, plus one central cyclone at the pole. The depression
# amplitude `η₀ = -13 m` and width `σ = 200 km` give an initial Rossby number
# `Ro ≈ -2gη₀/(f²σ²) ≈ 0.3` at each vortex centre.
#
# The free-surface bumps and the corresponding geostrophic velocities are
# both set so the flow starts in approximate balance and does not radiate the
# vortex energy away as gravity waves during initial adjustment.
#
# Distances and bearings are computed with great-circle formulas in
# geographic coordinates — no projection composition, no seam.

N_ring   = 6
r_ring   = 900kilometers
σ_vortex = 200kilometers
η₀       = -13.0

ring_λ   = [360k/N_ring for k in 0:N_ring-1]
ring_φ   = fill(90 - rad2deg(r_ring/R_earth), N_ring)
vortex_λ = vcat(ring_λ, [0.0])
vortex_φ = vcat(ring_φ, [90.0])

g_const = Oceananigans.defaults.gravitational_acceleration
f_pole  = 2 * Oceananigans.defaults.planet_rotation_rate

function great_circle_distance_and_bearing(λ, φ, λv, φv)
    λr,  φr  = deg2rad(λ),  deg2rad(φ)
    λvr, φvr = deg2rad(λv), deg2rad(φv)
    Δλ = λvr - λr
    a  = sin((φvr - φr)/2)^2 + cos(φr) * cos(φvr) * sin(Δλ/2)^2
    d  = R_earth * 2 * asin(sqrt(clamp(a, 0.0, 1.0)))
    yb = sin(Δλ) * cos(φvr)
    xb = cos(φr) * sin(φvr) - sin(φr) * cos(φvr) * cos(Δλ)
    return d, atan(yb, xb)
end

function η_init(λ, φ, z)
    η = 0.0
    for k in 1:length(vortex_λ)
        d, _ = great_circle_distance_and_bearing(λ, φ, vortex_λ[k], vortex_φ[k])
        η += η₀ * exp(-(d/σ_vortex)^2 / 2)
    end
    return η
end

function geostrophic_uv(λ, φ)
    u_e, v_n = 0.0, 0.0
    for k in 1:length(vortex_λ)
        d, θ  = great_circle_distance_and_bearing(λ, φ, vortex_λ[k], vortex_φ[k])
        v_tan = -(g_const * η₀ * d) / (f_pole * σ_vortex^2) * exp(-(d/σ_vortex)^2 / 2)
        u_e  +=  v_tan * cos(θ)
        v_n  += -v_tan * sin(θ)
    end
    return u_e, v_n
end

u_init(λ, φ, z) = geostrophic_uv(λ, φ)[1]
v_init(λ, φ, z) = geostrophic_uv(λ, φ)[2]

set!(model, η = η_init, u = u_init, v = v_init)

# ## Simulation
#
# Five-day run at Δt = 1 minute, with hourly snapshots of free-surface
# displacement, speed, and relative vorticity.

simulation = Simulation(model; Δt = 60, stop_time = 5days)

progress(sim) = @printf("iter %4d, t = %s, max|u| = %.3f m/s, max|η| = %.2f m\n",
                        iteration(sim), prettytime(time(sim)),
                        maximum(abs, sim.model.velocities.u),
                        maximum(abs, sim.model.free_surface.displacement))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(200))

u, v, w = model.velocities
η = model.free_surface.displacement
ζ = ∂x(v) - ∂y(u)
s = sqrt(u^2 + v^2)

filename = "polar_vortex_crystal.jld2"
simulation.output_writers[:fields] = JLD2Writer(model, (; η, ζ, s);
                                                filename,
                                                schedule = TimeInterval(1hour),
                                                overwrite_existing = true)

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

η_lim = maximum(maximum(abs, interior(ηts[n], :, :, 1)) for n in 1:Nt)
ζ_lim = maximum(maximum(abs, interior(ζts[n], :, :, 1)) for n in 1:Nt) * 0.5
s_lim = maximum(maximum(abs, interior(sts[n], :, :, 1)) for n in 1:Nt)

n = Observable(1)
title = @lift @sprintf("Polar vortex crystal — t = %.2f d", times[$n] / 86400)
η_n = @lift Array(interior(ηts[$n], :, :, 1))
ζ_n = @lift Array(interior(ζts[$n], :, :, 1))
s_n = @lift Array(interior(sts[$n], :, :, 1))

fig = Figure(size = (1500, 540))
Label(fig[0, 1:6], title, fontsize = 18)

ax_η = Axis(fig[1, 1], aspect = 1, title = "η (m)")
hm_η = heatmap!(ax_η, η_n; colormap = :balance, colorrange = (-η_lim, η_lim))
Colorbar(fig[1, 2], hm_η)

ax_ζ = Axis(fig[1, 3], aspect = 1, title = "ζ (1/s)")
hm_ζ = heatmap!(ax_ζ, ζ_n; colormap = :balance, colorrange = (-ζ_lim, ζ_lim))
Colorbar(fig[1, 4], hm_ζ)

ax_s = Axis(fig[1, 5], aspect = 1, title = "|u| (m/s)")
hm_s = heatmap!(ax_s, s_n; colormap = :speed, colorrange = (0, s_lim))
Colorbar(fig[1, 6], hm_s)

for ax in (ax_η, ax_ζ, ax_s)
    hidedecorations!(ax)
end

record(fig, "polar_vortex_crystal.mp4", 1:Nt; framerate = 12) do i
    n[] = i
end
nothing #hide

# ![](polar_vortex_crystal.mp4)
