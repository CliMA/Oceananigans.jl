# # Spurious diapycnal mixing of a mixed-layer front by the triad isopycnal closure
#
# `triad_Sx` and `triad_Sy` clip the slope, not the triad weight, when the column is not stably stratified:
#
#     bz = max(bz, zero(grid))
#     return ifelse(bz == 0, zero(grid), - bx / bz)
#
# A triad with `bz <= 0` therefore contributes `ϵκ (∂ₓc + 0 ⋅ ∂zc) ⋅ (1, 0)`, a purely *horizontal* flux carrying
# the full symmetric diffusivity. Everywhere else each triad contributes `ϵκ (∂ₓc + S ∂zc) ⋅ (1, S)`, which is
# exactly orthogonal to the triad's own buoyancy gradient and so moves no buoyancy at all.
#
# This validation isolates that branch with a neutrally stratified mixed layer sitting on a stratified interior.
# Inside the mixed layer `b` is independent of `z`, so `∂z b == 0` exactly and the mixed-layer isopycnals are
# *vertical*: an isoneutral closure should diffuse purely vertically there and leave the front untouched. Both
# closures are exactly degenerate on `b` in the stratified interior, so the reference solution is trivial — the
# buoyancy field must not change at all, and every change seen below is spurious diapycnal mixing.
#
# The model is run without dynamics (`PrescribedVelocityFields`, no tracer advection), so the closure is the only
# process acting. `TriadIsopycnalSkewSymmetricDiffusivity` erases roughly a third of the front in twenty days;
# `IsopycnalSkewSymmetricDiffusivity`, whose `calc_tapering` guards on `bz <= 0` and switches the whole tensor
# off, leaves the buoyancy field bitwise unchanged.
#
# The slope limiter is not involved: the interior isopycnal slope peaks at 5e-3, half the `FluxTapering(1e-2)`
# threshold, and the run prints the spurious tendency computed with the tapering removed entirely to confirm that
# it is unchanged.

using Printf
using Oceananigans
using Oceananigans.Units
using CairoMakie

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedVelocityFields
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, TriadIsopycnalSkewSymmetricDiffusivity
using Oceananigans.TurbulenceClosures: FluxTapering, ExplicitTimeDiscretization, ∇_dot_qᶜ

#####
##### Setup
#####

horizontal_extent   = 200kilometers
depth               = 1000
mixed_layer_depth   = 250
front_width         = 20kilometers

Nx = 128
Nz = 64

interior_stratification = 1e-5       # N² below the mixed layer [s⁻²]
front_buoyancy_jump     = 1e-3       # Δb across the front [m s⁻²]
symmetric_diffusivity   = 1e3        # κ_symmetric [m² s⁻¹]

stop_time = 20days
Δt        = 5minutes                 # below the explicit isoneutral limit set by κ (Δx⁻² + S² Δz⁻² + 2 S Δx⁻¹Δz⁻¹)

grid = RectilinearGrid(CPU();
                       topology = (Bounded, Flat, Bounded),
                       size = (Nx, Nz),
                       x = (0, horizontal_extent),
                       z = (-depth, 0),
                       halo = (3, 3))

ramp(x) = (1 + tanh((x - horizontal_extent / 2) / front_width)) / 2

# `min(z, -mixed_layer_depth)` freezes the vertical structure above the mixed layer base, so ∂z b is exactly zero
# there while the horizontal front is carried unchanged through the whole column.
initial_buoyancy(x, z) = interior_stratification * min(z, -mixed_layer_depth) + front_buoyancy_jump * ramp(x)

gerdes_koberle_willebrand_tapering = FluxTapering(1e-2)

triad_closure = TriadIsopycnalSkewSymmetricDiffusivity(ExplicitTimeDiscretization();
                                                       κ_skew = 0,
                                                       κ_symmetric = symmetric_diffusivity,
                                                       slope_limiter = gerdes_koberle_willebrand_tapering)

cox_closure = IsopycnalSkewSymmetricDiffusivity(ExplicitTimeDiscretization();
                                                κ_skew = 0,
                                                κ_symmetric = symmetric_diffusivity,
                                                slope_limiter = gerdes_koberle_willebrand_tapering)

#####
##### The slope limiter is not responsible: the spurious tendency is identical with the tapering removed
#####

function buoyancy_tendency(closure, buoyancy_field, grid)
    tendency = CenterField(grid)
    fields = (; b = buoyancy_field)
    Nx, Ny, Nz = size(grid)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        tendency[i, j, k] = - ∇_dot_qᶜ(i, j, k, grid, closure, nothing, Val(1), buoyancy_field,
                                       Clock(time=0.0), fields, BuoyancyTracer())
    end
    return tendency
end

untapered_triad_closure = TriadIsopycnalSkewSymmetricDiffusivity(ExplicitTimeDiscretization();
                                                                 κ_skew = 0,
                                                                 κ_symmetric = symmetric_diffusivity,
                                                                 slope_limiter = FluxTapering(1e6))

initial_buoyancy_field = CenterField(grid)
set!(initial_buoyancy_field, initial_buoyancy)
fill_halo_regions!(initial_buoyancy_field)

isopycnal_slope = Field(@at((Face, Center, Center), - ∂x(initial_buoyancy_field) / ∂z(initial_buoyancy_field)))
compute!(isopycnal_slope)
interior_slope = maximum(abs, interior(isopycnal_slope)[:, 1, 1:Nz - mixed_layer_depth ÷ (depth ÷ Nz) - 4])

tapered_tendency   = buoyancy_tendency(triad_closure, initial_buoyancy_field, grid)
untapered_tendency = buoyancy_tendency(untapered_triad_closure, initial_buoyancy_field, grid)
cox_tendency       = buoyancy_tendency(cox_closure, initial_buoyancy_field, grid)

@info @sprintf("interior isopycnal slope     : %.1e  (tapering threshold %.1e, so the limiter is inactive)",
               interior_slope, gerdes_koberle_willebrand_tapering.max_slope)
@info @sprintf("triad     max |∂b/∂t|        : %.3e s⁻¹ m s⁻²", maximum(abs, interior(tapered_tendency)))
@info @sprintf("triad     max |∂b/∂t| no taper: %.3e s⁻¹ m s⁻²  (identical: %s)",
               maximum(abs, interior(untapered_tendency)),
               interior(tapered_tendency) ≈ interior(untapered_tendency))
@info @sprintf("non-triad max |∂b/∂t|        : %.3e s⁻¹ m s⁻²", maximum(abs, interior(cox_tendency)))

#####
##### Runs
#####

function run_frontal_relaxation(closure, name)
    model = HydrostaticFreeSurfaceModel(grid;
                                        closure,
                                        velocities = PrescribedVelocityFields(),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        tracer_advection = nothing)

    set!(model, b = initial_buoyancy)

    simulation = Simulation(model; Δt, stop_time)

    progress(sim) = @info @sprintf("%s  %s  max|b - b₀| = %.3e",
                                   name, prettytime(sim), maximum(abs, interior(sim.model.tracers.b) .-
                                                                       interior(initial_buoyancy_field)))
    add_callback!(simulation, progress, IterationInterval(288))

    simulation.output_writers[:fields] =
        JLD2Writer(model, merge(model.tracers, (; b₀ = initial_buoyancy_field));
                   filename = "mixed_layer_front_$name",
                   schedule = TimeInterval(stop_time / 60),
                   overwrite_existing = true)

    run!(simulation)

    return FieldTimeSeries("mixed_layer_front_$name.jld2", "b")
end

triad_buoyancy = run_frontal_relaxation(triad_closure, "triad")
cox_buoyancy   = run_frontal_relaxation(cox_closure,   "cox")

#####
##### Animation
#####

times = triad_buoyancy.times
xkm   = xnodes(grid, Center()) ./ 1kilometers
zm    = znodes(grid, Center())
b₀    = interior(initial_buoyancy_field)[:, 1, :]

# The background stratification is time-independent, so subtracting it leaves the front itself, 0 → Δb.
background_stratification = [interior_stratification * min(z, -mixed_layer_depth) for x in xkm, z in zm]

frontal_structure(field) = (interior(field)[:, 1, :] .- background_stratification) ./ front_buoyancy_jump
spurious_change(field)   = (interior(field)[:, 1, :] .- b₀) ./ front_buoyancy_jump

triad_drift = [maximum(abs, spurious_change(triad_buoyancy[n])) for n in 1:length(times)]
cox_drift   = [maximum(abs, spurious_change(cox_buoyancy[n]))   for n in 1:length(times)]

n = Observable(1)

triad_front = @lift frontal_structure(triad_buoyancy[$n])
cox_front   = @lift frontal_structure(cox_buoyancy[$n])
triad_Δb    = @lift spurious_change(triad_buoyancy[$n])
cox_Δb      = @lift spurious_change(cox_buoyancy[$n])
title       = @lift @sprintf("mixed-layer front relaxed by isoneutral diffusion alone, no dynamics — %s",
                             prettytime(times[$n]))

fig = Figure(size = (1250, 1000))
Label(fig[0, 1:3], title, fontsize = 20, tellwidth = false)

axis_kwargs = (xlabel = "x [km]", ylabel = "z [m]",
               limits = ((0, horizontal_extent / 1kilometers), (-depth, 0)))

ax_triad_front = Axis(fig[1, 1]; title = "TriadIsopycnalSkewSymmetricDiffusivity", axis_kwargs...)
ax_cox_front   = Axis(fig[1, 2]; title = "IsopycnalSkewSymmetricDiffusivity", axis_kwargs...)

for (ax, front) in ((ax_triad_front, triad_front), (ax_cox_front, cox_front))
    hm = heatmap!(ax, xkm, zm, front, colormap = :thermal, colorrange = (0, 1))
    contour!(ax, xkm, zm, front, levels = range(0.05, 0.95, length = 10), color = (:white, 0.5), linewidth = 1)
    hlines!(ax, [-mixed_layer_depth], color = :cyan, linestyle = :dash, linewidth = 2)
    text!(ax, 3, -225, text = "neutral mixed layer:  ∂z b = 0", color = :cyan, fontsize = 14)
    ax === ax_cox_front && Colorbar(fig[1, 3], hm, label = "front structure (b - background) / Δb")
end

ax_triad_drift = Axis(fig[2, 1]; title = "spurious buoyancy change (b - b₀) / Δb", axis_kwargs...)
ax_cox_drift   = Axis(fig[2, 2]; title = "spurious buoyancy change (b - b₀) / Δb", axis_kwargs...)

for (ax, Δ) in ((ax_triad_drift, triad_Δb), (ax_cox_drift, cox_Δb))
    hm = heatmap!(ax, xkm, zm, Δ, colormap = :balance, colorrange = (-0.35, 0.35))
    hlines!(ax, [-mixed_layer_depth], color = :black, linestyle = :dash, linewidth = 2)
    ax === ax_cox_drift && Colorbar(fig[2, 3], hm)
end

ax_timeseries = Axis(fig[3, 1:3], xlabel = "time [days]", ylabel = "max |b - b₀| / Δb",
                     title = "b cannot change under an isoneutral closure — every departure from zero is spurious diapycnal mixing")
lines!(ax_timeseries, times ./ 1day, triad_drift, linewidth = 3, label = "triad (slope clipped to zero where ∂z b ≤ 0)")
lines!(ax_timeseries, times ./ 1day, cox_drift,   linewidth = 3, label = "non-triad (whole tensor switched off where ∂z b ≤ 0)")
vlines!(ax_timeseries, @lift(times[$n] / 1day), color = (:black, 0.4), linestyle = :dash)
axislegend(ax_timeseries, position = :lt)

rowsize!(fig.layout, 3, Relative(0.26))

record(fig, "mixed_layer_front_diapycnal_mixing.mp4", 1:length(times), framerate = 12) do i
    n[] = i
end

@info @sprintf("triad     erased %.1f%% of the front amplitude in %s", 100 * triad_drift[end], prettytime(stop_time))
@info @sprintf("non-triad erased %.1f%% of the front amplitude in %s", 100 * cox_drift[end], prettytime(stop_time))
