# Split tracer time stepping with Strang-split, sub-cycled biogeochemical sources (gate A3).
#
# A minimal NPZD model with sinking detritus and a fast (stiff) remineralization rate r is carried as the slow
# group of a z-star baroclinic adjustment. With Δt = 5 minutes, r Δt = 0.3 but r 16 Δt = 4.8, beyond the
# stability limit (≈ 2.5) of the third-order Runge-Kutta step on the negative real axis.
#
# For every ratio N and number of source sub-steps M the script reports the total nitrogen drift, the
# L² and L∞ error of detritus against the run with `ratio = 1` and sources in the Runge-Kutta step, and
# the maximum horizontal and vertical (flow plus sinking) Courant numbers of the long step.
#
# Usage: julia --project validation/split_tracer_stepping/biogeochemistry_subcycling.jl

using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: MutableVerticalDiscretization

include(joinpath(@__DIR__, "..", "..", "test", "setup", "split_tracer_stepping_test_utils.jl"))

ridge(x₀, width, height, depth) = (x, y) -> - depth + height * exp(- ((x - x₀) / width)^2)

z = MutableVerticalDiscretization(collect(range(-60, 0, length=7)))
underlying_grid = RectilinearGrid(size = (16, 8, 6), halo = (4, 4, 4), x = (0, 32kilometers), y = (0, 16kilometers), z = z,
                                  topology = (Bounded, Bounded, Bounded))
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge(16kilometers, 4kilometers, 30, 60)))

Δt = 5minutes
steps = 64
remineralization_rate = 1 / 1000seconds
sinking_speed = 20meters / day

function run_npzd(ratio, substeps)
    model = npzd_model(deepcopy(grid); ratio, biogeochemistry_substeps = substeps, remineralization_rate, sinking_speed)
    total₀ = total_nitrogen(model)
    drift = model.biogeochemistry.sinking_velocity.w
    horizontal_courant = 0.0
    vertical_courant = 0.0
    stable = true

    for step in 1:steps
        time_step!(model, Δt)
        if step % ratio == 0
            horizontal, vertical = long_step_courant_numbers(model, ratio * Δt; drift)
            horizontal_courant = max(horizontal_courant, horizontal)
            vertical_courant = max(vertical_courant, vertical)
        end
        # Unstable: non-finite, or an order of magnitude beyond the initial range of any NPZD tracer
        if any(name -> !all(isfinite, Array(interior(model.tracers[name]))) || maximum(abs, Array(interior(model.tracers[name]))) > 50,
               (:N, :P, :Z, :D))
            stable = false
            break
        end
    end

    nitrogen_drift = stable ? abs(total_nitrogen(model) - total₀) / total₀ : NaN
    return (; model, stable, nitrogen_drift, horizontal_courant, vertical_courant)
end

reference = run_npzd(1, nothing)
active = active_cells(reference.model.grid)

println("## A3: Strang-split biogeochemical sources (r Δt = ", remineralization_rate * Δt, ", Δt = ", prettytime(Δt), ", ", steps, " steps)")
println()
println("|  N | M | stable | total N drift | L² error (D) | L∞ error (D) | max C (h) | max C (v, with sinking) |")
println("|---:|---:|:---:|---:|---:|---:|---:|---:|")

for ratio in (1, 2, 4, 8, 16), substeps in (nothing, 1, 2, 4, 8)
    result = run_npzd(ratio, substeps)
    if result.stable
        L², L∞ = relative_errors(result.model.tracers.D, reference.model.tracers.D, active)
    else
        L², L∞ = NaN, NaN
    end
    M = isnothing(substeps) ? "in RK" : string(substeps)
    @printf("| %2d | %s | %s | %.2e | %.2e | %.2e | %.3f | %.3f |\n", ratio, M, result.stable ? "yes" : "**no**",
            result.nitrogen_drift, L², L∞, result.horizontal_courant, result.vertical_courant)
end
