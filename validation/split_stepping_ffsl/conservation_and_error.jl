# Split tracer stepping with FluxFormSemiLagrangian slow tracers (gate C): conservation, error and stability vs N.
#
# The dynamics time step Δt is fixed, so the long-step Courant number grows as N Δt. For every ratio N and for slow
# tracers advected with FFSL or WENO5 (both with the same adaptive implicit vertical WENO5) the script reports:
#   * the maximum long-step Courant numbers: the proxy N Δt |ū| / Δx, and for FFSL the volume-based swept Courant
#     number that the scheme uses (and, on the tripolar grid, its value in the rows next to the fold);
#   * D, the maximum net horizontal outflow of a cell during a long step relative to its volume (above 1 the
#     thickness after the horizontal FFSL step is negative);
#   * the maximum over long steps of the deviation of a uniform slow tracer and of the drift of ∫σc dV of a random
#     slow tracer, and the deviation of a uniform fast FFSL tracer;
#   * the L² and L∞ errors of a smooth slow tracer against the `ratio = 1` WENO5 run, and the L² error against the
#     `ratio = 1` run with the same slow scheme, all at the same final time;
#   * the overshoot of the smooth tracer beyond its initial range (relative to that range) and whether the run is
#     stable (finite, and overshoot within 10% of the overshoot of the `ratio = 1` run with the same slow scheme).
#
# Usage: julia --project validation/split_stepping_ffsl/conservation_and_error.jl [gyre|ridge|baroclinic|tripolar|all]

include(joinpath(@__DIR__, "common.jl"))

ratios = (1, 2, 4, 8, 16, 32, 64)

function error_table(name, build_model, grid, Δt, steps; fold_rows = nothing)
    println()
    println("### ", name, " (Δt = ", prettytime(Δt), ", ", steps, " steps, end time ", prettytime(steps * Δt), ")")
    println()
    println("| slow scheme |  N | C proxy | C swept | C swept at fold | D | uniform dev. | ∫σc drift | fast uniform dev. | L² vs WENO5 N=1 | L∞ vs WENO5 N=1 | L² vs same N=1 | overshoot | stable |")
    println("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")

    maximum_stable_ratio = Dict{Symbol, Int}()
    reference = nothing

    for slow in (:WENO, :FFSL)
        same_scheme_reference = nothing
        for ratio in ratios
            result = run_case(build_model, grid, ratio, Δt, steps; slow, fold_rows)
            ratio == 1 && (same_scheme_reference = result)
            ratio == 1 && slow == :WENO && (reference = result)

            if result.finite
                L², L∞ = relative_errors(result.model.tracers.smooth, reference.model.tracers.smooth, result.active)
                L²ˢ, _ = relative_errors(result.model.tracers.smooth, same_scheme_reference.model.tracers.smooth, result.active)
            else
                L², L∞, L²ˢ = NaN, NaN, NaN
            end

            stable = is_stable(result, same_scheme_reference)
            stable && (maximum_stable_ratio[slow] = max(get(maximum_stable_ratio, slow, 0), ratio))

            @printf("| %s | %2d | %.2f | %s | %s | %.2f | %.2e | %.2e | %.2e | %.2e | %.2e | %.2e | %.2e | %s |\n",
                    slow, ratio, result.proxy_courant,
                    slow == :FFSL ? @sprintf("%.2f", result.swept_courant) : "-",
                    slow == :FFSL && !isnothing(fold_rows) ? @sprintf("%.2f", result.fold_courant) : "-",
                    result.outflow_courant, result.uniform_deviation, result.inventory_drift, result.fast_uniform_deviation,
                    L², L∞, L²ˢ, result.overshoot, stable ? "yes" : "no")
            flush(stdout)
        end
    end

    println()
    println("Maximum stable N: WENO5 ", get(maximum_stable_ratio, :WENO, 0), ", FFSL ", get(maximum_stable_ratio, :FFSL, 0))
    return nothing
end

case = isempty(ARGS) ? "all" : ARGS[1]

if case ∈ ("gyre", "all")
    println()
    println("## Rectilinear basin with an immersed ridge, z-star, steady gyre going around the ridge (non-divergent)")
    error_table("RectilinearGrid 16×8×6, GridFittedBottom ridge", nondivergent_gyre_model, rectilinear_basin(), 3minutes, 640)
end

if case ∈ ("ridge", "all")
    println()
    println("## Rectilinear basin with an immersed ridge, z-star, depth-uniform gyre forced across the ridge (divergent)")
    error_table("RectilinearGrid 16×8×6, GridFittedBottom ridge", gyre_basin_model, rectilinear_basin(), 3minutes, 640)
end

if case ∈ ("baroclinic", "all")
    println()
    println("## Rectilinear basin with an immersed ridge, z-star baroclinic adjustment (transient)")
    error_table("RectilinearGrid 16×8×6, GridFittedBottom ridge", baroclinic_basin_model, rectilinear_basin(), 3minutes, 640)
end

if case ∈ ("tripolar", "all")
    println()
    println("## TripolarGrid with immersed islands, z-star, barotropic flow across the fold")
    for fold_topology in (RightCenterFolded, RightFaceFolded)
        grid = tripolar_basin(; fold_topology)
        Ny = size(grid, 2)
        error_table("$fold_topology TripolarGrid 20×32×3", tripolar_rotation_model, grid, 3hours, 640; fold_rows = Ny-1:Ny+1)
    end
end
