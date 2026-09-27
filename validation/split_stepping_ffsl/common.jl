# Diagnostics and runners for the validation of split tracer stepping with FluxFormSemiLagrangian slow tracers
# (Workstream C). The model setups live in test/setup/split_ffsl_test_utils.jl.

using Printf

include(joinpath(@__DIR__, "..", "..", "test", "setup", "split_ffsl_test_utils.jl"))

function initial_range(c, active)
    data = Array(interior(c))[active]
    return minimum(data), maximum(data)
end

function relative_overshoot(c, range, active)
    data = Array(interior(c))[active]
    lower, upper = range
    width = upper - lower
    return max(zero(width), maximum(data) - upper, lower - minimum(data)) / width
end

"""
    run_case(build_model, grid, ratio, Δt, steps; slow, fast, fold_rows = nothing)

Run `steps` dynamics steps, checking the slow tracers after every long step. Returns uniform deviation and
inventory drift (both maximized over long steps), the maximum long-step Courant numbers (proxy
`N Δt |ū| / Δx` for every scheme, and the volume-based swept Courant numbers of FFSL), the overshoot of
the smooth tracer beyond its initial range, the maximum net horizontal outflow of a cell during a long step
relative to its volume, and the model.
"""
function run_case(build_model, grid, ratio, Δt, steps; slow, fast = :FFSL, fold_rows = nothing)
    model = build_model(grid; ratio, slow, fast)
    active = active_cells(model.grid)
    inventory₀ = tracer_inventory(model.tracers.c)
    smooth_range = initial_range(model.tracers.smooth, active)
    N = isnothing(ratio) ? 1 : ratio

    uniform_deviation = 0.0
    fast_uniform_deviation = 0.0
    inventory_drift = 0.0
    proxy_courant = 0.0
    swept_courant = 0.0
    fold_courant = 0.0
    outflow_courant = 0.0
    finite = true

    for step in 1:steps
        time_step!(model, Δt)

        if step % N == 0
            if !isnothing(model.tracer_time_step_splitting)
                horizontal, _ = long_step_courant_numbers(model, N * Δt)
                proxy_courant = max(proxy_courant, horizontal)
                outflow_courant = max(outflow_courant, horizontal_outflow_courant_number(model, N * Δt))
                slow == :FFSL && (swept_courant = max(swept_courant, maximum(swept_courant_numbers(model))))
                if slow == :FFSL && !isnothing(fold_rows)
                    fold_courant = max(fold_courant, maximum(swept_courant_numbers(model; rows = fold_rows)))
                end
            end

            finite = all(isfinite, Array(interior(model.tracers.smooth))[active])
            finite || break

            uniform_deviation = max(uniform_deviation, maximum_uniform_deviation(model.tracers.constant, 1, active))
            fast_uniform_deviation = max(fast_uniform_deviation, maximum_uniform_deviation(model.tracers.fast, 1, active))
            inventory_drift = max(inventory_drift, abs(tracer_inventory(model.tracers.c) - inventory₀) / abs(inventory₀))
        end
    end

    overshoot = finite ? relative_overshoot(model.tracers.smooth, smooth_range, active) : Inf

    return (; model, active, finite, uniform_deviation, fast_uniform_deviation, inventory_drift,
              proxy_courant, swept_courant, fold_courant, outflow_courant, overshoot)
end

# A run is stable if the smooth slow tracer stays finite and its overshoot beyond its initial bounds exceeds the overshoot
# of the `ratio = 1` run with the same slow scheme by less than 10% of the initial range.
is_stable(result, reference) = result.finite && result.overshoot < reference.overshoot + 0.1
