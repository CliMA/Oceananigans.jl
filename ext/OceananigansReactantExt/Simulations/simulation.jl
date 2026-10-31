const ReactantSimulation = Simulation{<:ReactantModel}

validate_checkpointing(::Union{Periodic, Binomial}) = nothing

validate_checkpointing(checkpointing) = checkpointing === false ? nothing : throw(ArgumentError(
    "checkpointing = $checkpointing is not supported: use `Reactant.Periodic(n)`, " *
    "`Reactant.Binomial(budget)`, or `false` for no checkpointing."))

"""
    Simulation(model::ReactantModel; Δt, stop_iteration = Inf, stop_time = nothing, verbose = true,
               wall_time_limit = Inf, align_time_step = false, minimum_relative_step = 0,
               automatic_differentiation = AutomaticDifferentiation(checkpointing = false))

A `Simulation` of a model on `ReactantState`, meant to be compiled: `@compile run!(sim)` is one
program that steps the model to the stop criterion and fires `sim.callbacks` inside the loop
(see [`run!`](@ref)).

What differs from the eager `Simulation`:

- `Δt` is fixed. Adaptive stepping changes `Δt` between programs, not inside one.
- The stop criterion is resolved here, on the host, into a whole number of steps, because
  inside a program the clock is traced and cannot bound a loop. Give `stop_iteration`, or
  `stop_time`, which is converted from the model's current time: when `Δt` divides the interval
  the run is that many steps; otherwise a warning says so and [`run!`](@ref) takes the whole
  steps and then one final step of exactly the remaining time, as the eager `run!` does with
  `align_time_step`. That final step is a [`RemainderStep`](@ref) stored in `sim.stop_time`,
  sized on the host here, since the model's kernels take a plain-number step. Not both.
- `stop_iteration` is held as a device number, the step count of [`run!`](@ref), which a compiled
  `run!` reads at each call: setting `sim.stop_iteration = n` between calls runs `n` steps with
  the same compiled program, without recompiling. A remainder step stays as it was sized here.
- `callbacks` starts empty. The eager constructor installs the stop-criterion callbacks, which
  here are the loop bound, and a NaN checker, whose data-dependent error cannot exist inside a
  program. `add_callback!` works as usual, converting a `TimeInterval` to an `IterationInterval`;
  see [`time_step!`](@ref) for what a callback may do.
- `output_writers` and `diagnostics` are `nothing`: IO cannot happen inside a program.
- `automatic_differentiation` holds the options of the traced step loop that matter when `run!`
  is differentiated in reverse mode: `track_numbers`, `mincut`, and `checkpointing` (`false`,
  `Reactant.Periodic(n)` or `Reactant.Binomial(budget)`), as a `NamedTuple` of those fields or an
  [`AutomaticDifferentiation`](@ref). See [`time_step_for!`](@ref).
"""
function Simulation(model::ReactantModel; Δt,
                    verbose = true,
                    stop_iteration = Inf,
                    stop_time = nothing,
                    wall_time_limit = Inf,
                    align_time_step = false,
                    minimum_relative_step = 0,
                    automatic_differentiation = AutomaticDifferentiation(checkpointing = false))

    Δt = Float64(Δt)
    automatic_differentiation = convert(AutomaticDifferentiation, automatic_differentiation)
    validate_checkpointing(automatic_differentiation.checkpointing)

    if !isnothing(stop_time)
        isfinite(stop_iteration) && throw(ArgumentError(
            "Simulation on a ReactantState model: give `stop_iteration` or `stop_time`, not both."))
        stop_iteration, stop_time = stop_criteria(model, Δt, stop_time)
    end

    diagnostics = nothing
    output_writers = nothing
    callbacks = OrderedDict{Symbol, Callback}()
    stop_iteration = isfinite(stop_iteration) ? device_step_count(model, stop_iteration) : nothing

    return Simulation(model,
                      Δt,
                      stop_iteration,
                      stop_time,
                      Float64(wall_time_limit),
                      diagnostics,
                      output_writers,
                      callbacks,
                      0.0,
                      align_time_step,
                      false,
                      false,
                      verbose,
                      Float64(minimum_relative_step),
                      automatic_differentiation)
end

# The step count as a device number, replicated across the devices of a distributed model as its
# clock is, so that it is an input of the compiled program rather than a constant of it.
function device_step_count(model, steps)
    steps ≥ 0 || throw(ArgumentError("stop_iteration = $steps: a compiled run! takes a nonnegative number of steps."))
    arch = architecture(model)
    sharding = arch isa Distributed ? Sharding.Replicated(arch.connectivity) : Sharding.NoSharding()
    return ConcreteRNumber(Int(steps); sharding)
end

function Base.setproperty!(sim::ReactantSimulation, name::Symbol, value)
    if name === :stop_iteration && !(value isa ConcreteRNumber)
        value = device_step_count(sim.model, value)
    end
    return setfield!(sim, name, convert(fieldtype(typeof(sim), name), value))
end

"""
    whole_steps(interval, Δt) -> Int or nothing

The number of steps of `Δt` in `interval` when `Δt` divides it to within roundoff, else `nothing`.
The tolerance is `sqrt(eps)` relative: a remainder below that is roundoff from accumulating
`Δt`, not an interval the user meant to be off the step grid.
"""
function whole_steps(interval, Δt)
    n = round(Int, interval / Δt)
    return isapprox(n * Δt, interval; rtol = sqrt(eps(Float64))) ? n : nothing
end

"""
    RemainderStep(Δt)

The final step of a compiled run whose `stop_time` is not a whole number of steps away: `Δt`
is the remaining time after the whole steps. Held in `sim.stop_time`, so `sim.stop_time`
still means "the run ends here". Sized on the host at construction, because the model's
kernels take a plain-number step, not a traced one.
"""
struct RemainderStep{FT}
    Δt :: FT
end

Oceananigans.Utils.prettytime(step::RemainderStep) = "a remainder step of " * prettytime(step.Δt)

"""
    stop_criteria(model, Δt, stop_time) -> (stop_iteration, remainder)

Resolve `stop_time` into the loop bound of [`run!`](@ref), from the model's current clock time.

Returns `(N, nothing)` when `Δt` divides the interval: `N` whole steps land on `stop_time`.
Otherwise warns and returns `(N, RemainderStep(r))` with `N` the whole steps that fit and `r`
the time left after them, which `run!` crosses in one final step.
"""
function stop_criteria(model, Δt, stop_time)
    t₀ = Reactant.to_number(model.clock.time)
    interval = Float64(stop_time) - t₀
    interval > 0 || throw(ArgumentError(
        "Simulation on a ReactantState model: stop_time = $stop_time is not after the model's time $t₀."))

    N = whole_steps(interval, Δt)
    isnothing(N) || return (Float64(N), nothing)

    N = floor(Int, interval / Δt)
    remainder = interval - N * Δt
    @warn "Δt = $Δt does not divide stop_time - t₀ = $interval: run! will take $N steps of Δt and " *
          "then one step of $remainder to land on stop_time = $stop_time."
    return (Float64(N), RemainderStep(Float64(remainder)))
end

iteration(sim::ReactantSimulation) = Reactant.to_number(iteration(sim.model))
time(sim::ReactantSimulation) = Reactant.to_number(time(sim.model))
