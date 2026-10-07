#####
##### Running a ReactantSimulation: the compiled counterpart of src/Simulations/run.jl
#####
##### Same division of labour as the eager code, so that `@compile run!(sim)` is one program:
#####
#####   initialize!(sim)          iteration 0: initialize the model, refresh its state, initialize
#####                             and fire the callbacks
#####   time_step!(sim)           one step: the model step with its callsite callbacks, then the
#####                             TimeStepCallsite callbacks whose schedule says so
#####   time_step_for!(sim, N)    N steps in one traced loop
#####   run!(sim)                 initialize!, the whole steps, the remainder step if stop_time is
#####                             not a whole number of steps away, then finalize! every callback
#####
##### Not yet available inside a program: writers, diagnostics, stop criteria on the state,
##### wall time.
#####

# The callbacks at a callsite, as a Tuple so the loop over them unrolls at trace time and each
# callback keeps its own concrete type.
callbacks_at(sim, Callsite) = Tuple(cb for cb in values(sim.callbacks) if cb.callsite isa Callsite)

# Invoke a callback from inside a traced `if` through this plain function, never as `callback(sim)`
# directly. `@trace if` collects the variables its branch captures with ExpressionExplorer, which
# files a symbol in call position under `funccalls`, not `references`. A `Callback` invoked as a
# functor is therefore not captured as a branch input: the branch closes over the enclosing
# object, and a mutation of its parameters inside the branch escapes the region (silently lost,
# or "operand does not dominate this use"). In argument position it is captured and written back.
invoke_callback(callback, sim) = callback(sim)

#####
##### Schedules: what can fire inside a program
#####

"""
    add_callback!(sim::ReactantSimulation, callback::Callback; name = GenericName())

Add `callback` to `sim.callbacks`, with its schedule converted to one a program can evaluate.

A schedule fires inside the compiled loop as a traced predicate on the clock, so it must be
evaluable there. `IterationInterval` is as it stands. `TimeInterval` mutates a host-side
actuation counter when it fires, so it is converted here to the `IterationInterval` it equals
under a fixed `Δt`, counted from the simulation's current iteration; an interval that is not a
whole number of steps is an error, since a schedule has no remainder step to take.
`SpecifiedTimes` remembers which time comes next in a host `Int`, so it is rebuilt with its times
on the device, where its call needs no memory: with a fixed step, "the first step at or after
`t`" is the step whose `(time − Δt, time]` contains `t` (see [`DeviceSpecifiedTimes`](@ref)).
Other schedules are refused. The `add_callback!(sim, func, schedule; kw...)` form routes through
here.

The callback's function is checked too. Of the functions Oceananigans itself puts in callbacks,
four cannot run inside a program and are refused here with the reason, rather than failing inside
`@compile`: a `TimeStepWizard` changes `Δt`, which a program fixes and uses as its loop bound; a
`NaNChecker` raises a data-dependent error; the eager stop criteria (`stop_iteration_exceeded`,
`stop_time_exceeded`, `wall_time_limit_exceeded`) compare the traced clock on the host to set
`sim.running`, which the loop bound has replaced; a `TimeDerivative` branches on the traced clock
and stores it. Neither check sees a callback assigned into `sim.callbacks` directly, as
`conjure_time_step_wizard!` does.
"""
function add_callback!(sim::ReactantSimulation, callback::Callback; name = GenericName())
    check_traceable(callback.func)
    schedule = traced_schedule(callback.schedule, sim)
    callback = Callback(callback.func, schedule, callback.callsite, callback.parameters)
    return invoke(add_callback!, Tuple{Any, Callback}, sim, callback; name)
end

traced_schedule(schedule::IterationInterval, sim) = schedule

function traced_schedule(schedule::TimeInterval, sim)
    interval = schedule.interval
    m = whole_steps(interval, sim.Δt)
    isnothing(m) && throw(ArgumentError(
        "TimeInterval($interval) is not a whole number of steps of Δt = $(sim.Δt): a compiled run " *
        "cannot align its step to a schedule. Choose an interval that Δt divides, or an IterationInterval."))
    return IterationInterval(m; offset = iteration(sim))
end

"""
    DeviceSpecifiedTimes

A `SpecifiedTimes` whose times live on the device: what `traced_schedule` makes of an eager one,
before and during tracing. The eager schedule remembers which of its times comes next in
`previous_actuation` and advances it inside an `if`, neither of which can live inside a program.
Inside one the step is fixed, so no memory is needed: time `t` fires on exactly the step whose
interval `(time − last_Δt, time]` contains it, which is the eager "first step at or after `t`".
The call below is one comparison of the traced clock against the times, reduced with `|`, and
returns the traced `Bool` that `time_step!`'s traced `if` consumes. Keeping the times on the
device rather than in the program lets a run with different times reuse the compiled program.
"""
const DeviceSpecifiedTimes = SpecifiedTimes{<:Any, <:Union{AnyConcreteReactantArray, Reactant.AnyTracedRArray}}

function traced_schedule(schedule::SpecifiedTimes, sim)
    eltype(schedule.times) <: Number || throw(ArgumentError(
        "SpecifiedTimes with $(eltype(schedule.times)) times cannot fire inside a compiled run: the clock " *
        "is a number there. Give the times in simulation seconds."))
    FT = eltype(schedule.times)
    times = Reactant.to_rarray(copy(schedule.times))
    return SpecifiedTimes{FT, typeof(times)}(times, schedule.previous_actuation)   # FT is the times' float type, not the device array's traced eltype
end

function (schedule::SpecifiedTimes{<:Any, <:Union{AnyConcreteReactantArray, Reactant.AnyTracedRArray}})(model)
    times = schedule.times
    t = model.clock.time
    t₋ = t - model.clock.last_Δt                           # the step just taken is (t₋, t]
    in_step = (times .> t₋) .& (times .<= t)
    return any(in_step)   # a `|` over the times, as a traced Bool
end

traced_schedule(schedule, sim) = throw(ArgumentError(
    "$(typeof(schedule)) cannot fire inside a compiled run: a schedule there must be evaluable on the " *
    "traced clock. Use IterationInterval, a TimeInterval that Δt divides, or SpecifiedTimes."))

# Callback functions that cannot run inside a program. Anything else is admitted on trust: the
# rule it must follow (device work only) is in the `time_step!` docstring.
check_traceable(func) = nothing

check_traceable(::TimeStepWizard) = throw(ArgumentError(
    "A TimeStepWizard cannot fire inside a compiled run: Δt is fixed for the program and bounds its " *
    "loop. Adapt Δt between programs, by constructing a Simulation with the new Δt."))

check_traceable(::NaNChecker) = throw(ArgumentError(
    "A NaNChecker cannot fire inside a compiled run: its data-dependent error cannot exist inside a " *
    "program. Check the fields for NaNs on the host after the compiled call returns."))

check_traceable(func::Union{typeof(stop_iteration_exceeded), typeof(stop_time_exceeded), typeof(wall_time_limit_exceeded)}) = throw(ArgumentError(
    "$func cannot fire inside a compiled run: it compares the traced clock on the host to stop the " *
    "eager loop, and a compiled run is bounded by stop_iteration or stop_time at construction instead."))

check_traceable(::TimeDerivative) = throw(ArgumentError(
    "A TimeDerivative cannot be updated inside a compiled run: it branches on the traced clock and stores " *
    "it as a plain number. Difference the field between programs, or accumulate it in a callback of your own."))

#####
##### Stepping
#####

"""
    initialize!(sim::ReactantSimulation)

Iteration 0, as the eager `initialize!(sim)` does it: `initialize!(model)`, `update_state!` with
the model-callsite callbacks, `initialize!(callback, sim)` on every callback, then every
`TimeStepCallsite` callback once, unconditionally.

`initialize!(callback, sim)` forwards to `initialize!(callback.func, sim)`, a no-op unless the
callback's function type specializes it. It runs here outside the traced loop, on the initial
state, so it may read or write device arrays freely; its counterpart at the end of the run is
`finalize!`, see [`run!`](@ref).
"""
function initialize!(sim::ReactantSimulation)
    model = sim.model
    initialize!(model)
    update_state!(model, callbacks_at(sim, ModelCallsite))
    foreach(callback -> initialize!(callback, sim), values(sim.callbacks))
    foreach(callback -> callback(sim), callbacks_at(sim, TimeStepCallsite))
    sim.initialized = true
    return nothing
end

# The step of a Reactant model, with the Euler switch only the Adams-Bashforth stepper has.
# Eagerly, AB2 takes an Euler step whenever `Δt` differs from the last step's, which covers the
# first step and an aligned last step; the Reactant `time_step!` for AB2 needs to be told, so
# `run!` says so for the first step and the remainder step.
reactant_model_time_step!(model, Δt, callbacks, euler) = time_step!(model, Δt; callbacks)
reactant_model_time_step!(model::ReactantModel{<:QuasiAdamsBashforth2TimeStepper}, Δt, callbacks, euler) =
    time_step!(model, Δt; callbacks, euler)

"""
    time_step!(sim::ReactantSimulation; euler = false)
    time_step!(sim::ReactantSimulation, Δt; euler = false)

One step of `sim.Δt`, or of `Δt`: `time_step!(model, Δt; callbacks)` with the `TendencyCallsite`
and `UpdateStateCallsite` callbacks, then each `TimeStepCallsite` callback whose schedule fires.
`euler = true` makes an Adams-Bashforth step a forward Euler one, as the first step and the
remainder step must be. Unlike the eager `time_step!(sim, Δt)`, the two-argument form does not
store `Δt` on `sim`.

Inside a program the schedule evaluates to a traced `Bool` on the traced clock, so the firing is
a traced `if` rather than the eager `schedule(model) && callback(sim)`. A callback that fires
here must do only device work (kernel launches, broadcasts, reductions into arrays reachable
from `sim`). Printing, IO, and data-dependent errors belong between programs.
"""
time_step!(sim::ReactantSimulation; euler = false) = time_step!(sim, sim.Δt; euler)

function time_step!(sim::ReactantSimulation, Δt; euler = false)
    model = sim.model
    reactant_model_time_step!(model, Δt, callbacks_at(sim, ModelCallsite), euler)

    # `track_numbers = false`: the branch captures `sim`, and promoting the plain numbers inside
    # the model to traced numbers fails on structs whose type parameters do not cover every
    # numeric field.
    for callback in callbacks_at(sim, TimeStepCallsite)
        actuate = callback.schedule(model)
        @trace track_numbers = false if actuate
            invoke_callback(callback, sim)
        end
    end

    return nothing
end

"""
    time_step_for!(sim::ReactantSimulation, Nsteps)

`Nsteps` calls to `time_step!(sim)` as one traced loop: one `stablehlo.while` body rather than
`Nsteps` copies of the step. Does not initialize; see [`run!`](@ref).

The loop is traced with the `track_numbers`, `mincut` and `checkpointing` options of
`sim.automatic_differentiation`, which only matter when the program is differentiated in
reverse mode. See [`AutomaticDifferentiation`](@ref).
"""
time_step_for!(sim::ReactantSimulation, Nsteps) =
    time_step_for!(sim, Nsteps, Val(sim.automatic_differentiation.track_numbers))

# `@trace` needs `track_numbers` as a literal: it picks the promotion type at macro expansion.
function time_step_for!(sim::ReactantSimulation, Nsteps, ::Val{false})
    ad = sim.automatic_differentiation
    @trace track_numbers = false mincut = ad.mincut checkpointing = ad.checkpointing for _ = 1:Nsteps
        time_step!(sim)
    end
    return nothing
end

function time_step_for!(sim::ReactantSimulation, Nsteps, ::Val{true})
    ad = sim.automatic_differentiation
    @trace track_numbers = true mincut = ad.mincut checkpointing = ad.checkpointing for _ = 1:Nsteps
        time_step!(sim)
    end
    return nothing
end

# The final step onto `stop_time` when Δt does not divide the interval: the eager aligned last
# step, with its size decided at construction (see `stop_criteria`) rather than per step.
remainder_step!(sim, ::Nothing) = nothing
remainder_step!(sim, step::RemainderStep) = time_step!(sim, step.Δt; euler = true)

"""
    run!(sim::ReactantSimulation)

`initialize!(sim)`, a first step (forward Euler for an Adams-Bashforth stepper, as eagerly),
the remaining whole steps through [`time_step_for!`](@ref), one remainder step when the
simulation was given a `stop_time` that `Δt` does not divide, and then `finalize!(callback, sim)`
on every callback.

`finalize!(callback, sim)` forwards to `finalize!(callback.func, sim)`, a no-op unless the
callback's function type specializes it. It runs once, after the traced loop, on the final
state: the place for a terminal quantity such as a loss or an end-of-run error norm, which then
needs neither a schedule nor a traced branch. Like everything inside the program it must do only
device work; printing and IO belong after the compiled call returns.

`stop_iteration` is a plain number on the `Simulation`, so under `@compile` it is a static loop
bound. It counts steps from the state the simulation was constructed in: a run continued from
iteration `n` steps `stop_iteration` more times.
"""
function run!(sim::ReactantSimulation)
    isfinite(sim.stop_iteration) || throw(ArgumentError(
        "run! on a ReactantState model needs `stop_iteration` or `stop_time`: they bound the compiled loop."))

    N = Int(sim.stop_iteration)

    initialize!(sim)
    N ≥ 1 && time_step!(sim; euler = true)
    time_step_for!(sim, N - 1)
    remainder_step!(sim, sim.stop_time)
    foreach(callback -> finalize!(callback, sim), values(sim.callbacks))

    return nothing
end
