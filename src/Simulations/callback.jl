using Dates: Dates
using Oceananigans: Oceananigans, initialize!, prognostic_state, restore_prognostic_state!,
                    TimeStepCallsite, TendencyCallsite, UpdateStateCallsite
using Oceananigans.OutputWriters: WindowedTimeAverage, advance_time_average!, TimeDerivative
using Oceananigans.Utils: prettysummary

struct Callback{P, F, S, CS}
    func :: F
    schedule :: S
    callsite :: CS
    parameters :: P
end

@inline (callback::Callback)(sim) = callback.func(sim, callback.parameters)
@inline (callback::Callback{<:Nothing})(sim) = callback.func(sim)

"""
$(TYPEDSIGNATURES)

Initialize `callback` at the beginning of `run!(sim)`.
By default, this calls `initialize!` on `callback.func`,
which in turn does nothing by default.

`initialize!` can be specialized on `callback.parameters`,
or specialized for `callback.func`.
`
"""
Oceananigans.initialize!(callback::Callback, sim) = initialize!(callback.func, sim)

"""
$(TYPEDSIGNATURES)

Finalize `callback` at the end of `run!(sim)`.
By default, this calls `finalize!` on `callback.func`,
which in turn does nothing by default.

`finalize!` can be specialized on `callback.parameters`,
or specialized for `callback.func`.
"""
finalize!(callback::Callback, sim) = finalize!(callback.func, sim)

Oceananigans.initialize!(func, sim) = nothing
finalize!(func, sim) = nothing

"""
    Callback(func, schedule=IterationInterval(1);
             parameters=nothing, callsite=TimeStepCallsite())

Return `Callback` that executes `func` on `schedule`
at the `callsite` with optional `parameters`. By default,
`schedule = IterationInterval(1)` and `callsite = TimeStepCallsite()`.

If `isnothing(parameters)`, `func(sim::Simulation)` is called.
Otherwise, `func` is called via `func(sim::Simulation, parameters)`.

The `callsite` determines where `Callback` is executed. The possible values for
`callsite` are:

* `TimeStepCallsite()`: after a time-step.

* `TendencyCallsite()`: after tendencies are calculated, but before taking
  a time-step (useful for modifying tendency calculations).

* `UpdateStateCallsite()`: within `update_state!`, after auxiliary variables have
  been computed (for multi-stage time-steppers, `update_state!` may be called multiple
  times per time-step).
"""
function Callback(func, schedule=IterationInterval(1);
                  parameters = nothing,
                  callsite = TimeStepCallsite())

    return Callback(func, schedule, callsite, parameters)
end

Base.summary(cb::Callback{Nothing}) = string("Callback of ", prettysummary(cb.func, false), " on ", summary(cb.schedule))
Base.summary(cb::Callback) = string("Callback of ", prettysummary(cb.func, false), " on ", summary(cb.schedule),
                                    " with parameters ", cb.parameters)

Base.show(io::IO, cb::Callback) = print(io, summary(cb))

function Callback(wta::WindowedTimeAverage)
    function func(sim)
        model = sim.model
        advance_time_average!(wta, model)
        return nothing
    end
    return Callback(func, wta.schedule, nothing)
end

Callback(wta::WindowedTimeAverage, schedule; kw...) =
    throw(ArgumentError("Schedule must be inferred from WindowedTimeAverage.
                        Use Callback(windowed_time_average)"))

const TimeDerivativeCallback = Callback{<:Any, <:TimeDerivative}

"""
    TimeDerivativeCallback(operand, model=nothing; schedule=IterationInterval(1))

Return a [`Callback`](@ref) that updates a [`TimeDerivative`](@ref) of `operand` on
`schedule`, so that the derivative is differenced over the interval between actuations.
The derivative itself is `callback.func`, whose `result` is a `Field`.

Example
=======

```jldoctest
using Oceananigans

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))

model = NonhydrostaticModel(grid)

simulation = Simulation(model, Δt=1e-2, stop_iteration=10)

simulation.callbacks[:∂ₜu] = TimeDerivativeCallback(model.velocities.u, schedule=TimeInterval(0.1))

# output
Callback of TimeDerivative of 4×4×4 Field{Face, Center, Center} on RectilinearGrid on CPU on TimeInterval(100 ms)
```
"""
TimeDerivativeCallback(operand, model=nothing; schedule=IterationInterval(1)) =
    Callback(TimeDerivative(operand, model), schedule)

function Oceananigans.prognostic_state(callback::TimeDerivativeCallback)
    return (schedule = prognostic_state(callback.schedule),
            time_derivative = prognostic_state(callback.func))
end

function Oceananigans.restore_prognostic_state!(restored::TimeDerivativeCallback, from)
    restore_prognostic_state!(restored.schedule, from.schedule)
    restore_prognostic_state!(restored.func, from.time_derivative)
    return restored
end

Oceananigans.restore_prognostic_state!(::TimeDerivativeCallback, ::Nothing) = nothing

struct GenericName end

generic_callback_name(name, existing_names) = name

function generic_callback_name(::GenericName, existing_names)
    prefix = :callback # yeah, that's generic

    # Find a unique one
    n = 1
    while Symbol(prefix, n) ∈ existing_names
        n += 1
    end

    return Symbol(prefix, n)
end

"""
    add_callback!(simulation, callback::Callback; name = GenericName())
    add_callback!(simulation, func, schedule=IterationInterval(1); name = GenericName(), callback_kw...)

Add `Callback(func, schedule)` to `simulation.callbacks` under `name`. The default
`GenericName()` generates a name of the form `:callbackN`, where `N`
is big enough for the name to be unique.

If `name::Symbol` is supplied, it may be modified if `simulation.callbacks[name]`
already exists.

`callback_kw` are passed to the constructor for [`Callback`](@ref).

The `callback` (which contains a schedule) can also be supplied directly.
"""
function add_callback!(simulation, callback::Callback; name = GenericName())
    name = generic_callback_name(name, keys(simulation.callbacks))
    simulation.callbacks[name] = callback
    return nothing
end

function add_callback!(simulation, func, schedule = IterationInterval(1);
                       name = GenericName(), callback_kw...)

    callback = Callback(func, schedule; callback_kw...)
    return add_callback!(simulation, callback; name)
end

validate_schedule(func, schedule) = schedule

function Oceananigans.prognostic_state(callback::Callback)
    return (; schedule = prognostic_state(callback.schedule))
end

function Oceananigans.restore_prognostic_state!(restored::Callback, from)
    restore_prognostic_state!(restored.schedule, from.schedule)
    return restored
end

Oceananigans.restore_prognostic_state!(::Callback, ::Nothing) = nothing
