using Oceananigans.Utils: prettysummary
using Oceananigans.OutputWriters: WindowedTimeAverage, advance_time_average!
using Oceananigans: TimeStepCallsite, TendencyCallsite, UpdateStateCallsite

import Oceananigans: initialize!

struct Callback{P, F, S, CS}
    func :: F
    schedule :: S
    parameters :: P
    callsite :: CS
end

@inline (callback::Callback)(sim) = callback.func(sim, callback.parameters)
@inline (callback::Callback{<:Nothing})(sim) = callback.func(sim)

# Fallback initialization: call the schedule, then the callback
function initialize!(callback::Callback, sim)
    initialize!(callback.schedule, sim.model) && callback(sim)
    return nothing
end

"""
    Callback(func, schedule=IterationInterval(1);
             parameters=nothing, callsite=TimeStepCallsite())

Return `Callback` that executes `func` on `schedule`
at the `callsite` with optional `parameters`. By default,
`schedule = IterationInterval(1)` and `callsite = TimeStepCallsite()`.

If `isnothing(parameters)`, `func(sim::Simulation)` is called.
Otherwise, `func` is called via `func(sim::Simulation, parameters)`.

The `callsite` determines where `Callback` is executed. The possible values for 
`callsite` are

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

    return Callback(func, schedule, parameters, callsite)
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

struct GenericName end

function unique_callback_name(name, existing_names)
    if name ∈ existing_names
        return Symbol(:another_, name)
    else
        return name
    end
end

function unique_callback_name(::GenericName, existing_names)
    prefix = :callback # yeah, that's generic

    # Find a unique one
    n = 1
    while Symbol(prefix, n) ∈ existing_names
        n += 1
    end

    return Symbol(prefix, n)
end

"""
    add_callback!(simulation, callback::Callback; name = GenericName(), callback_kw...)

    add_callback!(simulation, func, schedule=IterationInterval(1); name = GenericName(), callback_kw...)

Add `Callback(func, schedule)` to `simulation.callbacks` under `name`. The default
`GenericName()` generates a name of the form `:callbackN`, where `N`
is big enough for the name to be unique.

If `name` is supplied, it may be modified if `simulation.callbacks[name]`
already exists.

`callback_kw` are passed to the constructor for [`Callback`](@ref).

The `callback` (which contains a schedule) can also be supplied directly.
"""
function add_callback!(simulation, callback::Callback; name = GenericName())
    name = unique_callback_name(name, keys(simulation.callbacks))
    simulation.callbacks[name] = callback
    return nothing
end

function add_callback!(simulation, func, schedule = IterationInterval(1);
                       name = GenericName(), callback_kw...)

    callback = Callback(func, schedule; callback_kw...)
    return add_callback!(simulation, callback; name)
end

