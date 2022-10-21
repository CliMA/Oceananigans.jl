using Oceananigans.Utils: prettysummary

struct Callback{P, F, S, CS}
    func :: F
    schedule :: S
    parameters :: P
    callsite :: CS
end

struct TimestepCallback end
struct TendencyCallback end
struct UpdateStateCallback end

@inline (callback::Callback)(sim) = callback.func(sim, callback.parameters)
@inline (callback::Callback{<:Nothing})(sim) = callback.func(sim)

"""
    Callback(func, schedule=IterationInterval(1); parameters=nothing)

Return `Callback` that executes `func` on `schedule`
with optional `parameters`. `schedule = IterationInterval(1)` by default.

If `isnothing(parameters)`, `func(sim::Simulation)` is called.
Otherwise, `func` is called via `func(sim::Simulation, parameteres)`.
"""
Callback(func, schedule=IterationInterval(1); parameters=nothing, callsite = TimestepCallback) =
    Callback(func, schedule, parameters, callsite())

Base.summary(cb::Callback{Nothing}) = string("Callback of ", prettysummary(cb.func, false), " on ", summary(cb.schedule))
Base.summary(cb::Callback) = string("Callback of ", prettysummary(cb.func, false), " on ", summary(cb.schedule),
                                    " with parameters ", cb.parameters)

Base.show(io::IO, cb::Callback) = print(io, summary(cb))

