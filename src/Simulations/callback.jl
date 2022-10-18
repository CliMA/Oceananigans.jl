using Oceananigans.Utils: prettysummary
using Oceananigans.OutputWriters: WindowedTimeAverage

struct Callback{P, F, S}
    func :: F
    schedule :: S
    parameters :: P
end

@inline (callback::Callback)(sim) = callback.func(sim, callback.parameters)
@inline (callback::Callback{<:Nothing})(sim) = callback.func(sim)

"""
    Callback(func, schedule=IterationInterval(1); parameters=nothing)

Return `Callback` that executes `func` on `schedule`
with optional `parameters`. `schedule = IterationInterval(1)` by default.

If `isnothing(parameters)`, `func(sim::Simulation)` is called.
Otherwise, `func` is called via `func(sim::Simulation, parameteres)`.
"""
Callback(func, schedule=IterationInterval(1); parameters=nothing) =
    Callback(func, schedule, parameters)

Base.summary(cb::Callback{Nothing}) = string("Callback of ", prettysummary(cb.func, false), " on ", summary(cb.schedule))
Base.summary(cb::Callback) = string("Callback of ", prettysummary(cb.func, false), " on ", summary(cb.schedule),
                                    " with parameters ", cb.parameters)

Base.show(io::IO, cb::Callback) = print(io, summary(cb))

function Callback(wta::WindowedTimeAverage)
    function func(sim)
        model = simulation.model
        advance_time_average!(wta, model)
        return nothing
    end
    return Callback(func, wta.schedule, nothing)
end

Callback(wta::WindowedTimeAverage, schedule; kw...) =
    throw(ArgumentError("Schedule must be inferred from WindowedTimeAverage. 
                        Use Callback(windowed_time_average)"))

