struct Callback{F, S}
    func :: F
    schedule :: S
end

(callback::Callback)(sim) = callback.func(sim)

"""
    Callback(func, schedule)

Return `Callback` that executes `func(sim::Simulation)` on `schedule`.

`schedule = IterationInterval(1)` by default.
"""
Callback(func) = Callback(func, IterationInterval(1))
