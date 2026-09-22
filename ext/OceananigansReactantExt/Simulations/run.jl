#####
##### Running a ReactantSimulation: the compiled counterpart of src/Simulations/run.jl
#####
##### Same division of labour as the eager code, so that `@compile run!(sim)` is one program:
#####
#####   initialize!(sim)          iteration 0: initialize the model, refresh its state, fire callbacks
#####   time_step!(sim)           one step: the model step with its callsite callbacks, then the
#####                             TimeStepCallsite callbacks whose schedule says so
#####   time_step_for!(sim, N)    N steps in one traced loop
#####   run!(sim)                 initialize!, then stop_iteration steps
#####
##### Not yet available inside a program: time-step alignment, writers, diagnostics, stop
##### criteria on the state, wall time.
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

"""
    initialize!(sim::ReactantSimulation)

Iteration 0, as the eager `initialize!(sim)` does it: `initialize!(model)`, `update_state!` with
the model-callsite callbacks, then every `TimeStepCallsite` callback once, unconditionally.
"""
function initialize!(sim::ReactantSimulation)
    model = sim.model
    initialize!(model)
    update_state!(model, callbacks_at(sim, ModelCallsite))
    foreach(callback -> callback(sim), callbacks_at(sim, TimeStepCallsite))
    sim.initialized = true
    return nothing
end

"""
    time_step!(sim::ReactantSimulation)

One step of `sim.Δt`: `time_step!(model, Δt; callbacks)` with the `TendencyCallsite` and
`UpdateStateCallsite` callbacks, then each `TimeStepCallsite` callback whose schedule fires.

Inside a program the schedule evaluates to a traced `Bool` on the traced clock, so the firing is
a traced `if` rather than the eager `schedule(model) && callback(sim)`. A callback that fires
here must be traceable: its schedule a pure function of the clock (`IterationInterval` is;
`TimeInterval` mutates a host counter and is not), and its function only device work (kernel
launches, broadcasts, reductions into arrays reachable from `sim`). Printing, IO, and
data-dependent errors belong between programs.
"""
function time_step!(sim::ReactantSimulation)
    model = sim.model
    time_step!(model, sim.Δt; callbacks = callbacks_at(sim, ModelCallsite))

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
"""
function time_step_for!(sim::ReactantSimulation, Nsteps)
    @trace track_numbers = false for _ = 1:Nsteps
        time_step!(sim)
    end
    return nothing
end

"""
    run!(sim::ReactantSimulation)

`initialize!(sim)`, then `stop_iteration` steps through [`time_step_for!`](@ref).

`stop_iteration` is a plain number on the `Simulation`, so under `@compile` it is a static loop
bound. It counts steps from a fresh simulation: a run continued from iteration `n` steps
`stop_iteration` more times.
"""
function run!(sim::ReactantSimulation)
    isfinite(sim.stop_iteration) || throw(ArgumentError(
        "run! on a ReactantState model needs `stop_iteration`: it is the bound of the compiled loop."))

    initialize!(sim)
    time_step_for!(sim, Int(sim.stop_iteration))

    return nothing
end
