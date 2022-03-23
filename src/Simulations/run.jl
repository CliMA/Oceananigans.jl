using Glob

using Oceananigans.Fields: set!
using Oceananigans.OutputWriters: WindowedTimeAverage, checkpoint_superprefix
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper, update_state!, next_time, unit_time

using Oceananigans: AbstractModel, run_diagnostic!, write_output!

import Oceananigans.OutputWriters: checkpoint_path, set!
import Oceananigans.TimeSteppers: time_step!
import Oceananigans.Utils: aligned_time_step

# Simulations are for running

#####
##### Time-step "alignment" with output and callbacks scheduled on TimeInterval
#####

function collect_scheduled_activities(sim)
    writers = values(sim.output_writers)
    callbacks = values(sim.callbacks)
    return tuple(writers..., callbacks...)
end

function schedule_aligned_Δt(sim, aligned_Δt)
    clock = sim.model.clock
    activities = collect_scheduled_activities(sim)

    for activity in activities
        aligned_Δt = aligned_time_step(activity.schedule, clock, aligned_Δt)
    end

    return aligned_Δt
end

"""
    aligned_time_step(sim, Δt)

Return a time step 'aligned' with `sim.stop_time`, output writer schedules, 
and callback schedules. Alignment with `sim.stop_time` takes precedence.
"""
function aligned_time_step(sim::Simulation, Δt)
    clock = sim.model.clock

    aligned_Δt = Δt

    # Align time step with output writing and callback execution
    aligned_Δt = schedule_aligned_Δt(sim, aligned_Δt)
    
    # Align time step with simulation stop time
    aligned_Δt = min(aligned_Δt, unit_time(sim.stop_time - clock.time))

    # Temporary fix for https://github.com/CliMA/Oceananigans.jl/issues/1280
    aligned_Δt = aligned_Δt <= 0 ? Δt : aligned_Δt

    return aligned_Δt
end

"""
    run!(simulation; pickup=false)

Run a `simulation` until one of `simulation.stop_criteria` evaluates `true`.
The simulation will then stop.

# Picking simulations up from a checkpoint

Simulations are "picked up" from a checkpoint if `pickup` is either `true`, a `String`, or an
`Integer` greater than 0.

Picking up a simulation sets field and tendency data to the specified checkpoint,
leaving all other model properties unchanged.

Possible values for `pickup` are:

  * `pickup=true` picks a simulation up from the latest checkpoint associated with
    the `Checkpointer` in `simulation.output_writers`.

  * `pickup=iteration::Int` picks a simulation up from the checkpointed file associated
     with `iteration` and the `Checkpointer` in `simulation.output_writers`.

  * `pickup=filepath::String` picks a simulation up from checkpointer data in `filepath`.

Note that `pickup=true` and `pickup=iteration` fails if `simulation.output_writers` contains
more than one checkpointer.
"""
function run!(sim; pickup=false)

    if we_want_to_pickup(pickup)
        checkpoint_file_path = checkpoint_path(pickup, sim.output_writers)
        set!(sim.model, checkpoint_file_path)
    end

    sim.initialized = false
    sim.running = true

    while sim.running
        time_step!(sim)
    end

    return nothing
end

""" Step `sim`ulation forward by one time step. """
function time_step!(sim::Simulation)

    start_time_step = time_ns()

    if !(sim.initialized) # execute initialization step
        initialize_simulation!(sim)

        if sim.running # check that initialization didn't stop time-stepping
            @info "Executing initial time step..."

            start_time = time_ns()
            Δt = aligned_time_step(sim, sim.Δt)
            time_step!(sim.model, Δt)

            elapsed_initial_step_time = prettytime(1e-9 * (time_ns() - start_time))
            @info "    ... initial time step complete ($elapsed_initial_step_time)."
        else
            @warn "Simulation stopped during initialization."
        end
    else # business as usual...
        Δt = aligned_time_step(sim, sim.Δt)
        time_step!(sim.model, Δt)
    end

    # Callbacks and callback-like things
    [diag.schedule(sim.model)     && run_diagnostic!(diag, sim.model) for diag     in values(sim.diagnostics)]
    [callback.schedule(sim.model) && callback(sim)                    for callback in values(sim.callbacks)]
    [writer.schedule(sim.model)   && write_output!(writer, sim.model) for writer   in values(sim.output_writers)]

    end_time_step = time_ns()

    # Increment the wall clock
    sim.run_wall_time += 1e-9 * (end_time_step - start_time_step)

    return nothing
end

#####
##### Simulation initialization
#####

add_dependency!(diagnostics, output) = nothing # fallback
add_dependency!(diags, wta::WindowedTimeAverage) = wta ∈ values(diags) || push!(diags, wta)

add_dependencies!(diags, writer) = [add_dependency!(diags, out) for out in values(writer.outputs)]
add_dependencies!(sim, ::Checkpointer) = nothing # Checkpointer does not have "outputs"

we_want_to_pickup(pickup::Bool) = pickup
we_want_to_pickup(pickup::Integer) = true
we_want_to_pickup(pickup::String) = true
we_want_to_pickup(pickup) = throw(ArgumentError("Cannot run! with pickup=$pickup"))

""" 
    initialize_simulation!(sim, pickup=false)

Initialize a simulation:

- Update the auxiliary state of the simulation (filling halo regions, computing auxiliary fields)
- Evaluate all diagnostics, callbacks, and output writers if sim.model.clock.iteration == 0
- Add diagnostics that "depend" on output writers

"""
function initialize_simulation!(sim)
    @info "Initializing simulation..."
    start_time = time_ns()

    model = sim.model
    clock = model.clock

    update_state!(model)

    # Output and diagnostics initialization
    [add_dependencies!(sim.diagnostics, writer) for writer in values(sim.output_writers)]

    # Reset! the model time-stepper, evaluate all diagnostics, and write all output at first iteration
    if clock.iteration == 0
        reset!(sim.model.timestepper)

        # Initialize schedules and run diagnostics, callbacks, and output writers
        for diag in values(sim.diagnostics)
            diag.schedule(sim.model)
            run_diagnostic!(diag, model)
        end

        for callback in values(sim.callbacks)
            callback.schedule(model)
            callback(sim)
        end

        for writer in values(sim.output_writers)
            writer.schedule(sim.model)
            write_output!(writer, model)
        end
    end

    sim.initialized = true

    initialization_time = prettytime(1e-9 * (time_ns() - start_time))
    @info "    ... simulation initialization complete ($initialization_time)"

    return nothing
end

