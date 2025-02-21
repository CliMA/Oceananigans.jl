using Oceananigans.Simulations: ModelCallsite
using Oceananigans: TimeStepCallsite, TendencyCallsite, UpdateStateCallsite
import Oceananigans.TimeSteppers: time_step!

aligned_time_step(::ReactantSimulation, Δt) = Δt

function initialize!(sim::ReactantSimulation)
    #=
    if sim.verbose
        @info "Initializing simulation..."
        start_time = time_ns()
    end
    =#

    model = sim.model
    clock = model.clock

    update_state!(model)

    #=
    # Output and diagnostics initialization
    [add_dependencies!(sim.diagnostics, writer) for writer in values(sim.output_writers)]

    # Initialize schedules
    scheduled_activities = Iterators.flatten((values(sim.diagnostics),
                                              values(sim.callbacks),
                                              values(sim.output_writers)))

    for activity in scheduled_activities
        initialize!(activity.schedule, sim.model)
    end
    =#

    #=
    # Reset! the model time-stepper, evaluate all diagnostics, and write all output at first iteration
    @trace if clock.iteration == 0
        reset!(timestepper(sim.model))

        # Initialize schedules and run diagnostics, callbacks, and output writers
        for diag in values(sim.diagnostics)
            run_diagnostic!(diag, model)
        end

        for callback in values(sim.callbacks)
            callback.callsite isa TimeStepCallsite && callback(sim)
        end

        for writer in values(sim.output_writers)
            writer.schedule(sim.model)
            write_output!(writer, model)
        end
    end
    =#

    sim.initialized = true

    #=
    if sim.verbose
        initialization_time = prettytime(1e-9 * (time_ns() - start_time))
        @info "    ... simulation initialization complete ($initialization_time)"
    end
    =#

    return nothing
end

""" Step `sim`ulation forward by one time step. """
function time_step!(sim::ReactantSimulation)

    start_time_step = time_ns()
    model_callbacks = Tuple(cb for cb in values(sim.callbacks) if cb.callsite isa ModelCallsite)
    Δt = aligned_time_step(sim, sim.Δt)

    if !(sim.initialized) # execute initialization step
        initialize!(sim)
        initialize!(sim.model)

        if sim.running # check that initialization didn't stop time-stepping
            if sim.verbose
                @info "Executing initial time step..."
                start_time = time_ns()
            end

            # Take first time-step
            time_step!(sim.model, Δt, callbacks=model_callbacks)

            if sim.verbose
                elapsed_initial_step_time = prettytime(1e-9 * (time_ns() - start_time))
                @info "    ... initial time step complete ($elapsed_initial_step_time)."
            end
        else
            @warn "Simulation stopped during initialization."
        end

    else # business as usual...
        if Δt < sim.minimum_relative_step * sim.Δt
            next_time = sim.model.clock.time + Δt
            @warn "Resetting clock to $next_time and skipping time step of size Δt = $Δt"
            sim.model.clock.time = next_time
        else
            time_step!(sim.model, Δt, callbacks=model_callbacks)
        end
    end

    for callback in values(sim.callbacks)
        need_to_call = callback.schedule(sim.model)
        @trace if need_to_call
            callback(sim)
        end

        #=
        @trace if callback.callsite isa TimeStepCallsite
            if callback.schedule(sim.model)
                callback(sim)
            else
                nothing
            end
        else
            nothing
        end
        =#
    end

    #=
    # Callbacks and callback-like things
    for diag in values(sim.diagnostics)
        diag.schedule(sim.model) && run_diagnostic!(diag, sim.model)
    end


    for writer in values(sim.output_writers)
        writer.schedule(sim.model) && write_output!(writer, sim.model)
    end

    end_time_step = time_ns()

    # Increment the wall clock
    sim.run_wall_time += 1e-9 * (end_time_step - start_time_step)
    =#

    return nothing
end

