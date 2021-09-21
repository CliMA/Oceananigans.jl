using Glob

using Oceananigans.Fields: set!
using Oceananigans.OutputWriters: WindowedTimeAverage, checkpoint_superprefix
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper, update_state!, next_time, unit_time

using Oceananigans: AbstractModel, run_diagnostic!, write_output!

import Oceananigans.OutputWriters: checkpoint_path, set!
import Oceananigans.TimeSteppers: time_step!
import Oceananigans.Utils: aligned_time_step

# Simulations are for running

"""
    @stopwatch sim expr

Increment sim.stopwatch with the execution time of expr.
"""
macro stopwatch(sim, expr)
    return esc(quote
       local time_before = time_ns() * 1e-9
       local output = $expr
       local time_after = time_ns() * 1e-9
       sim.wall_time += time_after - time_before
       output
   end)
end

#####
##### But they must be stopped
#####

function stop(sim)
    for criteria in sim.stop_criteria
        if criteria(sim)
            return true
        end
    end

    return false
end

function iteration_limit_exceeded(sim)
    if sim.model.clock.iteration >= sim.stop_iteration
          @info "Simulation is stopping. Model iteration $(sim.model.clock.iteration) " *
                "has hit or exceeded simulation stop iteration $(sim.stop_iteration)."
          return true
    end
    return false
end

function stop_time_exceeded(sim)
    if sim.model.clock.time >= sim.stop_time
          @info "Simulation is stopping. Model time $(prettytime(sim.model.clock.time)) " *
                "has hit or exceeded simulation stop time $(prettytime(sim.stop_time))."
          return true
    end
    return false
end

function wall_time_limit_exceeded(sim)
    if sim.wall_time >= sim.wall_time_limit
          @info "Simulation is stopping. Simulation run time $(prettytime(sim.wall_time)) " *
                "has hit or exceeded simulation wall time limit $(prettytime(sim.wall_time_limit))."
          return true
    end
    return false
end

#####
##### Time-step "alignment" with output and callbacks scheduled on TimeInterval
#####

appointments(sim) = Iterators.flatten(zip(values(sim.output_writers), values(sim.callbacks)))

function schedule_aligned_Δt(sim, aligned_Δt)
    clock = sim.model.clock

    for app in appointments(sim)
        @show app app.schedule
        app.schedule isa TimeInterval && @show app.schedule.previous_actuation_time
        aligned_Δt = aligned_time_step(app.schedule, clock, aligned_Δt)
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

    @stopwatch sim initialize_simulation!(sim, pickup)

    sim.running = !(stop(sim))

    while sim.running
        time_step!(sim)
        sim.running = !(stop(sim))
    end

    return nothing
end

""" Step `sim`ulation forward by one time step. """
function time_step!(sim::Simulation)

    initialization_step = !(sim.initialized)

    if initialization_step 
        @stopwatch(sim, initialize_simulation!(sim))
        start_time = time_ns()
        @info "Executing first time step..."
    end

    @stopwatch sim begin
        Δt = aligned_time_step(sim, sim.Δt)
        time_step!(sim.model, Δt)
    end

    if initialization_step
        elapsed_first_step_time = prettytime(1e-9 * (time_ns() - start_time))
        @info "    ... first time step complete ($elapsed_first_step_time)."
    end

    @stopwatch sim begin
        evaluate_diagnostics!(sim)           
        evaluate_callbacks!(sim)             
        evaluate_output_writers!(sim)        
    end

    return nothing
end

evaluate_diagnostics!(sim)    = [diag.schedule(sim.model)     && run_diagnostic!(diag, sim.model) for diag     in values(sim.diagnostics)]
evaluate_callbacks!(sim)      = [callback.schedule(sim.model) && callback(sim)                    for callback in values(sim.callbacks)]
evaluate_output_writers!(sim) = [writer.schedule(sim.model)   && write_output!(writer, sim.model) for writer   in values(sim.output_writers)]

#####
##### Simulation initialization
#####

add_dependency!(diagnostics, output) = nothing # fallback
add_dependency!(diags, wta::WindowedTimeAverage) = wta ∈ values(diags) || push!(diags, wta)

add_dependencies!(diags, writer) = [add_dependency!(diags, out) for out in values(writer.outputs)]
add_dependencies!(sim, ::Checkpointer) = nothing # Checkpointer does not have "outputs"

we_want_to_pickup(pickup::Bool) = pickup
we_want_to_pickup(pickup) = true

""" 
    initialize_simulation!(sim, pickup=false)

Initialize a simulation before running it. Initialization involves:

- Updating the auxiliary state of the simulation (filling halo regions, computing auxiliary fields)
- Evaluating all diagnostics, callbacks, and output writers if sim.model.clock.iteration == 0
- Adding diagnostics that "depend" on output writers
- If pickup != false, picking up the simulation from a checkpoint.

"""
function initialize_simulation!(sim, pickup=false)
    @info "Updating model auxiliary state during simulation initialization..."
    start_time = time_ns()

    model = sim.model
    clock = model.clock

    if we_want_to_pickup(pickup)
        checkpointers = filter(writer -> writer isa Checkpointer, collect(values(sim.output_writers)))
        checkpoint_filepath = checkpoint_path(pickup, checkpointers)

        # https://github.com/CliMA/Oceananigans.jl/issues/1159
        if pickup isa Bool && isnothing(checkpoint_filepath)
            @warn "pickup=true but no checkpoints were found. Simulation will run without picking up."
        else
            set!(model, checkpoint_filepath)
        end
    end

    # Conservatively initialize the model state
    update_state!(model)

    # Output and diagnostics initialization
    [add_dependencies!(sim.diagnostics, writer) for writer in values(sim.output_writers)]

    # Evaluate all diagnostics, and then write all output at first iteration
    if clock.iteration == 0
        [run_diagnostic!(diag, model) for diag in values(sim.diagnostics)]
        [callback(sim)                for callback in values(sim.callbacks)]
        [write_output!(writer, model) for writer in values(sim.output_writers)]
    end

    sim.initialized = true

    initialization_time = prettytime(1e-9 * (time_ns() - start_time))
    @info "    ... simulation initialization complete ($initialization_time)"

    return nothing
end

