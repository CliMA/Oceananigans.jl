using Glob

using Oceananigans.Utils: initialize_schedule!
using Oceananigans.Fields: set!
using Oceananigans.OutputWriters: WindowedTimeAverage, checkpoint_superprefix
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper, update_state!, next_time, unit_time

using Oceananigans: AbstractModel, run_diagnostic!, write_output!

import Oceananigans.OutputWriters: checkpoint_path, set!
import Oceananigans.TimeSteppers: time_step!
import Oceananigans.Utils: aligned_time_step

# Simulations are for running

function stop(sim)
    time_before = time()

    for sc in sim.stop_criteria
        if sc(sim)
            time_after = time()
            sim.run_wall_time += time_after - time_before
            return true
        end
    end

    time_after = time()
    sim.run_wall_time += time_after - time_before

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
    if sim.run_wall_time >= sim.wall_time_limit
          @info "Simulation is stopping. Simulation run time $(prettytime(sim.run_wall_time)) " *
                "has hit or exceeded simulation wall time limit $(prettytime(sim.wall_time_limit))."
          return true
    end
    return false
end

add_dependency!(diagnostics, output) = nothing # fallback
add_dependency!(diags, wta::WindowedTimeAverage) = wta ∈ values(diags) || push!(diags, wta)

add_dependencies!(diags, writer) = [add_dependency!(diags, out) for out in values(writer.outputs)]
add_dependencies!(sim, ::Checkpointer) = nothing # Checkpointer does not have "outputs"

we_want_to_pickup(pickup::Bool) = pickup
we_want_to_pickup(pickup) = true

"""
    aligned_time_step(sim, Δt)

Return a time step 'aligned' with `sim.stop_time`, output writer schedules,
and callback schedules. Alignment with `sim.stop_time` takes precedence.
"""
function aligned_time_step(sim::Simulation, Δt)
    clock = sim.model.clock

    # Align time step with output writing and callback execution
    for obj in Iterators.flatten(zip(values(sim.output_writers), values(sim.callbacks)))
        aligned_Δt = aligned_time_step(obj.schedule, clock, Δt)
    end

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

    initialize_run!(sim, pickup)

    time_before = time()

    while !stop(sim)
        time_step!(sim)
    end

    time_after = time()
    sim.run_wall_time += time_after - time_before

    return nothing
end

""" Step `sim`ulation forward by one time step. """
function time_step!(sim::Simulation)
    model = sim.model
    clock = model.clock

    # Evaluate all diagnostics, and then write all output at first iteration
    if clock.iteration == 0
        [run_diagnostic!(diag, model) for diag in values(sim.diagnostics)]
        [callback(sim)                for callback in values(sim.callbacks)]
        [write_output!(writer, model) for writer in values(sim.output_writers)]
    end

    Δt = aligned_time_step(sim, sim.Δt)
    time_step!(model, Δt)

    # Run diagnostics, execute callbacks, then write output
    [diag.schedule(model)     && run_diagnostic!(diag, model) for diag in values(sim.diagnostics)]
    [callback.schedule(model) && callback(sim)                for callback in values(sim.callbacks)]
    [writer.schedule(model)   && write_output!(writer, model) for writer in values(sim.output_writers)]

    return nothing
end

""" Initialization: pickup, update_state, initialize schedules. """
function initialize_run!(sim, pickup=false)
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

    for obj in Iterators.flatten(zip(values(sim.output_writers),
                                     values(sim.diagnostics),
                                     values(sim.callbacks)))

        initialize_schedule!(obj.schedule)
    end
end