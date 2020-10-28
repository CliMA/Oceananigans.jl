using Glob

using Oceananigans.Utils: initialize_schedule!
using Oceananigans.Fields: set!
using Oceananigans.OutputWriters: WindowedTimeAverage, checkpoint_superprefix
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper, update_state!

import Oceananigans.OutputWriters: checkpoint_path

# Simulations are for running

function stop(sim)
    time_before = time()

    for sc in sim.stop_criteria
        if sc(sim)
            time_after = time()
            sim.run_time += time_after - time_before
            return true
        end
    end

    time_after = time()
    sim.run_time += time_after - time_before

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
    if sim.run_time >= sim.wall_time_limit
          @info "Simulation is stopping. Simulation run time $(prettytime(sim.run_time)) " *
                "has hit or exceeded simulation wall time limit $(prettytime(sim.wall_time_limit))."
          return true
    end
    return false
end

add_dependency!(diagnostics, output) = nothing # fallback
add_dependency!(diags, wta::WindowedTimeAverage) = wta ∈ values(diags) || push!(diags, wta)

add_dependencies!(diags, writer) = [add_dependency!(diags, out) for out in values(writer.outputs)]
add_dependencies!(sim, ::Checkpointer) = nothing # Checkpointer does not have "outputs"

get_Δt(Δt) = Δt
get_Δt(wizard::TimeStepWizard) = wizard.Δt
get_Δt(simulation::Simulation) = get_Δt(simulation.Δt)

ab2_or_rk3_time_step!(model::IncompressibleModel{<:QuasiAdamsBashforth2TimeStepper}, Δt; euler) = time_step!(model, Δt, euler=euler)
ab2_or_rk3_time_step!(model::IncompressibleModel{<:RungeKutta3TimeStepper}, Δt; euler) = time_step!(model, Δt)

we_want_to_pickup(pickup::Bool) = pickup
we_want_to_pickup(pickup) = true

"""
    run!(simulation; pickup=false)

Run a `simulation` until one of `simulation.stop_criteria` evaluates `true`.
The simulation will then stop.

# Picking simulations up from a checkpoint

Simulations will be "picked up" from a checkpoint if `pickup` is either `true`, a `String`,
or an `Integer` greater than 0.

Picking up a simulation sets field and tendency data to the specified checkpoint,
leaving all other model properties unchanged.

Possible values for `pickup` are:

    * `pickup=true` will pick a simulation up from the latest checkpoint associated with
      the `Checkpointer` in simulation.output_writers`. 

    * `pickup=iteration::Int` will pick a simulation up from the checkpointed file associated
       with `iteration` and the `Checkpointer` in simulation.output_writers`. 

    * `pickup=filepath::String` will pick a simulation up from checkpointer data in `filepath`.

Note that `pickup=true` and `pickup=iteration` will fail if `simulation.output_writers` contains
more than one checkpointer.
"""
function run!(sim; pickup=false)

    model = sim.model
    clock = model.clock

    if we_want_to_pickup(pickup)
        checkpointers = filter(writer -> writer isa Checkpointer, collect(values(sim.output_writers)))
        set!(model, checkpoint_path(pickup, checkpointers))
    end

    # Conservatively initialize the model state
    update_state!(model)

    # Output and diagnostics initialization
    for writer in values(sim.output_writers)
        open(writer)
        initialize_schedule!(writer.schedule)
        add_dependencies!(sim.diagnostics, writer) 
    end

    [initialize_schedule!(diag.schedule) for diag in values(sim.diagnostics)]

    while !stop(sim)
        time_before = time()

        # Evaluate all diagnostics, and then write all output at first iteration
        if clock.iteration == 0
            [run_diagnostic!(diag, sim.model) for diag in values(sim.diagnostics)]
            [write_output!(writer, sim.model) for writer in values(sim.output_writers)]
        end

        for n in 1:sim.iteration_interval
            euler = clock.iteration == 0 || (sim.Δt isa TimeStepWizard && n == 1)
            ab2_or_rk3_time_step!(model, get_Δt(sim.Δt), euler=euler)

            # Run diagnostics, then write output
            [  diag.schedule(model) && run_diagnostic!(diag, sim.model) for diag in values(sim.diagnostics)]
            [writer.schedule(model) && write_output!(writer, sim.model) for writer in values(sim.output_writers)]
        end

        sim.progress(sim)

        sim.Δt isa TimeStepWizard && update_Δt!(sim.Δt, model)

        time_after = time()
        sim.run_time += time_after - time_before
    end

    return nothing
end

#####
##### Util for "picking up" a simulation from a checkpoint
#####

""" Returns `filepath`. Shortcut for `run!(simulation, pickup=filepath)`. """
checkpoint_path(filepath::AbstractString, checkpointers) = filepath

function checkpoint_path(pickup, checkpointers)
    length(checkpointers) == 0 && error("No checkpointers found: cannot pickup simulation!")
    length(checkpointers) > 1 && error("Multiple checkpointers found: not sure which one to pickup simulation from!")
    return checkpoint_path(pickup, first(checkpointers))
end

"""
    checkpoint_path(pickup::Bool, checkpointer)

For `pickup=true`, parse the filenames in `checkpointer.dir` associated with
`checkpointer.prefix` and return the path to the file whose name contains
the largest iteration.
"""
function checkpoint_path(pickup::Bool, checkpointer::Checkpointer)
    filepaths = glob(checkpoint_superprefix(checkpointer.prefix) * "*.jld2", checkpointer.dir)
    filenames = basename.(filepaths)

    # Parse filenames to find latest checkpointed iteration
    leading = length(checkpoint_superprefix(checkpointer.prefix))
    trailing = 5 # length(".jld2")
    iterations = map(name -> parse(Int, name[leading+1:end-trailing]), filenames)

    latest_iteration, idx = findmax(iterations)

    return filepaths[idx]
end
