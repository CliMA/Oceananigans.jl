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

get_Δt(Δt) = Δt
get_Δt(wizard::TimeStepWizard) = wizard.Δt
get_Δt(simulation::Simulation) = get_Δt(simulation.Δt)

"""
    run!(simulation)

Run a `simulation` until one of the stop criteria evaluates to true. The simulation
will then stop.
"""
function run!(sim)
    model = sim.model
    clock = model.clock

    [open(out) for out in values(sim.output_writers)]

    while !stop(sim)
        time_before = time()

        if clock.iteration == 0
            [run_diagnostic(sim.model, diag) for diag in values(sim.diagnostics)]
            [write_output(sim.model, out)    for out  in values(sim.output_writers)]
        end

        for n in 1:sim.iteration_interval
            euler = clock.iteration == 0 || (sim.Δt isa TimeStepWizard && n == 1)
            time_step!(model, get_Δt(sim.Δt), euler=euler)

            [time_to_run(clock, diag) && run_diagnostic(sim.model, diag) for diag in values(sim.diagnostics)]
            [time_to_run(clock, out)  && write_output(sim.model, out)    for out  in values(sim.output_writers)]
        end

        sim.progress isa Function && sim.progress(sim)
        sim.Δt isa TimeStepWizard && update_Δt!(sim.Δt, model)

        time_after = time()
        sim.run_time += time_after - time_before
    end

    [close(out) for out in values(sim.output_writers)]

    return nothing
end
