import ..TimeSteppers: first_time_step!

""" Step `sim`ulation forward by one time step. """
function time_step!(sim::ReactantSimulation) 
    n = iteration(sim) + 1
    if n == 1 
        first_time_step!(sim) # This automatically performs an Euler step if needed
    else
        time_step!(sim.model, sim.Δt)
    end
    return nothing
end

run!(sim::ReactantSimulation) = error("run! is not supported with ReactantState architecture.")

function first_time_step!(sim::ReactantSimulation)
    initialize!(sim)
    first_time_step!(sim.model, sim.Δt)
    return nothing
end

function time_step_for!(sim::ReactantSimulation, Nsteps)
    @trace for _ = 1:Nsteps
        time_step!(sim)
    end
    return nothing
end

