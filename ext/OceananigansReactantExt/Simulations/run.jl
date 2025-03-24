import Oceananigans: initialize!
import Oceananigans.TimeSteppers: first_time_step!

""" Step `sim`ulation forward by one time step. """
time_step!(sim::ReactantSimulation) = time_step!(sim.model, sim.Δt)

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

function initialize!(sim::ReactantSimulation)
    model = sim.model
    initialize!(model)
    update_state!(model)
    sim.initialized = true
    return nothing
end

