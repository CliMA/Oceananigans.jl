import ..TimeSteppers: first_time_step!

""" Step `sim`ulation forward by one time step. """
initialize!(sim::ReactantSimulation) = update_state!(sim.model)
time_step!(sim::ReactantSimulation) = time_step!(sim.model, Δt; euler)
run!(sim::ReactantSimulation) = error("run! is not supported with ReactantState architecture.")

function first_time_step!(sim::ReactantSimulation)
    initialize!(sim)
    first_time_step!(sim.model, Δt)
    return nothing
end

