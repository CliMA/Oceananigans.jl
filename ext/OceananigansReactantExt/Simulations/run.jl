import Oceananigans: initialize!
import Oceananigans.TimeSteppers: first_time_step!, time_step!

""" Step `sim`ulation forward by one time step. """
time_step!(sim::ReactantSimulation) = time_step!(sim.model, sim.Δt)

run!(sim::ReactantSimulation) = error("run! is not supported with ReactantState architecture.")

function time_step_for!(sim::ReactantSimulation, Nsteps)
    @trace for _ = 1:Nsteps
        time_step!(sim)
    end
    return nothing
end

