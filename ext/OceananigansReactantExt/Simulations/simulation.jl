const ReactantSimulation = Simulation{<:ReactantModel}

function Simulation(model::ReactantModel; Δt,
                    verbose = true,
                    stop_iteration = Inf)

   diagnostics = nothing
   output_writers = nothing
   callbacks = nothing

   Δt = Float64(Δt)

   return Simulation(model,
                     Δt,
                     Float64(stop_iteration),
                     nothing, # disallow stop_time
                     Inf,
                     diagnostics,
                     output_writers,
                     callbacks,
                     0.0,
                     false,
                     false,
                     false,
                     verbose,
                     0.0)
end

iteration(sim::ReactantSimulation) = Reactant.to_number(iteration(sim.model))
time(sim::ReactantSimulation) = Reactant.to_number(time(sim.model))

add_callback!(::ReactantSimulation, args...) =
    error("Cannot add callbacks to a Simulation with ReactantState architecture!")

