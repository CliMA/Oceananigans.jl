const ReactantSimulation = Simulation{<:ReactantModel}

function Simulation(model::ReactantModel; Δt,
                    verbose = true,
                    stop_iteration = Inf,
                    stop_time = nothing,
                    wall_time_limit = Inf,
                    align_time_step = false,
                    minimum_relative_step = 0)

   @assert isnothing(stop_time) "`stop_time` is not supported for ReactantModel"

   diagnostics = nothing
   output_writers = nothing
   callbacks = nothing

   Δt = Float64(Δt)

   return Simulation(model,
                     Δt,
                     Float64(stop_iteration),
                     nothing, # disallow stop_time
                     Float64(wall_time_limit),
                     diagnostics,
                     output_writers,
                     callbacks,
                     0.0,
                     align_time_step,
                     false,
                     false,
                     verbose,
                     Float64(minimum_relative_step))
end

iteration(sim::ReactantSimulation) = Reactant.to_number(iteration(sim.model))
time(sim::ReactantSimulation) = Reactant.to_number(time(sim.model))

add_callback!(::ReactantSimulation, args...) =
    error("Cannot add callbacks to a Simulation with ReactantState architecture!")
