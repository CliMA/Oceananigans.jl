const ReactantSimulation = Simulation{<:ReactantModel}

"""
    Simulation(model::ReactantModel; Δt, stop_iteration, verbose = true,
               wall_time_limit = Inf, align_time_step = false, minimum_relative_step = 0)

A `Simulation` of a model on `ReactantState`, meant to be compiled: `@compile run!(sim)` is one
program that steps the model `stop_iteration` times and fires `sim.callbacks` inside the loop
(see [`run!`](@ref)).

What differs from the eager `Simulation`:

- `Δt` is fixed. Adaptive stepping changes `Δt` between programs, not inside one.
- `stop_iteration` is the loop bound; `stop_time` is not supported (no time-step alignment
  exists inside a program, so an interval must be an integer number of steps).
- `callbacks` starts empty. The eager constructor installs the stop-criterion callbacks, which
  here are the loop bound, and a NaN checker, whose data-dependent error cannot exist inside a
  program. `add_callback!` works as usual; see [`time_step!`](@ref) for what a callback may do.
- `output_writers` and `diagnostics` are `nothing`: IO cannot happen inside a program. They are
  the next things to gain a compiled form (device-side staging, host-side writing).
"""
function Simulation(model::ReactantModel; Δt,
                    verbose = true,
                    stop_iteration = Inf,
                    stop_time = nothing,
                    wall_time_limit = Inf,
                    align_time_step = false,
                    minimum_relative_step = 0)

   @assert isnothing(stop_time) "`stop_time` is not supported for ReactantModel; use `stop_iteration`"

   diagnostics = nothing
   output_writers = nothing
   callbacks = OrderedDict{Symbol, Callback}()

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
