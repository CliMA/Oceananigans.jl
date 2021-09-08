# It's not a model --- its a simulation!

default_progress(simulation) = nothing

mutable struct Simulation{ML, DT, SC, SI, ST, WT, DI, OW, CB}
              model :: ML
                 Δt :: DT
      stop_criteria :: SC
     stop_iteration :: SI
          stop_time :: ST
    wall_time_limit :: WT
      run_wall_time :: Float64
        diagnostics :: DI
     output_writers :: OW
          callbacks :: CB
end

"""
    Simulation(model; Δt,
               stop_criteria = Any[iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded],
               stop_iteration = Inf,
               stop_time = Inf,
               wall_time_limit = Inf,
               diagnostics = OrderedDict{Symbol, AbstractDiagnostic}(),
               output_writers = OrderedDict{Symbol, AbstractOutputWriter}(),
               callback = OrderedDict{Symbol, Callback}())

Construct a `Simulation` for a `model` with time step `Δt`.

Keyword arguments
=================

- `Δt`: Required keyword argument specifying the simulation time step. Can be a `Number`
        for constant time steps or a `TimeStepWizard` for adaptive time-stepping.

- `stop_criteria`: A list of functions or callable objects (each taking a single argument,
                   the `simulation`). If any of the functions return `true` when the stop criteria is
                   evaluated the simulation will stop.

- `stop_iteration`: Stop the simulation after this many iterations.

- `stop_time`: Stop the simulation once this much model clock time has passed.

- `wall_time_limit`: Stop the simulation if it's been running for longer than this many
                     seconds of wall clock time.
"""
function Simulation(model; Δt,
                    stop_criteria = Any[iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded],
                    stop_iteration = Inf,
                    stop_time = Inf,
                    wall_time_limit = Inf,
                    diagnostics = OrderedDict{Symbol, AbstractDiagnostic}(),
                    output_writers = OrderedDict{Symbol, AbstractOutputWriter}(),
                    callbacks = OrderedDict{Symbol, Callback}())

   if stop_iteration == Inf && stop_time == Inf && wall_time_limit == Inf
       @warn "This simulation will run forever as stop iteration = stop time " *
             "= wall time limit = Inf."
   end

   # Check for NaNs in the model's first prognostic field every 100 iterations.
   model_fields = fields(model)
   field_to_check_nans = NamedTuple{keys(model_fields) |> first |> tuple}(first(model_fields) |> tuple)
   diagnostics[:nan_checker] = NaNChecker(fields=field_to_check_nans,
                                          schedule=IterationInterval(100))

   run_wall_time = 0.0

   Δt = convert(eltype(model.grid), Δt)

   return Simulation(model, Δt, stop_criteria, stop_iteration, stop_time, wall_time_limit, run_wall_time,
                     diagnostics, output_writers, callbacks)
end

Base.show(io::IO, s::Simulation) =
    print(io, "Simulation{$(typeof(s.model).name){$(Base.typename(typeof(s.model.architecture))), $(eltype(s.model.grid))}}\n",
            "├── Model clock: time = $(prettytime(s.model.clock.time)), iteration = $(s.model.clock.iteration) \n",
            "├── Next time step: $(prettytime(s.Δt)) \n",
            "├── Stop criteria: $(s.stop_criteria)\n",
            "├── Run wall time: $(prettytime(s.run_wall_time)), wall time limit: $(s.wall_time_limit)\n",
            "├── Stop time: $(prettytime(s.stop_time)), stop iteration: $(s.stop_iteration)\n",
            "├── Diagnostics: $(ordered_dict_show(s.diagnostics, "│"))\n",
            "└── Output writers: $(ordered_dict_show(s.output_writers, "│"))")

#####
##### Utilities
#####

"""
    time(sim::Simulation)

Return the current simulation time.
"""
Base.time(sim::Simulation) = sim.model.clock.time

"""
    iteration(sim::Simulation)

Return the current simulation iteration.
"""
iteration(sim::Simulation) = sim.model.clock.iteration
