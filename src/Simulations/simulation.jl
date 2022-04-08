import Oceananigans.Utils: prettytime
import Oceananigans.TimeSteppers: reset!

# It's not a model --- its a simulation!

default_progress(simulation) = nothing

mutable struct Simulation{ML, TS, DT, ST, DI, OW, CB}
              model :: ML
        timestepper :: TS
                 Δt :: DT
     stop_iteration :: Float64
          stop_time :: ST
    wall_time_limit :: Float64
        diagnostics :: DI
     output_writers :: OW
          callbacks :: CB
      run_wall_time :: Float64
            running :: Bool
        initialized :: Bool
end

"""
    Simulation(model; Δt,
               stop_iteration = Inf,
               stop_time = Inf,
               wall_time_limit = Inf)

Construct a `Simulation` for a `model` with time step `Δt`.

Keyword arguments
=================

- `Δt`: Required keyword argument specifying the simulation time step. Can be a `Number`
        for constant time steps or a `TimeStepWizard` for adaptive time-stepping.

- `stop_iteration`: Stop the simulation after this many iterations.

- `stop_time`: Stop the simulation once this much model clock time has passed.

- `wall_time_limit`: Stop the simulation if it's been running for longer than this many
                     seconds of wall clock time.
"""
function Simulation(model; Δt,
                    stop_iteration = Inf,
                    stop_time = Inf,
                    wall_time_limit = Inf)

   if stop_iteration == Inf && stop_time == Inf && wall_time_limit == Inf
       @warn "This simulation will run forever as stop iteration = stop time " *
             "= wall time limit = Inf."
   end

   diagnostics = OrderedDict{Symbol, AbstractDiagnostic}()
   output_writers = OrderedDict{Symbol, AbstractOutputWriter}()
   callbacks = OrderedDict{Symbol, Callback}()

   callbacks[:stop_time_exceeded] = Callback(stop_time_exceeded)
   callbacks[:stop_iteration_exceeded] = Callback(stop_iteration_exceeded)
   callbacks[:wall_time_limit_exceeded] = Callback(wall_time_limit_exceeded)

   # Check for NaNs in the model's first prognostic field every 100 iterations.
   model_fields = fields(model)
   field_to_check_nans = NamedTuple{keys(model_fields) |> first |> tuple}(first(model_fields) |> tuple)
   nan_checker = NaNChecker(field_to_check_nans)
   callbacks[:nan_checker] = Callback(nan_checker, IterationInterval(100))

   # Convert numbers to floating point; otherwise preserve type (eg for DateTime types)
   FT = eltype(model.grid)
   Δt = Δt isa Number ? FT(Δt) : Δt
   stop_time = stop_time isa Number ? FT(stop_time) : stop_time

   return Simulation(model,
                     model.timestepper,
                     Δt,
                     Float64(stop_iteration),
                     stop_time,
                     Float64(wall_time_limit),
                     diagnostics,
                     output_writers,
                     callbacks,
                     0.0,
                     false,
                     false)
end

function Base.show(io::IO, s::Simulation)
    modelstr = summary(s.model)
    return print(io, "Simulation of ", modelstr, '\n',
                     "├── Next time step: $(prettytime(s.Δt))", '\n',
                     "├── Elapsed wall time: $(prettytime(s.run_wall_time))", '\n',
                     "├── Wall time per iteration: $(prettytime(s.run_wall_time / iteration(s)))", '\n',
                     "├── Stop time: $(prettytime(s.stop_time))", '\n',
                     "├── Stop iteration : $(s.stop_iteration)", '\n',
                     "├── Wall time limit: $(s.wall_time_limit)", '\n',
                     "├── Callbacks: $(ordered_dict_show(s.callbacks, "│"))", '\n',
                     "├── Output writers: $(ordered_dict_show(s.output_writers, "│"))", '\n',
                     "└── Diagnostics: $(ordered_dict_show(s.diagnostics, "│"))")
end

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

"""
    prettytime(sim::Simulation)

Return `sim.model.clock.time` as a prettily formatted string."
"""
prettytime(sim::Simulation, longform=true) = prettytime(time(sim))

"""
    run_wall_time(sim::Simulation)

Return `sim.run_wall_time` as a prettily formatted string."
"""
run_wall_time(sim::Simulation) = prettytime(sim.run_wall_time)

"""
    reset!(sim)

Reset `sim`ulation, `model.clock`, and `model.timestepper` to their initial state.
"""
function reset!(sim::Simulation)
    sim.model.clock.time = 0.0
    sim.model.clock.iteration = 0
    sim.model.clock.stage = 1
    sim.run_wall_time = 0.0
    sim.initialized = false
    sim.running = true
    reset!(sim.model.timestepper)
    return nothing
end

#####
##### Default stop criteria callback functions
#####

function stop_iteration_exceeded(sim)
    if sim.model.clock.iteration >= sim.stop_iteration
        @info "Simulation is stopping. Model iteration $(sim.model.clock.iteration) " *
               "has hit or exceeded simulation stop iteration $(Int(sim.stop_iteration))."
       sim.running = false 
    end
    return nothing
end

function stop_time_exceeded(sim)
    if sim.model.clock.time >= sim.stop_time
       @info "Simulation is stopping. Model time $(prettytime(sim.model.clock.time)) " *
             "has hit or exceeded simulation stop time $(prettytime(sim.stop_time))."
       sim.running = false 
    end
    return nothing
end

function wall_time_limit_exceeded(sim)
    if sim.run_wall_time >= sim.wall_time_limit
        @info "Simulation is stopping. Simulation run time $(run_wall_time(sim)) " *
              "has hit or exceeded simulation wall time limit $(prettytime(sim.wall_time_limit))."
       sim.running = false 
    end
    return nothing
end

