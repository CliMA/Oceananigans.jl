using Oceananigans: prognostic_fields, AbstractModel
using Oceananigans.Diagnostics: default_nan_checker
using Oceananigans.DistributedComputations: Distributed, all_reduce
using Oceananigans.OutputWriters: JLD2Writer, NetCDFWriter

import Oceananigans.Utils: prettytime
import Oceananigans.TimeSteppers: reset!
import Oceananigans.OutputWriters: write_output!
import Oceananigans.Solvers: iteration

default_progress(simulation) = nothing

mutable struct Simulation{ML, DT, ST, DI, OW, CB, FT, BL}
    model :: ML
    Δt :: DT
    stop_iteration :: FT
    stop_time :: ST
    wall_time_limit :: FT
    diagnostics :: DI
    output_writers :: OW
    callbacks :: CB
    run_wall_time :: FT
    align_time_step :: BL
    running :: BL
    initialized :: BL
    verbose :: BL
    minimum_relative_step :: FT
end

"""
    Simulation(model;
               Δt,
               verbose = true,
               stop_iteration = Inf,
               stop_time = Inf,
               wall_time_limit = Inf,
               minimum_relative_step = 0)

Construct a `Simulation` for a `model` with time step `Δt`.

Keyword arguments
=================

- `Δt`: Required keyword argument specifying the simulation time step. Can be a `Number`
        for constant time steps or a `TimeStepWizard` for adaptive time-stepping.

- `stop_iteration`: Stop the simulation after this many iterations. Default: `Inf`.

- `stop_time`: Stop the simulation once this much model clock time has passed. Default: `Inf`.

- `wall_time_limit`: Stop the simulation if it's been running for longer than this many
                     seconds of wall clock time. Default: `Inf`.

- `align_time_step`: When `true` it implies that the simulation will automatically adjust the
                     time-step to meet a constraint imposed by various schedules like `ScheduledTimes`,
                     `TimeInterval`, `AveragedTimeInterval`, as well as a `stop_time` criterion.
                     If `false`, i.e., no time-step alignment, then the simulation might blithely step passed
                     the specified time. Default: `true`.
                     By `align_time_step = false` we ensure that the time-step does _not_ change within
                     `time_step!(simulation)`

- `minimum_relative_step`: time steps smaller than `Δt * minimum_relative_step` will be skipped.
                           This avoids extremely high values when writing the pressure to disk.
                           Default value is 0. See github.com/CliMA/Oceananigans.jl/issues/3593 for details.
"""
function Simulation(model; Δt,
                    verbose = true,
                    stop_iteration = Inf,
                    stop_time = Inf,
                    wall_time_limit = Inf,
                    align_time_step = true,
                    minimum_relative_step = 0)

   if verbose && stop_iteration == Inf && stop_time == Inf && wall_time_limit == Inf
       @warn "This simulation will run forever as stop iteration = stop time " *
             "= wall time limit = Inf."
   end

   Δt = validate_Δt(Δt, architecture(model))

   diagnostics = OrderedDict{Symbol, AbstractDiagnostic}()
   output_writers = OrderedDict{Symbol, AbstractOutputWriter}()
   callbacks = OrderedDict{Symbol, Callback}()

   callbacks[:stop_time_exceeded] = Callback(stop_time_exceeded)
   callbacks[:stop_iteration_exceeded] = Callback(stop_iteration_exceeded)
   callbacks[:wall_time_limit_exceeded] = Callback(wall_time_limit_exceeded)

   nan_checker = default_nan_checker(model)
   if !isnothing(nan_checker) # otherwise don't bother
       callbacks[:nan_checker] = Callback(nan_checker, IterationInterval(100))
   end

   # Convert numbers to floating point; otherwise preserve type (eg for DateTime types)
   #    TODO: implement TT = timetype(model) and FT = eltype(model)
   TT = eltype(model)
   Δt = Δt isa Number ? TT(Δt) : Δt
   stop_time = stop_time isa Number ? TT(stop_time) : stop_time

   return Simulation(model,
                     Δt,
                     Float64(stop_iteration),
                     stop_time,
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

function Base.show(io::IO, s::Simulation)
    modelstr = summary(s.model)
    return print(io, "Simulation of ", modelstr, "\n",
                     "├── Next time step: $(prettytime(s.Δt))", "\n",
                     "├── Elapsed wall time: $(prettytime(s.run_wall_time))", "\n",
                     "├── Wall time per iteration: $(prettytime(s.run_wall_time / iteration(s)))", "\n",
                     "├── Stop time: $(prettytime(s.stop_time))", "\n",
                     "├── Stop iteration: $(s.stop_iteration)", "\n",
                     "├── Wall time limit: $(s.wall_time_limit)", "\n",
                     "├── Minimum relative step: ", prettysummary(s.minimum_relative_step), "\n",
                     "├── Callbacks: $(ordered_dict_show(s.callbacks, "│"))", "\n",
                     "├── Output writers: $(ordered_dict_show(s.output_writers, "│"))", "\n",
                     "└── Diagnostics: $(ordered_dict_show(s.diagnostics, "│"))")
end

#####
##### Utilities
#####

"""
    validate_Δt(Δt, arch)

Make sure different workers are using the same time step
"""
function validate_Δt(Δt, arch::Distributed)
    Δt_min = all_reduce(min, Δt, arch)
    if Δt != Δt_min
        @warn "On rank $(arch.local_rank), Δt = $Δt is not the same as for the other workers. Using the minimum Δt = $Δt_min instead."
    end
    return Δt_min
end

# Fallback
validate_Δt(Δt, arch) = Δt

"""
    time(sim::Simulation)

Return the current simulation time.
"""
Base.time(sim::Simulation) = time(sim.model)

"""
    iteration(sim::Simulation)

Return the current simulation iteration.
"""
iteration(sim::Simulation) = iteration(sim.model)

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
    reset_clock!(sim.model)
    sim.stop_iteration = Inf
    sim.stop_time = Inf
    sim.wall_time_limit = Inf
    sim.run_wall_time = 0.0
    sim.initialized = false
    sim.running = true
    reset!(timestepper(sim.model))
    return nothing
end

# Fallback. Models without clocks should extend this function.
"""
    reset_clock!(model::AbstractModel)

Reset `model.clock` to its initial state.
"""
reset_clock!(model::AbstractModel) = reset!(model.clock)

#####
##### Default stop criteria callback functions
#####

wall_time_msg(sim) = string("Simulation is stopping after running for ", run_wall_time(sim), ".")

function stop_iteration_exceeded(sim)
    if sim.model.clock.iteration >= sim.stop_iteration
        if sim.verbose
            msg = string("Model iteration ", iteration(sim), " equals or exceeds stop iteration ", Int(sim.stop_iteration), ".")
            @info wall_time_msg(sim)
            @info msg
        end

        sim.running = false
    end

    return nothing
end

function stop_time_exceeded(sim)
    if sim.model.clock.time >= sim.stop_time
        if sim.verbose
            msg = string("Simulation time ", prettytime(sim), " equals or exceeds stop time ", prettytime(sim.stop_time), ".")
            @info wall_time_msg(sim)
            @info msg
        end

        sim.running = false
    end

    return nothing
end

function wall_time_limit_exceeded(sim)
    if sim.run_wall_time >= sim.wall_time_limit
        if sim.verbose
            msg = string("Simulation run time ", run_wall_time(sim), " equals or exceeds wall time limit ", prettytime(sim.wall_time_limit), ".")
            @info wall_time_msg(sim)
            @info msg
        end

        sim.running = false
    end

    return nothing
end

#####
##### Writing output and checkpointing
#####

# Fallback, to be elaborated on
write_output!(writer::JLD2Writer,   sim::Simulation) = write_output!(writer, sim.model)
write_output!(writer::NetCDFWriter, sim::Simulation) = write_output!(writer, sim.model)
write_output!(writer::Checkpointer, sim::Simulation) = write_output!(writer, sim.model)
