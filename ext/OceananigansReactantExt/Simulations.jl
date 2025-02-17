module Simulations

using Reactant
using Oceananigans

using OrderedCollections: OrderedDict

using ..Architectures: ReactantState
using Oceananigans: AbstractModel, run_diagnostic!
using Oceananigans.Architectures: architecture
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.OutputWriters: write_output!

using Oceananigans.Simulations:
    validate_Δt,
    stop_iteration_exceeded,
    add_dependencies!,
    reset!,
    AbstractDiagnostic,
    AbstractOutputWriter

import Oceananigans.Simulations: Simulation, aligned_time_step, initialize!

const ReactantModel = AbstractModel{<:Any, <:ReactantState}
const ReactantSimulation = Simulation{<:ReactantModel}

aligned_time_step(::ReactantSimulation, Δt) = Δt

function initialize!(sim::ReactantSimulation)
    if sim.verbose
        @info "Initializing simulation..."
        start_time = time_ns()
    end

    model = sim.model
    clock = model.clock

    update_state!(model)

    # Output and diagnostics initialization
    [add_dependencies!(sim.diagnostics, writer) for writer in values(sim.output_writers)]

    # Initialize schedules
    scheduled_activities = Iterators.flatten((values(sim.diagnostics),
                                              values(sim.callbacks),
                                              values(sim.output_writers)))

    for activity in scheduled_activities
        initialize!(activity.schedule, sim.model)
    end

    # Reset! the model time-stepper, evaluate all diagnostics, and write all output at first iteration
    @trace if clock.iteration == 0
        reset!(timestepper(sim.model))

        # Initialize schedules and run diagnostics, callbacks, and output writers
        for diag in values(sim.diagnostics)
            run_diagnostic!(diag, model)
        end

        for callback in values(sim.callbacks)
            callback.callsite isa TimeStepCallsite && callback(sim)
        end

        for writer in values(sim.output_writers)
            writer.schedule(sim.model)
            write_output!(writer, model)
        end
    end

    sim.initialized = true

    if sim.verbose
        initialization_time = prettytime(1e-9 * (time_ns() - start_time))
        @info "    ... simulation initialization complete ($initialization_time)"
    end

    return nothing
end

function Simulation(model::ReactantModel; Δt,
                    verbose = true,
                    stop_iteration = Inf,
                    wall_time_limit = Inf,
                    minimum_relative_step = 0)

   Δt = validate_Δt(Δt, architecture(model))

   diagnostics = OrderedDict{Symbol, AbstractDiagnostic}()
   output_writers = OrderedDict{Symbol, AbstractOutputWriter}()
   callbacks = OrderedDict{Symbol, Callback}()

   callbacks[:stop_iteration_exceeded] = Callback(stop_iteration_exceeded)

   # Convert numbers to floating point; otherwise preserve type (eg for DateTime types)
   #    TODO: implement TT = timetype(model) and FT = eltype(model)
   TT = eltype(model)
   Δt = Δt isa Number ? TT(Δt) : Δt

   return Simulation(model,
                     Δt,
                     Float64(stop_iteration),
                     nothing, # disallow stop_time
                     Float64(wall_time_limit),
                     diagnostics,
                     output_writers,
                     callbacks,
                     0.0,
                     false,
                     false,
                     verbose,
                     Float64(minimum_relative_step))
end

end # module
