module Simulations

export
    TimeStepWizard,
    conjure_time_step_wizard!,
    Simulation,
    run!,
    Callback,
    add_callback!,
    iteration

using DocStringExtensions: TYPEDSIGNATURES
using OrderedCollections: OrderedDict

using ..Oceananigans: Oceananigans, AbstractDiagnostic, AbstractOutputWriter
using ..Advection: Advection
using ..OutputWriters: OutputWriters, Checkpointer, checkpoint
using ..TimeSteppers: TimeSteppers
using ..Utils: Utils, IterationInterval, ordered_dict_show

# To be extended in the `Models` module
timestepper(model) = nothing

include("callback.jl")
include("simulation.jl")
include("run.jl")
include("time_step_wizard.jl")

end # module
