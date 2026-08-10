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

using Oceananigans: Oceananigans, AbstractDiagnostic, AbstractOutputWriter
using Oceananigans.Advection: Advection
using Oceananigans.OutputWriters: OutputWriters, Checkpointer, checkpoint
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.Utils: Utils, IterationInterval, ordered_dict_show

# To be extended in the `Models` module
timestepper(model) = nothing

include("callback.jl")
include("simulation.jl")
include("run.jl")
include("time_step_wizard.jl")

end # module
