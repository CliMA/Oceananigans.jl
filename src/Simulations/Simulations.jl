module Simulations

export TimeStepWizard, conjure_time_step_wizard!
export Simulation
export run!
export Callback, add_callback!
export iteration

using OrderedCollections: OrderedDict
using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans
using Oceananigans: AbstractDiagnostic, AbstractOutputWriter
using Oceananigans.Advection: Advection
using Oceananigans.OutputWriters
using Oceananigans.TimeSteppers
using Oceananigans.Utils

# To be extended in the `Models` module
timestepper(model) = nothing

include("callback.jl")
include("simulation.jl")
include("run.jl")
include("time_step_wizard.jl")

end # module
