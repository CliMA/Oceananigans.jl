module Simulations

export first_time_step!, time_step_for!

using Reactant
using Oceananigans

using OrderedCollections: OrderedDict

using ..Architectures: ReactantState
using ..TimeSteppers: ReactantModel

using Oceananigans: run_diagnostic!
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

import Oceananigans.Simulations:
    iteration,
    add_callback!,
    Simulation,
    aligned_time_step,
    initialize!,
    stop_iteration_exceeded

import Oceananigans.TimeSteppers: time_step!

include("simulation.jl")
include("run.jl")

end # module
