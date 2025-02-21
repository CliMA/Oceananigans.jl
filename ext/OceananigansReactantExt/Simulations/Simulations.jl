module Simulations

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

import Oceananigans.Simulations: Simulation, aligned_time_step, initialize!

include("simulation.jl")
include("run.jl")

end # module
