module Simulations

export first_time_step!, time_step_for!

using Reactant
using Oceananigans

using OrderedCollections: OrderedDict

using ..Architectures: ReactantState
using ..TimeSteppers: ReactantModel

using Oceananigans: run_diagnostic!, Callback, TimeStepCallsite, TimeInterval, IterationInterval, TimeStepWizard, TimeDerivative
using Oceananigans.Architectures: architecture
using Oceananigans.Diagnostics: NaNChecker
using Oceananigans.TimeSteppers: update_state!, QuasiAdamsBashforth2TimeStepper
using Oceananigans.Utils: prettytime
using Oceananigans.OutputWriters: write_output!

using Oceananigans.Simulations:
    validate_Δt,
    stop_iteration_exceeded,
    add_dependencies!,
    reset!,
    AbstractDiagnostic,
    AbstractOutputWriter,
    ModelCallsite,
    GenericName,
    finalize!,
    stop_time_exceeded,
    wall_time_limit_exceeded

import Oceananigans: run!

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
