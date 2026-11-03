module Simulations

export first_time_step!, time_step_for!

using Reactant
using Reactant: Periodic, Binomial
using Oceananigans

using OrderedCollections: OrderedDict

using ..Architectures: ReactantState
using ..TimeSteppers: ReactantModel

using Oceananigans: initialize!, run_diagnostic!, Callback, TimeStepCallsite, TimeInterval, IterationInterval, TimeStepWizard, TimeDerivative
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: initialize_boundary_conditions!
using Oceananigans.Diagnostics: NaNChecker
using Oceananigans.TimeSteppers: update_state!, time_step!, QuasiAdamsBashforth2TimeStepper
using Oceananigans.Utils: prettytime
using Oceananigans.OutputWriters: write_output!

using Oceananigans.Simulations:
    AutomaticDifferentiation,
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
    wall_time_limit_exceeded,
    Simulation,
    iteration,
    add_callback!

include("simulation.jl")
include("run.jl")

end # module
