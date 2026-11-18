module Simulations

export first_time_step!, time_step_for!

using Reactant: Reactant, @trace, ConcreteRNumber, Sharding, Periodic, Binomial
using Oceananigans: Oceananigans
using Oceananigans: initialize!, Callback, TimeStepCallsite, TimeInterval, IterationInterval, TimeStepWizard, TimeDerivative, Distributed, SpecifiedTimes
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: initialize_boundary_conditions!
using Oceananigans.Diagnostics: NaNChecker
using Oceananigans.Simulations:
    AutomaticDifferentiation,
    stop_iteration_exceeded,
    ModelCallsite,
    GenericName,
    finalize!,
    stop_time_exceeded,
    wall_time_limit_exceeded,
    Simulation,
    iteration,
    add_callback!
using Oceananigans.TimeSteppers: update_state!, time_step!, QuasiAdamsBashforth2TimeStepper
using Oceananigans.Utils: prettytime

using OrderedCollections: OrderedDict

using ..Architectures: AnyConcreteReactantArray
using ..TimeSteppers: ReactantModel

include("simulation.jl")
include("run.jl")

end # module
