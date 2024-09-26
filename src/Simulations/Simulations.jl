module Simulations

export TimeStepWizard, conjure_time_step_wizard!
export Simulation
export run!
export Callback, add_callback!
export iteration
export stopwatch
export JLD2_output!

using Oceananigans.Models
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.TimeSteppers
using Oceananigans.Utils

using Oceananigans.Advection: cell_advection_timescale
using Oceananigans: AbstractDiagnostic, AbstractOutputWriter, fields

using OrderedCollections: OrderedDict

import Base: show

function unique_name(prefix::Symbol, existing_names)
    if !(prefix ∈ existing_names)
        name = prefix
    else # make it unique
        n = 1
        while Symbol(prefix, n) ∈ existing_names
            n += 1
        end
        name = Symbol(prefix, n)
    end
    return name
end

include("callback.jl")
include("simulation.jl")
include("run.jl")
include("time_step_wizard.jl")
include("output_helpers.jl")

end # module
