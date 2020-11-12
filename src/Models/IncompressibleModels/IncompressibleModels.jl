module IncompressibleModels

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!

include("incompressible_model.jl")
include("non_dimensional_model.jl")
include("show_incompressible_model.jl")

include("update_hydrostatic_pressure.jl")
include("update_state.jl")

include("pressure_correction.jl")

include("velocity_and_tracer_tendencies.jl")
include("calculate_tendencies.jl")

end # module
