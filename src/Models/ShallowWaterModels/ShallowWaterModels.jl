module ShallowWaterModels

export ShallowWaterModel, ShallowWaterScalarDiffusivity,
       ConservativeFormulation, VectorInvariantFormulation

using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll

using Adapt
using Oceananigans.Utils: launch!

import Oceananigans: fields, prognostic_fields

#####
##### ShallowWaterModel definition
#####

include("shallow_water_model.jl")
include("set_shallow_water_model.jl")
include("show_shallow_water_model.jl")

#####
##### Time-stepping ShallowWaterModels
#####

"""
    fields(model::ShallowWaterModel)

Return a flattened `NamedTuple` of the fields in `model.solution` and `model.tracers` for
a `ShallowWaterModel` model.
"""
fields(model::ShallowWaterModel) = merge(model.solution, model.tracers)

"""
    prognostic_fields(model::HydrostaticFreeSurfaceModel)

Return a flattened `NamedTuple` of the prognostic fields associated with `ShallowWaterModel`.
"""
prognostic_fields(model::ShallowWaterModel) = fields(model)

include("solution_and_tracer_tendencies.jl")
include("compute_shallow_water_tendencies.jl")
include("update_shallow_water_state.jl")
include("shallow_water_advection_operators.jl")
include("shallow_water_diffusion_operators.jl")
include("shallow_water_cell_advection_timescale.jl")

end # module
