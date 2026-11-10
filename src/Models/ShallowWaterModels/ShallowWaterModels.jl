module ShallowWaterModels

export ShallowWaterModel, ShallowWaterScalarDiffusivity,
       ConservativeFormulation, VectorInvariantFormulation

using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @index, @kernel

using Oceananigans: Oceananigans, fields, prognostic_fields
using Oceananigans.Advection: Advection
using Oceananigans.Architectures: Architectures
using Oceananigans.Fields: Fields
using Oceananigans.Simulations: Simulations
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.TurbulenceClosures: TurbulenceClosures
using Oceananigans.Utils: launch!

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
$(TYPEDSIGNATURES)

Return a flattened `NamedTuple` of the fields in `model.solution` and `model.tracers` for
a `ShallowWaterModel` model.
"""
Oceananigans.fields(model::ShallowWaterModel) = merge(model.solution, model.tracers)

"""
$(TYPEDSIGNATURES)

Return a flattened `NamedTuple` of the prognostic fields associated with `ShallowWaterModel`.
"""
Oceananigans.prognostic_fields(model::ShallowWaterModel) = fields(model)

include("solution_and_tracer_tendencies.jl")
include("compute_shallow_water_tendencies.jl")
include("shallow_water_rk3_substep.jl")
include("shallow_water_ab2_step.jl")
include("update_shallow_water_state.jl")
include("cache_shallow_water_tendencies.jl")
include("shallow_water_advection_operators.jl")
include("shallow_water_diffusion_operators.jl")
include("shallow_water_cell_advection_timescale.jl")

end # module
