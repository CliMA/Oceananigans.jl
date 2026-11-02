module ShallowWaterModels

export ShallowWaterModel, ShallowWaterScalarDiffusivity,
       ConservativeFormulation, VectorInvariantFormulation

using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @index, @kernel

using Oceananigans.Advection: Advection, UpwindBiased, WENO
using Oceananigans.BoundaryConditions: BoundaryConditions, FieldBoundaryConditions, compute_x_bcs!, compute_y_bcs!
using Oceananigans.Coriolis: Coriolis, x_f_cross_U, y_f_cross_U
using Oceananigans.DistributedComputations: DistributedComputations
using Oceananigans.Operators: Operators, Az⁻¹ᶜᶜᶜ, Az⁻¹ᶜᶠᶜ, Az⁻¹ᶠᶜᶜ, div_xyᶜᶜᶜ, Δx_qᶜᶠᶜ, Δxᶠᶜᶜ, Δy_qᶠᶜᶜ, Δyᶜᶠᶜ,
                              δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, ℑxyᶠᶠᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ∂xᵣᶠᶜᶜ, ∂yᵣᶜᶠᶜ
using Oceananigans.Utils: launch!

import Oceananigans: fields, prognostic_fields
import Oceananigans.Simulations: timestepper

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
fields(model::ShallowWaterModel) = merge(model.solution, model.tracers)

"""
$(TYPEDSIGNATURES)

Return a flattened `NamedTuple` of the prognostic fields associated with `ShallowWaterModel`.
"""
prognostic_fields(model::ShallowWaterModel) = fields(model)

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
