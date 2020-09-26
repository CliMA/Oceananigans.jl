module Models

export IncompressibleModel, NonDimensionalModel, Clock, tick!, all_model_fields, fields

using Adapt

using Oceananigans.Architectures
using Oceananigans.Fields
using Oceananigans.Coriolis
using Oceananigans.Buoyancy
using Oceananigans.TurbulenceClosures
using Oceananigans.BoundaryConditions
using Oceananigans.Solvers
using Oceananigans.Utils

"""
    AbstractModel

Abstract supertype for models.
"""
abstract type AbstractModel end

regularize_diffusivity_fields(diffusivities::Tuple) = (diffusivities=datatuple(diffusivities),)
regularize_diffusivity_fields(diffusivities::NamedTuple) = datatuple(diffusivities)
regularize_diffusivity_fields(::Nothing) = NamedTuple()

"""
    all_model_fields(model)

Returns a flattened `NamedTuple` with data from the `NamedTuples`
`model.velocities`, `model.tracers`, and `model.diffusivities`,
corresponding `OffsetArray`s that reference each of the field's data.
"""
@inline all_model_fields(model) = merge(datatuple(model.velocities),
                                        datatuple(model.tracers),
                                        regularize_diffusivity_fields(model.diffusivities))

fields(model) = merge(model.velocities, model.tracers)

include("clock.jl")
include("incompressible_model.jl")
include("non_dimensional_model.jl")
include("show_models.jl")

end
