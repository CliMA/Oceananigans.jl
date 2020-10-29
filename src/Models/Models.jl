module Models

export IncompressibleModel, NonDimensionalModel, Clock, tick!, fields

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

"""
    fields(model)

Returns a flattened `NamedTuple` of the fields in `model.velocities` and `model.tracers`.
"""
fields(model) = merge(model.velocities, model.tracers)

include("clock.jl")
include("incompressible_model.jl")
include("non_dimensional_model.jl")
include("show_models.jl")

end
