module Models

export IncompressibleModel, NonDimensionalModel, Clock, tick!, state

using Adapt

using Oceananigans.Architectures
using Oceananigans.Fields
using Oceananigans.Coriolis
using Oceananigans.Buoyancy
using Oceananigans.TurbulenceClosures
using Oceananigans.BoundaryConditions
using Oceananigans.Solvers
using Oceananigans.Forcing
using Oceananigans.Utils

"""
    AbstractModel

Abstract supertype for models.
"""
abstract type AbstractModel end

"""
    state(model)

Returns a `NamedTuple` with fields `velocities, tracers, diffusivities, tendencies` 
corresponding to `NamedTuple`s of `OffsetArray`s that reference each of the field's data.
"""
@inline state(model) = (   velocities = datatuple(model.velocities),
                              tracers = datatuple(model.tracers),
                        diffusivities = datatuple(model.diffusivities))

include("clock.jl")
include("incompressible_model.jl")
include("non_dimensional_model.jl")
include("show_models.jl")

end
