module Models

export
    IncompressibleModel, NonDimensionalIncompressibleModel, ShallowWaterModel,
    HydrostaticFreeSurfaceModel, VectorInvariant,
    ExplicitFreeSurface, ImplicitFreeSurface,
    HydrostaticSphericalCoriolis, VectorInvariantEnstrophyConserving,
    PrescribedVelocityFields

using Oceananigans: AbstractModel

import Oceananigans.Architectures: device_event

device_event(model::AbstractModel) = device_event(model.architecture)

abstract type AbstractIncompressibleModel{TS} <: AbstractModel{TS} end

include("IncompressibleModels/IncompressibleModels.jl")
include("HydrostaticFreeSurfaceModels/HydrostaticFreeSurfaceModels.jl")
include("ShallowWaterModels/ShallowWaterModels.jl")

using .IncompressibleModels: IncompressibleModel

using .HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel, VectorInvariant,
    ExplicitFreeSurface, ImplicitFreeSurface,
    HydrostaticSphericalCoriolis,
    VectorInvariantEnstrophyConserving, VectorInvariantEnergyConserving,
    PrescribedVelocityFields

using .ShallowWaterModels: ShallowWaterModel

end
