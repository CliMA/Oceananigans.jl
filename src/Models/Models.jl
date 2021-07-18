module Models

export
    NonhydrostaticModel, NonDimensionalNonhydrostaticModel, ShallowWaterModel,
    HydrostaticFreeSurfaceModel, VectorInvariant,
    ExplicitFreeSurface, ImplicitFreeSurface,
    HydrostaticSphericalCoriolis, VectorInvariantEnstrophyConserving,
    PrescribedVelocityFields

using Oceananigans: AbstractModel

import Oceananigans.Architectures: device_event

device_event(model::AbstractModel) = device_event(model.architecture)

abstract type AbstractNonhydrostaticModel{TS} <: AbstractModel{TS} end

include("NonhydrostaticModels/NonhydrostaticModels.jl")
include("HydrostaticFreeSurfaceModels/HydrostaticFreeSurfaceModels.jl")
include("ShallowWaterModels/ShallowWaterModels.jl")

using .NonhydrostaticModels: NonhydrostaticModel

using .HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel, VectorInvariant,
    ExplicitFreeSurface, ImplicitFreeSurface,
    HydrostaticSphericalCoriolis,
    VectorInvariantEnstrophyConserving, VectorInvariantEnergyConserving,
    PrescribedVelocityFields

using .ShallowWaterModels: ShallowWaterModel

end
