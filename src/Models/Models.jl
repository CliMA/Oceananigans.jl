module Models

export
    IncompressibleModel, NonDimensionalIncompressibleModel, HydrostaticFreeSurfaceModel, ShallowWaterModel,
    ExplicitFreeSurface, VectorInvariant, HydrostaticSphericalCoriolis, VectorInvariantEnstrophyConserving

using Oceananigans: AbstractModel

abstract type AbstractIncompressibleModel{TS} <: AbstractModel{TS} end

include("IncompressibleModels/IncompressibleModels.jl")
include("HydrostaticFreeSurfaceModels/HydrostaticFreeSurfaceModels.jl")
include("ShallowWaterModels/ShallowWaterModels.jl")

using .IncompressibleModels: IncompressibleModel, NonDimensionalIncompressibleModel
using .ShallowWaterModels: ShallowWaterModel

using .HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel, ExplicitFreeSurface, VectorInvariant,
    HydrostaticSphericalCoriolis, VectorInvariantEnstrophyConserving

end
