module Models

export IncompressibleModel, NonDimensionalModel, HydrostaticFreeSurfaceModel, ShallowWaterModel

using Oceananigans: AbstractModel

abstract type AbstractIncompressibleModel{TS} <: AbstractModel{TS} end

include("IncompressibleModels/IncompressibleModels.jl")
include("HydrostaticFreeSurfaceModels/HydrostaticFreeSurfaceModels.jl")
include("ShallowWaterModels/ShallowWaterModels.jl")

using .IncompressibleModels: IncompressibleModel, NonDimensionalModel
using .HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using .ShallowWaterModels: ShallowWaterModel

end
