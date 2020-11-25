module Models

export IncompressibleModel, NonDimensionalModel

include("IncompressibleModels/IncompressibleModels.jl")
include("ShallowWaterModels/ShallowWaterModels.jl")

using .IncompressibleModels: IncompressibleModel, NonDimensionalModel
using .ShallowWaterModels: ShallowWaterModel

end
