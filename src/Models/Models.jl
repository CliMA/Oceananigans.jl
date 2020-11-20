module Models

export IncompressibleModel, NonDimensionalModel, fields

function fields end

include("IncompressibleModels/IncompressibleModels.jl")
include("ShallowWaterModels/ShallowWaterModels.jl")

using .IncompressibleModels: IncompressibleModel, NonDimensionalModel
using .ShallowWaterModels: ShallowWaterModel

end
