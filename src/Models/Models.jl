module Models

export IncompressibleModel, NonDimensionalModel, fields

function fields end

include("IncompressibleModels/IncompressibleModels.jl")

using .IncompressibleModels: IncompressibleModel, NonDimensionalModel

include("ShallowWaterModels/ShallowWaterModels.jl")

using .IncompressibleModels: ShallowWaterModel

end
