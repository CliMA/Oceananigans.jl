module Models

export IncompressibleModel, NonDimensionalModel, Clock, tick!, fields

function fields end

include("clock.jl")

include("IncompressibleModels/IncompressibleModels.jl")

using .IncompressibleModels: IncompressibleModel, NonDimensionalModel

end
