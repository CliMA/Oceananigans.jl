module Models

export IncompressibleModel, NonDimensionalModel, Clock, tick!, fields

using Adapt

"""
    fields(model)

Returns a flattened `NamedTuple` of the fields in `model.velocities` and `model.tracers`.
"""
fields(model) = merge(model.velocities, model.tracers)

include("clock.jl")

include("IncompressibleModels/IncompressibleModels.jl")

using .IncompressibleModels: IncompressibleModel, NonDimensionalModel

end
