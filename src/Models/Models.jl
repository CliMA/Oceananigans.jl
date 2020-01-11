module Models

export Model, ChannelModel, NonDimensionalModel, Clock

using Oceananigans.Coriolis
using Oceananigans.Buoyancy
using Oceananigans.Utils

"""
    AbstractModel

Abstract supertype for models.
"""
abstract type AbstractModel end

include("model_utils.jl")
include("clock.jl")
include("model.jl")
include("channel_model.jl")
include("non_dimensional_model.jl")
include("show_models.jl")

end
