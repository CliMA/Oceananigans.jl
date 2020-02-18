module Models

export Model, ChannelModel, NonDimensionalModel, Clock, increment_clock!

using Dates

using Oceananigans.Architectures
using Oceananigans.Fields
using Oceananigans.Coriolis
using Oceananigans.Buoyancy
using Oceananigans.TurbulenceClosures
using Oceananigans.BoundaryConditions
using Oceananigans.Solvers
using Oceananigans.Forcing
using Oceananigans.Utils

"""
    AbstractModel

Abstract supertype for models.
"""
abstract type AbstractModel end

include("clock.jl")
include("model.jl")
include("non_dimensional_model.jl")
include("show_models.jl")

end
