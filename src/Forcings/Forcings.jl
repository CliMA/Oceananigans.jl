module Forcings

export Forcing, ContinuousForcing, DiscreteForcing, Relaxation, GaussianMask, LinearTarget

using Oceananigans.Fields

include("continuous_forcing.jl")
include("discrete_forcing.jl")
include("relaxation.jl")
include("forcing.jl")
include("model_forcing.jl")

end
