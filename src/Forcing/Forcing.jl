module Forcing

export ModelForcing, SimpleForcing, ParameterizedForcing, RestoringForce, GaussianMask, LinearTarget

using Oceananigans.Fields

zeroforcing(args...) = 0

include("simple_forcing.jl")
include("model_forcing.jl")
include("parameterized_forcing.jl")

end
