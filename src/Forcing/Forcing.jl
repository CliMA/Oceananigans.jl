module Forcing

export ModelForcing, SimpleForcing

using Oceananigans.Fields

zeroforcing(args...) = 0

include("simple_forcing.jl")
include("model_forcing.jl")

end
