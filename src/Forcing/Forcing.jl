module Forcing

export ModelForcing, SimpleForcing

zeroforcing(args...) = 0

include("simple_forcing.jl")
include("model_forcing.jl")

end
