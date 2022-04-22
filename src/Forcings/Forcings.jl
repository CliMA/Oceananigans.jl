module Forcings

export Forcing, ContinuousForcing, DiscreteForcing, Relaxation, GaussianMask, LinearTarget, AdvectiveForcing

using Oceananigans.Fields

include("continuous_forcing.jl")
include("discrete_forcing.jl")
include("relaxation.jl")
include("advective_forcing.jl")
include("forcing.jl")
include("model_forcing.jl")
include("multiple_forcings.jl")

end # module
