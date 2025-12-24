module Forcings

export Forcing, ContinuousForcing, DiscreteForcing, Relaxation, GaussianMask, PiecewiseLinearMask, LinearTarget, AdvectiveForcing

using Oceananigans.Fields
using Oceananigans.OutputReaders: FlavorOfFTS
using Oceananigans.Units: Time
import Oceananigans.Architectures: on_architecture

include("multiple_forcings.jl")
include("continuous_forcing.jl")
include("discrete_forcing.jl")
include("relaxation.jl")
include("advective_forcing.jl")
include("forcing.jl")
include("model_forcing.jl")

end # module
