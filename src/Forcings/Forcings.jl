module Forcings

export Forcing, ContinuousForcing, DiscreteForcing, Relaxation, GaussianMask, PiecewiseLinearMask, LinearTarget, AdvectiveForcing

using Oceananigans.Fields: field, location
using Oceananigans.OutputReaders: FlavorOfFTS
using Oceananigans.Units: Time
using Oceananigans.Architectures: Architectures, on_architecture

using Adapt: Adapt, adapt

include("multiple_forcings.jl")
include("continuous_forcing.jl")
include("discrete_forcing.jl")
include("relaxation.jl")
include("advective_forcing.jl")
include("forcing.jl")
include("model_forcing.jl")

end # module
