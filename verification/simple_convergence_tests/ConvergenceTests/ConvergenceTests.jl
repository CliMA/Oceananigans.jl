module ConvergenceTests

using JLD2, Statistics

using Oceananigans, Oceananigans.Fields

import Oceananigans: RegularCartesianGrid

include("file_wrangling.jl")
include("analysis.jl")

# c = exp(-t) | ∂c/∂t = - c
# Tests time-stepper convergence.
include("PointExponentialDecay.jl")

# c = 1/√(4πκt) * exp( -(x - Ut)^2 / 4κt )
# Tests x-advection and y-advection.
include("GaussianAdvectionDiffusion.jl")

# c = exp(-κt) * cos(x - Ut)
# Tests x-advection and y-advection.
include("CosineAdvectionDiffusion.jl")

include("ForcedFlowFreeSlip.jl")
include("DoublyPeriodicFreeDecay.jl")

end
