"""
`ConvergenceTests` implements a few simple convergence tests for verifying
the expected numerical behavior of the spatial discretization
and time-stepping
"""
module ConvergenceTests

export test_rate_of_convergence

using Printf
using Statistics
using Logging

using JLD2
using OffsetArrays

using Oceananigans
using Oceananigans.Fields

using Oceananigans: Face, Center
using Oceananigans.Fields: Face, Center, nodes

import Oceananigans: RectilinearGrid
import Oceananigans.Fields: location

figspath = joinpath(@__DIR__, "..", "figs")
ispath(figspath) || mkpath(figspath)

include("file_wrangling.jl")
include("analysis.jl")

# Exponential decay at a point:
#
# c = exp(-t) | ∂c/∂t = - c
#
# Tests time-stepper convergence.
#
include("PointExponentialDecay.jl")

# Utilities for analyzing and plotting the results
# of 1D convergence tests.
#
include("OneDimensionalUtils.jl")

# Advection and diffusion of a 1D Gaussian:
#
# c = 1/√(4πκt) * exp( -(x - Ut)^2 / 4κt )
#
# Tests x-advection and y-advection.
#
include("OneDimensionalGaussianAdvectionDiffusion.jl")

# Advection and diffusion of a 1D cosine:
#
# c = exp(-κt) * cos(x - Ut)
#
# Tests x-advection and y-advection.
#
include("OneDimensionalCosineAdvectionDiffusion.jl")

# Free decay of a 2D sinusoidal tracer patch
#
# c(x, y, t) = exp(-2t) * cos(x) * cos(y)
#
include("TwoDimensionalDiffusion.jl")

# Free decay of a horizontally-advected Taylor-Green vortex:
#
# u(x, y, t) = U + cos(x - Ut) sin(y)
# v(x, y, t) =   - sin(x - Ut) cos(y)
#
include("DoublyPeriodicTaylorGreen.jl")

include("ForcedFlowFreeSlip.jl")

include("ForcedFlowFixedSlip.jl")

function __init__()
    Logging.global_logger(OceananigansLogger())
end

end
