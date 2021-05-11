module Diagnostics

export NaNChecker, StateChecker, CFL, AdvectiveCFL, DiffusiveCFL, WindowedSpatialAverage

using CUDA
using Oceananigans
using Oceananigans.Operators

using Oceananigans: AbstractDiagnostic
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval

import Base: show
import Oceananigans: run_diagnostic!

include("nan_checker.jl")
include("state_checker.jl")
include("cfl.jl")
include("windowed_spatial_average.jl")

end
