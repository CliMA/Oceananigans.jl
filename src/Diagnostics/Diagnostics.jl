module Diagnostics

export StateChecker, CFL, AdvectiveCFL, DiffusiveCFL

using CUDA
using Oceananigans
using Oceananigans.Operators

using Oceananigans: AbstractDiagnostic
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval

import Base: show
import Oceananigans: run_diagnostic!

include("state_checker.jl")
include("cfl.jl")

end # module
