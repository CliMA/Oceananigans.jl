module Diagnostics

export
    NaNChecker,
    CFL, AdvectiveCFL, DiffusiveCFL,
    run_diagnostic!,
    TimeInterval, IterationInterval, WallTimeInterval

using CUDA
using Oceananigans
using Oceananigans.Operators
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval

using Oceananigans: AbstractDiagnostic

include("nan_checker.jl")
include("cfl.jl")

end
