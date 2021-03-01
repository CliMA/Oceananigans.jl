module Diagnostics

export
    NaNChecker,
    CFL, AdvectiveCFL, DiffusiveCFL,
    run_diagnostic!,
    TimeInterval, IterationInterval, WallTimeInterval, WindowedSpatialAverage

using CUDA
using Oceananigans
using Oceananigans.Operators
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval

using Oceananigans: AbstractDiagnostic

include("nan_checker.jl")
include("cfl.jl")
include("field_slicer.jl")
include("windowed_spatial_average.jl")

end
