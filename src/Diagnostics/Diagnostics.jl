module Diagnostics

export
    Average, TimeSeries, FieldMaximum, WindowedTimeAverage,
    CFL, AdvectiveCFL, DiffusiveCFL, NaNChecker,
    run_diagnostic

using Oceananigans
using Oceananigans.Operators

using Oceananigans: AbstractDiagnostic

include("nan_checker.jl")
include("average.jl")
include("windowed_time_average.jl")
include("time_series.jl")
include("field_maximum.jl")
include("cfl.jl")

end
