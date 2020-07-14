module Diagnostics

export
    HorizontalAverage, ZonalAverage, VolumeAverage,
    TimeSeries, FieldMaximum,
    CFL, AdvectiveCFL, DiffusiveCFL, NaNChecker,
    run_diagnostic

using Oceananigans,
      Oceananigans.Operators

using Oceananigans: AbstractDiagnostic
using Oceananigans: AbstractAverage

include("diagnostics_kernels.jl")
include("nan_checker.jl")
include("averages.jl")
include("time_series.jl")
include("field_maximum.jl")
include("cfl.jl")

end
