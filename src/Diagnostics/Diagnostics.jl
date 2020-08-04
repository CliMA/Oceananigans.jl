module Diagnostics

export
    HorizontalAverage, TimeSeries, FieldMaximum,
    CFL, AdvectiveCFL, DiffusiveCFL, NaNChecker,
    run_diagnostic

using Oceananigans,
      Oceananigans.Operators

using Oceananigans: AbstractDiagnostic

include("nan_checker.jl")
include("horizontal_average.jl")
include("time_series.jl")
include("field_maximum.jl")
include("cfl.jl")

end
