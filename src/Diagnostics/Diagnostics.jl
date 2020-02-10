module Diagnostics

export
    HorizontalAverage, TimeSeries, FieldMaximum,
    CFL, AdvectiveCFL, DiffusiveCFL, NaNChecker

using Oceananigans,
      Oceananigans.Operators

using Oceananigans: AbstractDiagnostic

include("diagnostics_kernels.jl")
include("nan_checker.jl")
include("horizontal_average.jl")
include("time_series.jl")
include("field_maximum.jl")
include("cfl.jl")

end
