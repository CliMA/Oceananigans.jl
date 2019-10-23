module Diagnostics

export
    run_diagnostic,
    HorizontalAverage, Timeseries, FieldMaximum, CFL, AdvectiveCFL, DiffusiveCFL, NaNChecker

using Oceananigans,
      Oceananigans.Operators

using Oceananigans: AbstractDiagnostic

include("kernels.jl")
include("nan_checker.jl")
include("horizontal_average.jl")
include("timeseries.jl")
include("field_maximum.jl")
include("cfl.jl")

end
