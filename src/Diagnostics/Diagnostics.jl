module Diagnostics

export
    TimeSeries, FieldMaximum,
    CFL, AdvectiveCFL, DiffusiveCFL, NaNChecker,
    run_diagnostic!

using Oceananigans
using Oceananigans.Operators

using Oceananigans: AbstractDiagnostic

include("nan_checker.jl")
include("time_series.jl")
include("field_maximum.jl")
include("cfl.jl")

end
