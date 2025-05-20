module Diagnostics

export StateChecker, CFL, AdvectiveCFL, DiffusiveCFL,
       MovieMaker, add_movie_maker!

using CUDA
using Oceananigans
using Oceananigans.Operators

using Oceananigans: AbstractDiagnostic
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval

import Base: show
import Oceananigans: run_diagnostic!

include("state_checker.jl")
include("cfl.jl")
include("plotter.jl")

# Implementation of the diagnostic for computing the dissipation rate
include("VarianceDissipationComputations/VarianceDissipationComputations.jl")

using .VarianceDissipationComputations

end # module
