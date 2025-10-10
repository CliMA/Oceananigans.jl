module Diagnostics

export StateChecker, CFL, AdvectiveCFL, DiffusiveCFL

using Oceananigans
using Oceananigans.Operators

using Oceananigans: AbstractDiagnostic
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval

import Base: show
import Oceananigans: run_diagnostic!

import Oceananigans.Advection: cell_advection_timescale

function cell_diffusion_timescale end

include("state_checker.jl")
include("nan_checker.jl")
include("cfl.jl")

end # module
