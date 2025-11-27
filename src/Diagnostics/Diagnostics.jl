module Diagnostics

export StateChecker, CFL, AdvectiveCFL, DiffusiveCFL

using Oceananigans
using Oceananigans.Operators

using Oceananigans: AbstractDiagnostic

import Oceananigans: run_diagnostic!

function cell_diffusion_timescale end

include("state_checker.jl")
include("nan_checker.jl")
include("cfl.jl")

end # module
