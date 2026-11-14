module Diagnostics

export StateChecker, CFL, AdvectiveCFL, DiffusiveCFL

using Oceananigans: Oceananigans, AbstractDiagnostic, fields, run_diagnostic!
using Oceananigans.Utils: SIGNATURES, TYPEDSIGNATURES

function cell_diffusion_timescale end

include("state_checker.jl")
include("nan_checker.jl")
include("cfl.jl")

end # module
