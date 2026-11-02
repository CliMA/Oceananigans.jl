module Diagnostics

export StateChecker, CFL, AdvectiveCFL, DiffusiveCFL

using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans: Oceananigans, fields, AbstractDiagnostic

using Oceananigans: run_diagnostic!

function cell_diffusion_timescale end

include("state_checker.jl")
include("nan_checker.jl")
include("cfl.jl")

end # module
