using
    Test,
    Statistics,
    OffsetArrays

import FFTW

using
    Oceananigans,
    Oceananigans.Operators,
    Oceananigans.TurbulenceClosures

using Oceananigans: PoissonSolver, PPN, PNN, solve_poisson_3d!,
                    parentdata,
                    buoyancy_perturbation, fill_halo_regions!, run_diagnostic

archs = (CPU(),)
@hascuda archs = (CPU(), GPU())
@hascuda using CuArrays

float_types = (Float32, Float64)

@testset "Oceananigans" begin
    include("test_grids.jl")
    include("test_fields.jl")
    include("test_halo_regions.jl")
    include("test_operators.jl")
    include("test_poisson_solvers.jl")
    include("test_models.jl")
    include("test_time_stepping.jl")
    include("test_boundary_conditions.jl")
    include("test_forcings.jl")
    include("test_turbulence_closures.jl")
    include("test_dynamics.jl")
    include("test_diagnostics.jl")
    include("test_output_writers.jl")
    include("test_regression.jl")
end
