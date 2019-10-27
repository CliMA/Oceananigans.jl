using
    Oceananigans,
    Oceananigans.Operators,
    Oceananigans.TurbulenceClosures,
    Test,
    Random,
    JLD2,
    Printf,
    Statistics,
    OffsetArrays,
    FFTW

@hascuda using CuArrays

using Statistics: mean
using LinearAlgebra: norm
using GPUifyLoops: @launch, @loop
using NCDatasets: Dataset

using Oceananigans: PoissonSolver, PPN, PNN, solve_poisson_3d!,
                    velocity_div!,
                    architecture, device, launch_config, datatuples, Face, Cell, with_tracers,
                    interiorparent, interior, fill_halo_regions!, run_diagnostic, location,
                    TracerFields, buoyancy_frequency_squared, thermal_expansion, haline_contraction, ρ′,
                    RoquetIdealizedNonlinearEquationOfState, required_tracers

import Oceananigans: interior, datatuple

using Oceananigans.TurbulenceClosures

using Oceananigans.TimeSteppers: _compute_w_from_continuity!

using Oceananigans.TurbulenceClosures: ∂x_caa, ∂x_faa, ∂x²_caa, ∂x²_faa,
                                       ∂y_aca, ∂y_afa, ∂y²_aca, ∂y²_afa,
                                       ∂z_aac, ∂z_aaf, ∂z²_aac, ∂z²_aaf,
                                       ▶x_caa, ▶x_faa, ▶y_aca, ▶y_afa,
                                       ▶z_aac, ▶z_aaf

using Oceananigans.AbstractOperations

using Oceananigans.AbstractOperations: Computation, compute!

const seed = 420  # Random seed to use for all pseudorandom number generators.


datatuple(A) = NamedTuple{propertynames(A)}(Array(data(a)) for a in A)

function get_output_tuple(output, iter, tuplename)
    file = jldopen(output.filepath, "r")
    output_tuple = file["timeseries/$tuplename/$iter"]
    close(file)
    return output_tuple
end

float_types = (Float32, Float64)

archs = (CPU(),)
@hascuda archs = (CPU(), GPU())

closures = (
            :ConstantIsotropicDiffusivity,
            :ConstantAnisotropicDiffusivity,
            :SmagorinskyLilly,
            :BlasiusSmagorinsky,
            :RozemaAnisotropicMinimumDissipation,
            :VerstappenAnisotropicMinimumDissipation
           )

EquationsOfState = (LinearEquationOfState, RoquetIdealizedNonlinearEquationOfState)

@testset "Oceananigans" begin
    include("test_grids.jl")
    include("test_fields.jl")
    include("test_halo_regions.jl")
    include("test_operators.jl")
    include("test_poisson_solvers.jl")
    include("test_coriolis.jl")
    include("test_buoyancy.jl")
    include("test_models.jl")
    include("test_time_stepping.jl")
    include("test_boundary_conditions.jl")
    include("test_forcings.jl")
    include("test_turbulence_closures.jl")
    include("test_dynamics.jl")
    include("test_diagnostics.jl")
    include("test_output_writers.jl")
    include("test_regression.jl")
    include("test_examples.jl")
    include("test_abstract_operations.jl")
    include("test_verification.jl")
end
