using Test
using Printf
using Random
using Statistics
using LinearAlgebra
using Logging

using CUDA
using JLD2
using FFTW
using OffsetArrays
using SeawaterPolynomials

using Oceananigans
using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Advection
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.Coriolis
using Oceananigans.Buoyancy
using Oceananigans.Forcing
using Oceananigans.Solvers
using Oceananigans.Models
using Oceananigans.Simulations
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.TurbulenceClosures
using Oceananigans.AbstractOperations
using Oceananigans.Logger
using Oceananigans.Utils
using Oceananigans.Architectures: device # to resolve conflict with CUDA.device

using Dates: DateTime, Nanosecond
using TimesDates: TimeDate
using Statistics: mean
using LinearAlgebra: norm
using NCDatasets: Dataset
using KernelAbstractions: @kernel, @index, Event

import Oceananigans.Fields: interior
import Oceananigans.Utils: launch!, datatuple

using Oceananigans.Diagnostics: run_diagnostic
using Oceananigans.TimeSteppers: _compute_w_from_continuity!
using Oceananigans.AbstractOperations: Computation, compute!

Logging.global_logger(OceananigansLogger())

#####
##### Testing parameters
#####

float_types = (Float32, Float64)

         archs = (CPU(),)
@hascuda archs = (GPU(),)

closures = (
    :IsotropicDiffusivity,
    :AnisotropicDiffusivity,
    :AnisotropicBiharmonicDiffusivity,
    :TwoDimensionalLeith,
    :SmagorinskyLilly,
    :BlasiusSmagorinsky,
    :RozemaAnisotropicMinimumDissipation,
    :VerstappenAnisotropicMinimumDissipation
)

#####
##### Run tests!
#####

include("runtests_utils.jl")

group = get(ENV, "TEST_GROUP", :all) |> Symbol

@testset "Oceananigans" begin
    if group == :unit || group == :all
        @testset "Unit tests" begin
            include("test_grids.jl")
            include("test_operators.jl")
            include("test_boundary_conditions.jl")
            include("test_fields.jl")
            include("test_halo_regions.jl")
            include("test_solvers.jl")
            include("test_pressure_solvers.jl")
            include("test_coriolis.jl")
            include("test_buoyancy.jl")
            include("test_surface_waves.jl")
        end
    end

    if group == :integration || group == :all
        @testset "Integration tests" begin
            include("test_models.jl")
            include("test_simulations.jl")
            include("test_time_stepping.jl")
            include("test_time_stepping_bcs.jl")
            include("test_forcings.jl")
            include("test_turbulence_closures.jl")
            include("test_dynamics.jl")
            include("test_diagnostics.jl")
            include("test_averaged_field.jl")
            include("test_output_writers.jl")
            include("test_abstract_operations.jl")
        end
    end

    if group == :regression || group == :all
        include("test_regression.jl")
    end

    if group == :scripts || group == :all
        @testset "Scripts" begin
            include("test_examples.jl")
            include("test_verification.jl")
            include("test_benchmarks.jl")
        end
    end

    if group == :convergence
        include("test_convergence.jl")
    end
end
