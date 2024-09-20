using Oceananigans
using Test
using Printf
using Random
using Statistics
using LinearAlgebra
using Logging
using Enzyme
using SparseArrays
using JLD2
using FFTW
using OffsetArrays
using SeawaterPolynomials
using CUDA
using MPI

using Dates: DateTime, Nanosecond
using Statistics: mean, mean!, norm
using LinearAlgebra: norm
using NCDatasets: Dataset
using KernelAbstractions: @kernel, @index

MPI.versioninfo()
MPI.Initialized() || MPI.Init()

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Advection
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.AbstractOperations
using Oceananigans.Coriolis
using Oceananigans.BuoyancyModels
using Oceananigans.Forcings
using Oceananigans.Solvers
using Oceananigans.Models
using Oceananigans.MultiRegion
using Oceananigans.Simulations
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.TurbulenceClosures
using Oceananigans.DistributedComputations
using Oceananigans.Logger
using Oceananigans.Units
using Oceananigans.Utils

using Oceananigans: Clock
using Oceananigans.Architectures: device, array_type # to resolve conflict with CUDA.device
using Oceananigans.Architectures: on_architecture
using Oceananigans.AbstractOperations: UnaryOperation, Derivative, BinaryOperation, MultiaryOperation
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyModels: BuoyancyField
using Oceananigans.Grids: architecture
using Oceananigans.Fields: ZeroField, ConstantField, FunctionField, compute_at!, indices
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Oceananigans.ImmersedBoundaries: conditional_length
using Oceananigans.Operators: ℑxyᶜᶠᵃ, ℑxyᶠᶜᵃ, hack_cosd
using Oceananigans.Solvers: constructors, unpack_constructors
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.MultiRegion: reconstruct_global_grid, reconstruct_global_field, getnamewrapper

import Oceananigans.Utils: launch!, datatuple
Logging.global_logger(OceananigansLogger())

#####
##### Testing parameters
#####

closures = (
    :ScalarDiffusivity,
    :ScalarBiharmonicDiffusivity,
    :TwoDimensionalLeith,
    :SmagorinskyLilly,
    :AnisotropicMinimumDissipation,
    :ConvectiveAdjustmentVerticalDiffusivity,
)

if !(@isdefined already_included)
    already_included = Ref(false)
    macro include_once(expr)
        return !(already_included[]) ? :($(esc(expr))) : :(nothing)
    end
end

@include_once include("utils_for_runtests.jl")
already_included[] = true

float_types = (Float32, Float64)
archs = test_architectures()

