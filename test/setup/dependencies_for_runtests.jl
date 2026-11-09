using Oceananigans
using Test
using Printf
using Random
using Statistics
using LinearAlgebra
using Logging
using SparseArrays
using JLD2
using FFTW
using OffsetArrays
using SeawaterPolynomials
using MPI
using Adapt
using GPUArraysCore
using CUDA

MPI.Initialized() || MPI.Init()

using Dates: DateTime, Nanosecond
using Statistics: mean, mean!, norm
using LinearAlgebra: norm
using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Advection
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.AbstractOperations
using Oceananigans.Coriolis
using Oceananigans.BuoyancyFormulations
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

using Oceananigans: Clock, location
using Oceananigans.Architectures: device, array_type # to resolve conflict with CUDA.device
using Oceananigans.Architectures: on_architecture
using Oceananigans.AbstractOperations: UnaryOperation, Derivative, BinaryOperation, MultiaryOperation
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Models: buoyancy_field
using Oceananigans.Grids: architecture
using Oceananigans.Fields: ZeroField, ConstantField, FunctionField, compute_at!, indices, instantiated_location
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Oceananigans.ImmersedBoundaries: conditional_length
using Oceananigans.Operators: ℑxyᶠᶜᵃ, hack_cosd
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.MultiRegion: reconstruct_global_grid, reconstruct_global_field
using Oceananigans.Utils: prettysummary

import Oceananigans.Utils: launch!, getnamewrapper
Logging.global_logger(OceananigansLogger())

# Legacy tests index GPU arrays with scalars; the process-wide default is the only setting
# that reaches every task ParallelTestRunner spawns. The warning discouraging this is silenced.
with_logger(NullLogger()) do
    GPUArraysCore.allowscalar(true)
end

#####
##### Testing parameters
#####

closures = (
    :ScalarDiffusivity,
    :ScalarBiharmonicDiffusivity,
    :TwoDimensionalLeith,
    :ConstantSmagorinsky,
    :SmagorinskyLilly,
    :LagrangianAveragedDynamicSmagorinsky,
    :DirectionallyAveragedDynamicSmagorinsky,
    :AnisotropicMinimumDissipation,
    :ConvectiveAdjustmentVerticalDiffusivity,
)

include(joinpath(@__DIR__, "utils_for_runtests.jl"))

float_types = (Float32, Float64)
archs = test_architectures()

# We need to Mock a grid since it provides an architecture for advection materialization
struct MockGrid{A <: AbstractArchitecture}
    arch::A
end
Oceananigans.Grids.architecture(a::MockGrid) = a.arch

# Tests write fixed-name output files; each test module gets its own directory so parallel
# workers never collide. Only ParallelTestRunner workers have the runner loaded in Main: MPI ranks
# (launched by srun or spawned by a test) must share the parent's directory, and so must a REPL.
isdefined(Main, :ParallelTestRunner) && !mpi_test && cd(mktempdir())
