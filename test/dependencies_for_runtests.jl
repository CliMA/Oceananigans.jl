using Test
using Printf
using Random
using Statistics
using LinearAlgebra
using Logging

using CUDA
using MPI

MPI.versioninfo()
MPI.Initialized() || MPI.Init()

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
using Oceananigans.AbstractOperations
using Oceananigans.Coriolis
using Oceananigans.BuoyancyModels
using Oceananigans.Forcings
using Oceananigans.Solvers
using Oceananigans.Models
using Oceananigans.Simulations
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.TurbulenceClosures
using Oceananigans.DistributedComputations
using Oceananigans.Logger
using Oceananigans.Units
using Oceananigans.Utils
using Oceananigans.MultiRegion

using Oceananigans: Clock
using Oceananigans.Architectures: device, array_type # to resolve conflict with CUDA.device
using Oceananigans.Architectures: on_architecture
using Oceananigans.AbstractOperations: UnaryOperation, Derivative, BinaryOperation, MultiaryOperation
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyModels: BuoyancyField
using Oceananigans.Fields: ZeroField, ConstantField, compute_at!, indices
using Oceananigans.Operators: ℑxyᶜᶠᵃ, ℑxyᶠᶜᵃ
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, conditional_length

using Dates: DateTime, Nanosecond
using Statistics: mean, mean!, norm
using LinearAlgebra: norm
using NCDatasets: Dataset
using KernelAbstractions: @kernel, @index

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

include("utils_for_runtests.jl")

float_types = (Float32, Float64)
archs = test_architectures()

