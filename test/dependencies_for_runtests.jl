using Test
using Printf
using Random
using Statistics
using LinearAlgebra
using Logging

using CUDA
using MPI
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
using Oceananigans.Distributed
using Oceananigans.Logger
using Oceananigans.Units
using Oceananigans.Utils
using Oceananigans.MultiRegion
using Oceananigans.Architectures: device, array_type # to resolve conflict with CUDA.device

using Oceananigans: Clock
using Dates: DateTime, Nanosecond
using Statistics: mean
using LinearAlgebra: norm
using NCDatasets: Dataset
using KernelAbstractions: @kernel, @index

import Oceananigans.Fields: interior
import Oceananigans.Utils: launch!, datatuple

Logging.global_logger(OceananigansLogger())

#####
##### Testing parameters
#####

float_types = (Float32, Float64)

closures = (
    :ScalarDiffusivity,
    :ScalarBiharmonicDiffusivity,
    :TwoDimensionalLeith,
    :SmagorinskyLilly,
    :AnisotropicMinimumDissipation,
    :ConvectiveAdjustmentVerticalDiffusivity
)

#####
##### Run tests!
#####

float_types = (Float32, Float64)

include("utils_for_runtests.jl")

archs = test_architectures()

group     = get(ENV, "TEST_GROUP", :all) |> Symbol
test_file = get(ENV, "TEST_FILE", :none) |> Symbol
