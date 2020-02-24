module Oceananigans

if VERSION < v"1.1"
    @error "Oceananigans requires Julia v1.1 or newer."
end

export
    # Architectures
    CPU, GPU,

    # Logging
    ModelLogger, Diagnostic, Setup,

    # Grids
    Periodic, Bounded, Flat,
    RegularCartesianGrid, VerticallyStretchedCartesianGrid,

    # Boundary conditions
    BoundaryCondition,
    Flux, Value, Gradient,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition,
    CoordinateBoundaryConditions, FieldBoundaryConditions,
    UVelocityBoundaryConditions, VVelocityBoundaryConditions, WVelocityBoundaryConditions,
    TracerBoundaryConditions, PressureBoundaryConditions,
    BoundaryFunction,

    # Fields and field manipulation
    Field, CellField, XFaceField, YFaceField, ZFaceField,
    interior, set!,

    # Forcing functions
    ModelForcing, SimpleForcing,

    # Coriolis forces
    FPlane, BetaPlane, NonTraditionalFPlane,

    # Buoyancy and equations of state
    BuoyancyTracer, SeawaterBuoyancy,
    LinearEquationOfState, RoquetIdealizedNonlinearEquationOfState,

    # Surface waves via Craik-Leibovich equations
    SurfaceWaves,

    # Time stepping
    time_step!,
    TimeStepWizard, update_Δt!,

    # Models
    IncompressibleModel, NonDimensionalModel,

    # Simulations
    Simulation, run!,
    iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded,

    # Utilities
    prettytime, pretty_filesize,

    # Turbulence closures
    ConstantIsotropicDiffusivity, ConstantAnisotropicDiffusivity,
    AnisotropicBiharmonicDiffusivity,
    ConstantSmagorinsky, AnisotropicMinimumDissipation

# Standard library modules
using Printf
using Logging
using Statistics
using LinearAlgebra

# Third-party modules
using Adapt
using OffsetArrays
using FFTW
using JLD2
using NCDatasets

import CUDAapi
import GPUifyLoops

using Base: @propagate_inbounds
using Statistics: mean
using GPUifyLoops: @launch, @loop, @unroll

import Base:
    +, -, *, /,
    size, length, eltype,
    iterate, similar, show,
    getindex, lastindex, setindex!,
    push!

#####
##### Abstract types
#####

"""
    Cell

A type describing the location at the center of a grid cell.
"""
struct Cell end

"""
	Face

A type describing the location at the face of a grid cell.
"""
struct Face end

"""
    AbstractDiagnostic

Abstract supertype for diagnostics that compute information from the current
model state.
"""
abstract type AbstractDiagnostic end

"""
    AbstractOutputWriter

Abstract supertype for output writers that write data to disk.
"""
abstract type AbstractOutputWriter end

#####
##### Place-holder functions
#####

function TimeStepper end
function run_diagnostic end
function write_output end

#####
##### Include all the submodules
#####

include("Architectures.jl")

using Oceananigans.Architectures: @hascuda
@hascuda begin
    # Import CUDA utilities if it's detected.
    using CUDAdrv
    using CUDAnative
    using CuArrays

    println("CUDA-enabled GPU(s) detected:")
    for (gpu, dev) in enumerate(CUDAnative.devices())
        println(dev)
    end
end

include("Utils/Utils.jl")
include("Logger.jl")
include("Grids/Grids.jl")
include("Operators/Operators.jl")
include("BoundaryConditions/BoundaryConditions.jl")
include("Fields/Fields.jl")
include("Coriolis/Coriolis.jl")
include("Buoyancy/Buoyancy.jl")
include("SurfaceWaves.jl")
include("TurbulenceClosures/TurbulenceClosures.jl")
include("Solvers/Solvers.jl")
include("Forcing/Forcing.jl")
include("Models/Models.jl")
include("TimeSteppers/TimeSteppers.jl")
include("Diagnostics/Diagnostics.jl")
include("OutputWriters/OutputWriters.jl")
include("Simulations.jl")
include("AbstractOperations/AbstractOperations.jl")

#####
##### Re-export stuff from submodules
#####

using .Architectures
using .Utils
using .Grids
using .BoundaryConditions
using .Fields
using .Coriolis
using .Buoyancy
using .SurfaceWaves
using .TurbulenceClosures
using .Solvers
using .Forcing
using .Models
using .TimeSteppers
using .Simulations

end # module
