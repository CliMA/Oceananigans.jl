module Oceananigans

if VERSION < v"1.4"
    error("This version of Oceananigans.jl requires Julia v1.4 or newer.")
end

export
    # Architectures
    CPU, GPU,

    # Logging
    OceananigansLogger,

    # Grids
    Periodic, Bounded, Flat,
    RegularCartesianGrid, VerticallyStretchedCartesianGrid,

    # Boundary conditions
    BoundaryCondition,
    Flux, Value, Gradient, NormalFlow,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition,
    CoordinateBoundaryConditions, FieldBoundaryConditions,
    UVelocityBoundaryConditions, VVelocityBoundaryConditions, WVelocityBoundaryConditions,
    TracerBoundaryConditions, PressureBoundaryConditions,
    BoundaryFunction, ParameterizedBoundaryCondition,

    # Fields and field manipulation
    Field, CellField, XFaceField, YFaceField, ZFaceField,
    interior, set!,

    # Forcing functions
    ModelForcing, SimpleForcing, ParameterizedForcing,

    # Coriolis forces
    FPlane, BetaPlane, NonTraditionalFPlane, NonTraditionalBetaPlane,

    # Buoyancy and equations of state
    BuoyancyTracer, SeawaterBuoyancy,
    LinearEquationOfState, RoquetIdealizedNonlinearEquationOfState, TEOS10,

    # Surface waves via Craik-Leibovich equations
    SurfaceWaves,

    # Time stepping
    time_step!,
    TimeStepWizard, update_Δt!,

    # Models
    IncompressibleModel, NonDimensionalModel, Clock,

    # Simulations
    Simulation, run!,
    iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded,

    # Utilities
    prettytime, pretty_filesize,

    # Turbulence closures
    IsotropicDiffusivity, AnisotropicDiffusivity,
    AnisotropicBiharmonicDiffusivity,
    ConstantSmagorinsky, AnisotropicMinimumDissipation

# Standard library modules
using Printf
using Logging
using Statistics
using LinearAlgebra

# Third-party modules
using CUDA
using Adapt
using OffsetArrays
using FFTW
using JLD2
using NCDatasets

using Base: @propagate_inbounds
using Statistics: mean

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
include("Grids/Grids.jl")
include("Utils/Utils.jl")
include("Logger.jl")
include("Operators/Operators.jl")
include("Advection/Advection.jl")
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
include("Simulations/Simulations.jl")
include("AbstractOperations/AbstractOperations.jl")

#####
##### Re-export stuff from submodules
#####

using .Logger
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

function __init__()
    threads = Threads.nthreads()
    if threads > 1
        @info "Oceananigans will use $threads threads"
        FFTW.set_num_threads(threads)
    end

    @hascuda begin
        @debug "CUDA-enabled GPU(s) detected:"
        for (gpu, dev) in enumerate(CUDA.devices())
            @debug "$dev: $(CUDA.name(dev))"
        end

        CUDA.allowscalar(false)
    end
end

end # module
