module Oceananigans

if VERSION < v"1.5"
    error("This version of Oceananigans.jl requires Julia v1.5 or newer.")
end

export
    # Architectures
    CPU, GPU,

    # Logging
    OceananigansLogger,

    # Grids
    Periodic, Bounded, Flat,
    RegularRectilinearGrid, VerticallyStretchedRectilinearGrid,

    # Advection schemes
    CenteredSecondOrder, CenteredFourthOrder, UpwindBiasedThirdOrder, UpwindBiasedFifthOrder, WENO5,

    # Boundary conditions
    BoundaryCondition,
    Flux, Value, Gradient, NormalFlow,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition,
    CoordinateBoundaryConditions, FieldBoundaryConditions,
    UVelocityBoundaryConditions, VVelocityBoundaryConditions, WVelocityBoundaryConditions,
    TracerBoundaryConditions, PressureBoundaryConditions,

    # Fields and field manipulation
    Field, CenterField, XFaceField, YFaceField, ZFaceField,
    BackgroundField, interior, set!,

    # Forcing functions
    Forcing, Relaxation, LinearTarget, GaussianMask,

    # Coriolis forces
    FPlane, BetaPlane, NonTraditionalFPlane, NonTraditionalBetaPlane,

    # Buoyancy and equations of state
    BuoyancyTracer, SeawaterBuoyancy,
    LinearEquationOfState, RoquetIdealizedNonlinearEquationOfState, TEOS10,

    # Surface waves via Craik-Leibovich equations
    StokesDrift,

    # Turbulence closures
    IsotropicDiffusivity, AnisotropicDiffusivity,
    AnisotropicBiharmonicDiffusivity,
    ConstantSmagorinsky, AnisotropicMinimumDissipation,

    # Lagrangian particle tracking
    LagrangianParticles,

    # Models
    IncompressibleModel, NonDimensionalModel, HydrostaticFreeSurfaceModel, Clock,

    # Time stepping
    time_step!, TimeStepWizard,

    # Simulations
    Simulation, run!,
    iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded,

    # Output writers
    FieldSlicer, NetCDFOutputWriter, JLD2OutputWriter, Checkpointer, restore_from_checkpoint,

    # Misc.
    fields

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
    AbstractModel

Abstract supertype for models.
"""
abstract type AbstractModel{TS} end

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

function run_diagnostic! end
function write_output! end
function location end
function tupleit end
function short_show end

function fields end

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
include("StokesDrift.jl")
include("TurbulenceClosures/TurbulenceClosures.jl")
include("LagrangianParticleTracking/LagrangianParticleTracking.jl")
include("Solvers/Solvers.jl")
include("Forcings/Forcings.jl")
include("TimeSteppers/TimeSteppers.jl")
include("Models/Models.jl")
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
using .StokesDrift
using .TurbulenceClosures
using .LagrangianParticleTracking
using .Solvers
using .Forcings
using .Models
using .TimeSteppers
using .Simulations

function __init__()
    threads = Threads.nthreads()
    if threads > 1
        @info "Oceananigans will use $threads threads"

        # See: https://github.com/CliMA/Oceananigans.jl/issues/1113
        FFTW.set_num_threads(4*threads)
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
