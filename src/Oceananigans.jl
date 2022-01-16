"""
Main module for `Oceananigans.jl` -- a Julia software for fast, friendly, flexible,
data-driven, ocean-flavored fluid dynamics on CPUs and GPUs.
"""
module Oceananigans

if VERSION < v"1.6"
    error("This version of Oceananigans.jl requires Julia v1.6 or newer.")
end

export
    # Architectures
    CPU, GPU,

    # Logging
    OceananigansLogger,

    # Grids
    Center, Face,
    Periodic, Bounded, Flat,
    RectilinearGrid, 
    LatitudeLongitudeGrid,
    ConformalCubedSphereFaceGrid,
    xnodes, ynodes, znodes, nodes,

    # Advection schemes
    CenteredSecondOrder, CenteredFourthOrder, UpwindBiasedFirstOrder, UpwindBiasedThirdOrder, UpwindBiasedFifthOrder, WENO5,

    # Boundary conditions
    BoundaryCondition,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition, OpenBoundaryCondition,
    FieldBoundaryConditions,

    # Fields and field manipulation
    Field, CenterField, XFaceField, YFaceField, ZFaceField,
    Average, Integral, Reduction, BackgroundField,
    interior, set!, compute!, regrid!,

    # Forcing functions
    Forcing, Relaxation, LinearTarget, GaussianMask,

    # Coriolis forces
    FPlane, ConstantCartesianCoriolis, BetaPlane, NonTraditionalBetaPlane,

    # BuoyancyModels and equations of state
    Buoyancy, BuoyancyTracer, SeawaterBuoyancy,
    LinearEquationOfState, TEOS10,
    BuoyancyField,

    # Surface wave Stokes drift via Craik-Leibovich equations
    UniformStokesDrift,

    # Turbulence closures
    IsotropicDiffusivity,
    AnisotropicDiffusivity,
    AnisotropicBiharmonicDiffusivity,
    SmagorinskyLilly,
    AnisotropicMinimumDissipation,
    HorizontallyCurvilinearAnisotropicDiffusivity,
    ConvectiveAdjustmentVerticalDiffusivity,
    IsopycnalSkewSymmetricDiffusivity,

    # Lagrangian particle tracking
    LagrangianParticles,

    # Models
    NonhydrostaticModel,
    HydrostaticFreeSurfaceModel,
    ShallowWaterModel,
    PressureField,
    fields,

    # Hydrostatic free surface model stuff
    VectorInvariant, ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    HydrostaticSphericalCoriolis, VectorInvariantEnstrophyConserving, VectorInvariantEnergyConserving,
    PrescribedVelocityFields,

    # Time stepping
    Clock, TimeStepWizard, time_step!,

    # Simulations
    Simulation, run!, Callback, iteration, stopwatch,
    iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded,
    erroring_NaNChecker!,

    # Diagnostics
    StateChecker, CFL, AdvectiveCFL, DiffusiveCFL,

    # Output writers
    FieldSlicer, NetCDFOutputWriter, JLD2OutputWriter, Checkpointer,
    TimeInterval, IterationInterval, AveragedTimeInterval, SpecifiedTimes,
    AndSchedule, OrSchedule,

    # Output readers
    FieldTimeSeries, FieldDataset, InMemory, OnDisk,

    # Abstract operations
    ∂x, ∂y, ∂z, @at, KernelFunctionOperation,

    # Cubed sphere
    ConformalCubedSphereGrid,

    # Utils
    prettytime


using Printf
using Logging
using Statistics
using LinearAlgebra

using CUDA
using Adapt
using DocStringExtensions
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
function instantiated_location end
function tupleit end
function short_show end

function fields end
function prognostic_fields end
function tracer_tendency_kernel_function end

#####
##### Include all the submodules
#####

# Basics
include("Architectures.jl")
include("Units.jl")
include("Grids/Grids.jl")
include("Utils/Utils.jl")
include("Logger.jl")
include("Operators/Operators.jl")
include("BoundaryConditions/BoundaryConditions.jl")
include("Fields/Fields.jl")
include("AbstractOperations/AbstractOperations.jl")
include("Advection/Advection.jl")
include("Solvers/Solvers.jl")
include("Distributed/Distributed.jl")

# Physics, time-stepping, and models
include("Coriolis/Coriolis.jl")
include("BuoyancyModels/BuoyancyModels.jl")
include("StokesDrift.jl")
include("TurbulenceClosures/TurbulenceClosures.jl")
include("LagrangianParticleTracking/LagrangianParticleTracking.jl")
include("Forcings/Forcings.jl")

include("ImmersedBoundaries/ImmersedBoundaries.jl")
include("TimeSteppers/TimeSteppers.jl")
include("Models/Models.jl")

# Output and Physics, time-stepping, and models
include("Diagnostics/Diagnostics.jl")
include("OutputWriters/OutputWriters.jl")
include("OutputReaders/OutputReaders.jl")
include("Simulations/Simulations.jl")

# Abstractions for distributed and multi-region models
include("CubedSpheres/CubedSpheres.jl")

#####
##### Needed so we can export names from sub-modules at the top-level
#####

using .Logger
using .Architectures
using .Utils
using .Advection
using .Grids
using .BoundaryConditions
using .Fields
using .Coriolis
using .BuoyancyModels
using .StokesDrift
using .TurbulenceClosures
using .LagrangianParticleTracking
using .Solvers
using .Forcings
using .Distributed
using .Models
using .TimeSteppers
using .Diagnostics
using .OutputWriters
using .OutputReaders
using .Simulations
using .AbstractOperations
using .CubedSpheres

function __init__()
    threads = Threads.nthreads()
    if threads > 1
        @info "Oceananigans will use $threads threads"

        # See: https://github.com/CliMA/Oceananigans.jl/issues/1113
        FFTW.set_num_threads(4*threads)
    end

    if CUDA.has_cuda()
        @debug "CUDA-enabled GPU(s) detected:"
        for (gpu, dev) in enumerate(CUDA.devices())
            @debug "$dev: $(CUDA.name(dev))"
        end

        CUDA.allowscalar(false)
    end
end

end # module
