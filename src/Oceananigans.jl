"""
Main module for `Oceananigans.jl` -- a Julia software for fast, friendly, flexible,
data-driven, ocean-flavored fluid dynamics on CPUs and GPUs.
"""
module Oceananigans

if VERSION < v"1.8"
    @warn "Oceananigans is tested on Julia v1.8 and therefore it is strongly recommended you run Oceananigans on Julia v1.8 or newer."
end

export
    # Architectures
    CPU, GPU, 

    # Logging
    OceananigansLogger,

    # Grids
    Center, Face,
    Periodic, Bounded, Flat, 
    FullyConnected, LeftConnected, RightConnected,
    RectilinearGrid, 
    LatitudeLongitudeGrid,
    OrthogonalSphericalShellGrid,
    xnodes, ynodes, znodes, nodes,
    xspacings, yspacings, zspacings,
    minimum_xspacing, minimum_yspacing, minimum_zspacing,

    # Immersed boundaries
    ImmersedBoundaryGrid, GridFittedBoundary, GridFittedBottom, ImmersedBoundaryCondition,

    # Advection schemes
    Centered, CenteredSecondOrder, CenteredFourthOrder, 
    UpwindBiased, UpwindBiasedFirstOrder, UpwindBiasedThirdOrder, UpwindBiasedFifthOrder, 
    WENO, WENOThirdOrder, WENOFifthOrder,
    VectorInvariant, EnergyConservingScheme, EnstrophyConservingScheme,

    # Boundary conditions
    BoundaryCondition,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition, OpenBoundaryCondition,
    FieldBoundaryConditions,

    # Fields and field manipulation
    Field, CenterField, XFaceField, YFaceField, ZFaceField,
    Average, Integral, Reduction, BackgroundField,
    interior, set!, compute!, regrid!, location,

    # Forcing functions
    Forcing, Relaxation, LinearTarget, GaussianMask, AdvectiveForcing,

    # Coriolis forces
    FPlane, ConstantCartesianCoriolis, BetaPlane, NonTraditionalBetaPlane,

    # BuoyancyModels and equations of state
    Buoyancy, BuoyancyTracer, SeawaterBuoyancy,
    LinearEquationOfState, TEOS10,
    BuoyancyField,

    # Surface wave Stokes drift via Craik-Leibovich equations
    UniformStokesDrift,

    # Turbulence closures
    VerticalScalarDiffusivity,
    HorizontalScalarDiffusivity,
    ScalarDiffusivity,
    VerticalScalarBiharmonicDiffusivity,
    HorizontalScalarBiharmonicDiffusivity,
    ScalarBiharmonicDiffusivity,
    SmagorinskyLilly,
    AnisotropicMinimumDissipation,
    ConvectiveAdjustmentVerticalDiffusivity,
    RiBasedVerticalDiffusivity,
    IsopycnalSkewSymmetricDiffusivity,
    FluxTapering,
    VerticallyImplicitTimeDiscretization,
    viscosity, diffusivity,

    # Lagrangian particle tracking
    LagrangianParticles,

    # Models
    NonhydrostaticModel,
    HydrostaticFreeSurfaceModel,
    ShallowWaterModel, ConservativeFormulation, VectorInvariantFormulation,
    PressureField,
    fields,

    # Hydrostatic free surface model stuff
    VectorInvariant, ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    HydrostaticSphericalCoriolis, 
    PrescribedVelocityFields,

    # Time stepping
    Clock, TimeStepWizard, time_step!,

    # Simulations
    Simulation, run!, Callback, iteration, stopwatch,
    iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded,
    erroring_NaNChecker!,
    TimeStepCallsite, TendencyCallsite, UpdateStateCallsite,

    # Diagnostics
    StateChecker, CFL, AdvectiveCFL, DiffusiveCFL,

    # Output writers
    NetCDFOutputWriter, JLD2OutputWriter, Checkpointer,
    TimeInterval, IterationInterval, AveragedTimeInterval, SpecifiedTimes,
    AndSchedule, OrSchedule,

    # Output readers
    FieldTimeSeries, FieldDataset, InMemory, OnDisk,

    # Abstract operations
    ∂x, ∂y, ∂z, @at, KernelFunctionOperation,

    # MultiRegion and Cubed sphere
    MultiRegionGrid, XPartition, 
    ConformalCubedSphereGrid,

    # Utils
    prettytime, apply_regionally!, construct_regionally, @apply_regionally, MultiRegionObject
    
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

struct TimeStepCallsite end
struct TendencyCallsite end
struct UpdateStateCallsite end

#####
##### Place-holder functions
#####

function run_diagnostic! end
function write_output! end
function location end
function instantiated_location end
function tupleit end
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

# Physics, time-stepping, and models
include("Coriolis/Coriolis.jl")
include("BuoyancyModels/BuoyancyModels.jl")
include("StokesDrift.jl")
include("TurbulenceClosures/TurbulenceClosures.jl")
include("Forcings/Forcings.jl")
include("Biogeochemistry.jl")

include("ImmersedBoundaries/ImmersedBoundaries.jl")
include("Distributed/Distributed.jl")
include("LagrangianParticleTracking/LagrangianParticleTracking.jl")
include("TimeSteppers/TimeSteppers.jl")
include("Models/Models.jl")

# Output and Physics, time-stepping, and models
include("Diagnostics/Diagnostics.jl")
include("OutputWriters/OutputWriters.jl")
include("OutputReaders/OutputReaders.jl")
include("Simulations/Simulations.jl")

# Abstractions for distributed and multi-region models
include("MultiRegion/MultiRegion.jl")
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
using .ImmersedBoundaries
using .Distributed
using .Models
using .TimeSteppers
using .Diagnostics
using .OutputWriters
using .OutputReaders
using .Simulations
using .AbstractOperations
using .MultiRegion
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
