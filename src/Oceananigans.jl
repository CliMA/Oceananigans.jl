"""
Main module for `Oceananigans.jl` -- a Julia software for fast, friendly, flexible,
data-driven, ocean-flavored fluid dynamics on CPUs and GPUs.
"""
module Oceananigans

export
    # Architectures
    CPU, GPU,

    # Grids
    Center, Face,
    Periodic, Bounded, Flat, 
    RectilinearGrid, LatitudeLongitudeGrid, OrthogonalSphericalShellGrid, TripolarGrid,
    nodes, xnodes, ynodes, znodes, λnodes, φnodes,
    xspacings, yspacings, zspacings, λspacings, φspacings,
    minimum_xspacing, minimum_yspacing, minimum_zspacing,

    # Immersed boundaries
    ImmersedBoundaryGrid,
    GridFittedBoundary, GridFittedBottom, PartialCellBottom,
    ImmersedBoundaryCondition,

    # Distributed
    Distributed, Partition,

    # Advection schemes
    Centered, UpwindBiased, WENO,
    VectorInvariant, WENOVectorInvariant, FluxFormAdvection,

    # Boundary conditions
    BoundaryCondition,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition, OpenBoundaryCondition,
    FieldBoundaryConditions,

    # Fields and field manipulation
    Field, CenterField, XFaceField, YFaceField, ZFaceField,
    Average, Integral, CumulativeIntegral, Reduction, Accumulation, BackgroundField,
    interior, set!, compute!, regrid!,

    # Forcing functions
    Forcing, Relaxation, LinearTarget, GaussianMask, AdvectiveForcing,

    # Coriolis forces
    FPlane, ConstantCartesianCoriolis, BetaPlane, NonTraditionalBetaPlane,

    # BuoyancyModels and equations of state
    Buoyancy, BuoyancyTracer, SeawaterBuoyancy,
    LinearEquationOfState, TEOS10,
    BuoyancyField,

    # Surface wave Stokes drift via Craik-Leibovich equations
    UniformStokesDrift, StokesDrift,

    # Turbulence closures
    VerticalScalarDiffusivity,
    HorizontalScalarDiffusivity,
    ScalarDiffusivity,
    VerticalScalarBiharmonicDiffusivity,
    HorizontalScalarBiharmonicDiffusivity,
    ScalarBiharmonicDiffusivity,
    SmagorinskyLilly,
    Smagorinsky,
    LillyCoefficient,
    DynamicCoefficient,
    AnisotropicMinimumDissipation,
    ConvectiveAdjustmentVerticalDiffusivity,
    CATKEVerticalDiffusivity,
    RiBasedVerticalDiffusivity,
    VerticallyImplicitTimeDiscretization,
    viscosity, diffusivity,

    # Lagrangian particle tracking
    LagrangianParticles,

    # Models
    NonhydrostaticModel, HydrostaticFreeSurfaceModel, ShallowWaterModel,
    ConservativeFormulation, VectorInvariantFormulation,
    PressureField, fields,

    # Hydrostatic free surface model stuff
    VectorInvariant, ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    HydrostaticSphericalCoriolis, PrescribedVelocityFields,

    # Time stepping
    Clock, TimeStepWizard, conjure_time_step_wizard!, time_step!,

    # Simulations
    Simulation, run!, Callback, add_callback!, iteration,
    iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded,

    # Diagnostics
    CFL, AdvectiveCFL, DiffusiveCFL,

    # Output writers
    NetCDFOutputWriter, JLD2OutputWriter, Checkpointer,
    TimeInterval, IterationInterval, WallTimeInterval, AveragedTimeInterval,
    SpecifiedTimes, FileSizeLimit, AndSchedule, OrSchedule, written_names,

    # Output readers
    FieldTimeSeries, FieldDataset, InMemory, OnDisk,

    # Abstract operations
    ∂x, ∂y, ∂z, @at, KernelFunctionOperation,

    # Utils
    prettytime

using CUDA
using DocStringExtensions
using FFTW

function __init__()
    threads = Threads.nthreads()
    if threads > 1
        @info "Oceananigans will use $threads threads"

        # See: https://github.com/CliMA/Oceananigans.jl/issues/1113
        FFTW.set_num_threads(4threads)
    end

    if CUDA.has_cuda()
        @debug "CUDA-enabled GPU(s) detected:"
        for (gpu, dev) in enumerate(CUDA.devices())
            @debug "$dev: $(CUDA.name(dev))"
        end

        CUDA.allowscalar(false)
    end
end

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

# Callsites for Callbacks
struct TimeStepCallsite end
struct TendencyCallsite end
struct UpdateStateCallsite end

#####
##### Place-holder functions
#####

function run_diagnostic! end
function write_output! end
function initialize! end # for initializing models, simulations, etc
function location end
function instantiated_location end
function tupleit end
function fields end
function prognostic_fields end
function tracer_tendency_kernel_function end
function boundary_conditions end

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
include("ImmersedBoundaries/ImmersedBoundaries.jl")
include("OrthogonalSphericalShellGrids/OrthogonalSphericalShellGrids.jl")
include("TimeSteppers/TimeSteppers.jl")
include("Advection/Advection.jl")
include("Solvers/Solvers.jl")
include("OutputReaders/OutputReaders.jl")
include("DistributedComputations/DistributedComputations.jl")

# TODO: move here
#include("MultiRegion/MultiRegion.jl")

# Physics, time-stepping, and models
include("Coriolis/Coriolis.jl")
include("BuoyancyModels/BuoyancyModels.jl")
include("StokesDrifts.jl")
include("TurbulenceClosures/TurbulenceClosures.jl")
include("Forcings/Forcings.jl")
include("Biogeochemistry.jl")

# TODO: move above
include("Models/Models.jl")

# Output and Physics, time-stepping, and models
include("Diagnostics/Diagnostics.jl")
include("OutputWriters/OutputWriters.jl")
include("Simulations/Simulations.jl")

# Abstractions for distributed and multi-region models
include("MultiRegion/MultiRegion.jl")

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
using .OrthogonalSphericalShellGrids
using .Coriolis
using .BuoyancyModels
using .StokesDrifts
using .TurbulenceClosures
using .Solvers
using .OutputReaders
using .Forcings
using .ImmersedBoundaries
using .DistributedComputations
using .Models
using .TimeSteppers
using .Diagnostics
using .OutputWriters
using .Simulations
using .AbstractOperations
using .MultiRegion

end # module
