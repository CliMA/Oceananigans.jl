module Oceananigans

if VERSION < v"1.1"
    @error "Oceananigans requires Julia v1.1 or newer."
end

export
    # Helper macro for determining if a CUDA-enabled GPU is available.
    @hascuda,

    # Architectures
    CPU, GPU,

    # Logging
    ModelLogger, Diagnostic, Setup, Simulation,

    # Grids
    RegularCartesianGrid,

    # Fields and field manipulation
    Field, CellField, FaceFieldX, FaceFieldY, FaceFieldZ,
    interior, set!,
    nodes, xnodes, ynodes, znodes,
    compute_w_from_continuity!,

    # Forcing functions
    ModelForcing, SimpleForcing,

    # Coriolis forces
    FPlane, BetaPlane,

    # Buoyancy and equations of state
    BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState, RoquetIdealizedNonlinearEquationOfState,

    # Surface waves via Craik-Leibovich equations
    SurfaceWaves,

    # Boundary conditions
    BoundaryCondition,
    Periodic, Flux, Gradient, Value, Dirchlet, Neumann,
    CoordinateBoundaryConditions, FieldBoundaryConditions, HorizontallyPeriodicBCs, ChannelBCs,
    BoundaryConditions, SolutionBoundaryConditions, HorizontallyPeriodicSolutionBCs, ChannelSolutionBCs,
    BoundaryFunction, getbc, setbc!,

    # Time stepping
    TimeStepWizard,
    update_Δt!, time_step!,

    # Clock
    Clock,

    # Models
    Model, ChannelModel, NonDimensionalModel,

    # Package utilities
    prettytime, pretty_filesize,

    # Turbulence closures
    ConstantIsotropicDiffusivity, ConstantAnisotropicDiffusivity,
    AnisotropicBiharmonicDiffusivity,
    ConstantSmagorinsky, AnisotropicMinimumDissipation

# Standard library modules
using
    Printf,
    Logging,
    Statistics,
    LinearAlgebra

# Third-party modules
using
    Adapt,
    OffsetArrays,
    FFTW,
    JLD2,
    NCDatasets

import
    CUDAapi,
    GPUifyLoops

using Base: @propagate_inbounds
using Statistics: mean
using CUDAapi: has_cuda
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
    AbstractArchitecture

Abstract supertype for architectures supported by Oceananigans.
"""
abstract type AbstractArchitecture end

"""
    AbstractGrid{T}

Abstract supertype for grids with elements of type `T`.
"""
abstract type AbstractGrid{T} end

"""
    AbstractPoissonSolver

Abstract supertype for solvers for Poisson's equation.
"""
abstract type AbstractPoissonSolver end

"""
    AbstractDiagnostic

Abstract supertype for types that compute diagnostic information from the current model
state.
"""
abstract type AbstractDiagnostic end

"""
    AbstractOutputWriter

Abstract supertype for types that perform input and output.
"""
abstract type AbstractOutputWriter end

#####
##### All the functionality
#####

"""
    CPU <: AbstractArchitecture

Run Oceananigans on a single-core of a CPU.
"""
struct CPU <: AbstractArchitecture end

"""
    GPU <: AbstractArchitecture

Run Oceananigans on a single NVIDIA CUDA GPU.
"""
struct GPU <: AbstractArchitecture end

"""
    @hascuda expr

A macro to compile and execute `expr` only if CUDA is installed and available. Generally used to
wrap expressions that can only be compiled if `CuArrays` and `CUDAnative` can be loaded.
"""
macro hascuda(expr)
    return has_cuda() ? :($(esc(expr))) : :(nothing)
end

@hascuda begin
    # Import CUDA utilities if it's detected.
    using CUDAdrv, CUDAnative, CuArrays

    println("CUDA-enabled GPU(s) detected:")
    for (gpu, dev) in enumerate(CUDAnative.devices())
        println(dev)
    end
end

device(::CPU) = GPUifyLoops.CPU()
device(::GPU) = GPUifyLoops.CUDA()

architecture(::Array) = CPU()
@hascuda architecture(::CuArray) = GPU()

# Place-holder functions
function buoyancy_perturbation end
function buoyancy_frequency_squared end
function ∂x_b end
function ∂y_b end
function ∂z_b end
function TracerFields end
function TimeStepper end
function run_diagnostic end
function write_output end

include("Utils/Utils.jl")

include("Grids/Grids.jl")

using .Grids

include("Fields/Fields.jl")
include("Operators/Operators.jl")
include("TurbulenceClosures/TurbulenceClosures.jl")

using .TurbulenceClosures

include("Coriolis/Coriolis.jl")
include("Buoyancy/Buoyancy.jl")

using .Buoyancy

include("SurfaceWaves.jl")
include("BoundaryConditions/BoundaryConditions.jl")

using .BoundaryConditions

include("Solvers/Solvers.jl")

using .Solvers

include("Forcing/Forcing.jl")
include("logger.jl")
include("Models/Models.jl")

using .Models

include("Diagnostics/Diagnostics.jl")
include("OutputWriters/OutputWriters.jl")

include("TimeSteppers/TimeSteppers.jl")

using .TimeSteppers

include("AbstractOperations/AbstractOperations.jl")

using .SurfaceWaves

end # module
