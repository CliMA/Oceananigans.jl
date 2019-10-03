module Oceananigans

if VERSION < v"1.1"
    @error "Oceananigans requires Julia v1.1 or newer."
end

export
    # Helper macro for determining if a CUDA-enabled GPU is available.
    @hascuda,

    # Architectures
    CPU, GPU,

    # Constants
    FPlane,
    second, minute, hour, day,

    # Grids
    RegularCartesianGrid,

    # Fields
    Field, CellField, FaceFieldX, FaceFieldY, FaceFieldZ,
    data, set!, set_ic!,
    nodes, xnodes, ynodes, znodes,

    # Forcing functions
    Forcing,

    # Equation of state
    BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState,

    # Boundary conditions
    BoundaryCondition,
    Periodic, Flux, Gradient, Value, Dirchlet, Neumann,
    CoordinateBoundaryConditions,
    FieldBoundaryConditions, HorizontallyPeriodicBCs, ChannelBCs,
    BoundaryConditions, SolutionBoundaryConditions, HorizontallyPeriodicSolutionBCs, ChannelSolutionBCs,
    getbc, setbc!,

    # Time stepping
    TimeStepWizard,
    update_Δt!, time_step!,

    # Clock
    Clock,

    # Models
    Model, BasicModel, ChannelModel, BasicChannelModel,

    # Model output writers
    Checkpointer, restore_from_checkpoint, read_output,
    JLD2OutputWriter, FieldOutput, FieldOutputs,
    write_grid, NetCDFOutputWriter,

    # Model diagnostics
    HorizontalAverage, NaNChecker,
    Timeseries, CFL, AdvectiveCFL, DiffusiveCFL, FieldMaximum,

    # Package utilities
    prettytime, pretty_filesize, KiB, MiB, GiB, TiB,

    # Turbulence closures
    ConstantIsotropicDiffusivity, ConstantAnisotropicDiffusivity,
    ConstantSmagorinsky, AnisotropicMinimumDissipation

# Standard library modules
using
    Statistics,
    LinearAlgebra,
    Printf

# Third-party modules
using
    Adapt,
    FFTW,
    Distributed,
    StaticArrays,
    OffsetArrays,
    JLD2,
    NCDatasets

import
    CUDAapi,
    GPUifyLoops

using Base: @propagate_inbounds
using Statistics: mean
using OrderedCollections: OrderedDict
using CUDAapi: has_cuda
using GPUifyLoops: @launch, @loop, @unroll

import Base:
    +, -, *,
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
abstract type AbstractModel end

"""
    AbstractArchitecture

Abstract supertype for architectures supported by Oceananigans.
"""
abstract type AbstractArchitecture end

"""
    AbstractRotation

Abstract supertype for parameters related to background rotation rates.
"""
abstract type AbstractRotation end

"""
    AbstractGrid{T}

Abstract supertype for grids with elements of type `T`.
"""
abstract type AbstractGrid{T} end

"""
    AbstractField{A, G}

Abstract supertype for fields stored on an architecture `A` and defined on a grid `G`.
"""
abstract type AbstractField{A, G} end

"""
    AbstractFaceField{A, G} <: AbstractField{A, G}

Abstract supertype for fields stored on an architecture `A` and defined the cell faces of a grid `G`.
"""
abstract type AbstractFaceField{A, G} <: AbstractField{A, G} end

"""
    AbstractEquationOfState

Abstract supertype for buoyancy models.
"""
abstract type AbstractBuoyancy{EOS} end

"""
    AbstractEquationOfState

Abstract supertype for equations of state.
"""
abstract type AbstractEquationOfState end

"""
    AbstractEquationOfState

Abstract supertype for nonlinar equations of state.
"""
abstract type AbstractNonlinearEquationOfState <: AbstractEquationOfState end

"""
    AbstractEquationOfState

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

device(::CPU) = GPUifyLoops.CPU()
device(::GPU) = GPUifyLoops.CUDA()

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

architecture(::Array) = CPU()
@hascuda architecture(::CuArray) = GPU()

# Place-holder buoyancy functions for use in TurbulenceClosures module
function buoyancy_perturbation end
function buoyancy_frequency_squared end

include("utils.jl")

include("clock.jl")
include("grids.jl")
include("fields.jl")

include("Operators/Operators.jl")
include("TurbulenceClosures/TurbulenceClosures.jl")

include("coriolis.jl")
include("buoyancy.jl")
include("boundary_conditions.jl")
include("halo_regions.jl")
include("poisson_solvers.jl")
include("models.jl")
include("time_steppers.jl")

include("output_writers.jl")
include("diagnostics.jl")

end # module
