module Oceananigans

if VERSION < v"1.1"
    @error "Oceananigans requires Julia v1.1 or newer."
end

export
    # Macros
    @hascuda,

    # Architectures
    CPU, GPU,

    # Constants
    PlanetaryConstants, Earth,

    # Grids
    RegularCartesianGrid,

    # Fields
    Field, CellField, FaceFieldX, FaceFieldY, FaceFieldZ,
    data, set!, set_ic!,
    nodes, xnodes, ynodes, znodes,

    # Equation of state
    NoEquationOfState, LinearEquationOfState,

    # Boundary conditions
    Periodic, Flux, Gradient, Value, NoPenetration, Dirchlet, Neumann,
    BoundaryCondition, CoordinateBoundaryConditions,
    FieldBoundaryConditions, HorizontallyPeriodicBCs, ChannelBCs,
    SolutionBoundaryConditions, HorizontallyPeriodicSolutionBCs, ChannelSolutionBCs, BoundaryConditions,
    getbc, setbc!,

    # Time stepping
    time_step!,
    TimeStepWizard, update_Δt!,

    # Clock
    Clock,

    # Forcing functions
    Forcing,

    # Models
    Model, BasicModel, ChannelModel, BasicChannelModel,

    # Model output writers
    NetCDFOutputWriter,
    Checkpointer, restore_from_checkpoint, read_output,
    JLD2OutputWriter, FieldOutput, FieldOutputs,

    # Model diagnostics
    HorizontalAverage, NaNChecker,
    Timeseries, CFL, AdvectiveCFL, DiffusiveCFL, FieldMaximum,

    # Package utilities
    prettytime, pretty_filesize,
    second, minute, hour, day,
    KiB, MiB, GiB, TiB,

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
    NetCDF

import
    CUDAapi,
    GPUifyLoops

using Base: @propagate_inbounds
using Statistics: mean
using CUDAapi: has_cuda
using GPUifyLoops: @launch, @loop, @unroll

import Base:
    +, -, *,
    size, length, eltype, zeros,
    iterate, similar, show,
    getindex, lastindex, setindex!

####
#### Abstract types
#### Define them all here so we can use start using them early, e.g. in utils.jl. This
#### would not be possible if we defined them in the appropriate files, e.g. AbstractGrid
#### in grids.jl.
####

"""
    AbstractArchitecture

Abstract supertype for architectures supported by Oceananigans.
"""
abstract type AbstractArchitecture end

"""
    AbstractEquationOfState

Abstract supertype for equations of state.
"""
abstract type AbstractEquationOfState end

"""
    AbstractGrid{T}

Abstract supertype for grids with elements of type `T`.
"""
abstract type AbstractGrid{T} end
abstract type AbstractField{A, G} end
abstract type AbstractFaceField{A, G} <: AbstractField{A, G} end
abstract type AbstractPoissonSolver end

abstract type AbstractModel end

"""
    AbstractDiagnostic

Abstract supertype for types that compute diagnostic information from the current model
state.
"""
abstract type AbstractDiagnostic end

"""
    AbstractTimeseriesDiagnostic

Abstract supertype for types that compute timeseries of diagnostic information from the
current model state.
"""
abstract type AbstractTimeseriesDiagnostic <: AbstractDiagnostic end

"""
    AbstractOutputWriter

Abstract supertype for types that perform input and output.
"""
abstract type AbstractOutputWriter end

#####
##### All the code
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

# Convert between Oceananigans architectures and GPUifyLoops.jl devices.
device(::CPU) = GPUifyLoops.CPU()
device(::GPU) = GPUifyLoops.CUDA()

"""
    @hascuda

A macro to execute an expression only if CUDA is installed and available. Generally used to
wrap expressions that can only execute with a GPU.
"""
macro hascuda(ex)
    return has_cuda() ? :($(esc(ex))) : :(nothing)
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

# Placeholder definition needed by TurbulenceClosures submodule.
function buoyancy_perturbation end

include("utils.jl")

include("clock.jl")
include("planetary_constants.jl")
include("grids.jl")
include("fields.jl")

include("operators/operators.jl")
include("turbulence_closures/TurbulenceClosures.jl")

include("boundary_conditions.jl")
include("halo_regions.jl")
include("equation_of_state.jl")
include("poisson_solvers.jl")
include("models.jl")
include("time_steppers.jl")

include("output_writers.jl")
include("diagnostics.jl")

end # module
