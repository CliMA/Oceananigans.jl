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
    PlanetaryConstants, Earth, Europa, Enceladus,
    second, minute, hour, day,

    # Grids
    RegularCartesianGrid,

    # Fields
    CellField, FaceFieldX, FaceFieldY, FaceFieldZ,
    data, set!, set_ic!,
    nodes, xnodes, ynodes, znodes,

    # Forcing functions
    Forcing,

    # Equation of state
    NoEquationOfState, LinearEquationOfState,

    # Boundary conditions
    BoundaryCondition,
    Periodic, Flux, Gradient, Value, Dirchlet, Neumann,
    CoordinateBoundaryConditions,
    FieldBoundaryConditions, HorizontallyPeriodicBCs, ChannelBCs,
    BoundaryConditions, ModelBoundaryConditions, HorizontallyPeriodicModelBCs, ChannelModelBCs,
    getbc, setbc!,

    # Time stepping
    TimeStepWizard,
    update_Δt!, time_step!,

    # Clock
    Clock,

    # Models
    Model, BasicModel, ChannelModel, BasicChannelModel,

    # Model output writers
    NetCDFOutputWriter, JLD2OutputWriter, Checkpointer,
    restore_from_checkpoint, read_output,

    # Model diagnostics
    HorizontalAverage, NaNChecker,

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
    NetCDF

import
    CUDAapi,
    GPUifyLoops

using CUDAapi: has_cuda
using GPUifyLoops: @launch, @loop, @unroll

import Base:
    +, -, *,
    size, length, eltype,
    iterate, similar, show,
    getindex, lastindex, setindex!

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

abstract type AbstractArchitecture end
struct CPU <: AbstractArchitecture end
struct GPU <: AbstractArchitecture end

device(::CPU) = GPUifyLoops.CPU()
device(::GPU) = GPUifyLoops.CUDA()

abstract type AbstractEquationOfState end
abstract type AbstractGrid{T} end
abstract type AbstractModel end
abstract type AbstractField{A, G} end
abstract type AbstractFaceField{A, G} <: AbstractField{A, G} end
abstract type AbstractOutputWriter end
abstract type AbstractDiagnostic end
abstract type AbstractPoissonSolver end

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
