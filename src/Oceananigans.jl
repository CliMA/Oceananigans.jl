module Oceananigans

if VERSION < v"1.1"
    @error "Oceananigans requires Julia v1.1 or newer."
end

export
    # Helper variables and macros for determining if machine is CUDA-enabled.
    HAVE_CUDA,
    @hascuda,

    Architecture,
    CPU,
    GPU,
    device,

    # Planetary Constants
    ConstantsCollection,
    PlanetaryConstants,
    Earth,
    EarthStationary,
    Europa,
    Enceladus,
    minute,
    hour,
    day,

    # Grids
    Grid,
    RegularCartesianGrid,

    # Fields
    Field,
    FaceField,
    CellField,
    FaceFieldX,
    FaceFieldY,
    FaceFieldZ,
    EdgeField,
    data,
    ardata,
    ardata_view,
    underlying_data,
    set!,
    xnodes,
    ynodes,
    znodes,
    nodes,
    set_ic!,

    # FieldSets (collections of related fields)
    FieldSet,
    VelocityFields,
    TracerFields,
    PressureFields,
    SourceTerms,

    # Forcing functions
    Forcing,

    # Equation of state
    NoEquationOfState,
    LinearEquationOfState,
    δρ,
    buoyancy,

    # Boundary conditions
    BoundaryCondition,
    Periodic,
    Flux,
    Gradient,
    Value,
    CoordinateBoundaryConditions,
    ZBoundaryConditions,
    FieldBoundaryConditions,
    ModelBoundaryConditions,
    BoundaryConditions,
    HorizontallyPeriodicBCs,
    ChannelBCs,
    HorizontallyPeriodicModelBCs,
    ChannelModelBCs,
    getbc,
    setbc!,

    # Halo regions
    fill_halo_regions!,

    # Time stepping
    TimeStepWizard,
    cell_advection_timescale,
    update_Δt!,
    time_step!,

    # Poisson solver
    PoissonBCs,
    PPN, PNN,
    PoissonSolver,
    PoissonSolverCPU,
    PoissonSolverGPU,
    solve_poisson_3d!,
    solve_poisson_3d_ppn_gpu_planned!,

    # Clock
    Clock,

    # Models
    Model,
    ChannelModel,

    # Model output writers
    OutputWriter,
    BinaryOutputWriter,
    NetCDFOutputWriter,
    JLD2OutputWriter,
    write_output,
    read_output,

    # Model diagnostics
    Diagnostic,
    run_diagnostic,
    FieldSummary,
    NaNChecker,
    VelocityDivergenceChecker,

    # Package utilities
    prettytime,

    # Turbulence closures
    TurbulenceClosures,
    ConstantIsotropicDiffusivity,
    ConstantAnisotropicDiffusivity,
    ConstantSmagorinsky,
    AnisotropicMinimumDissipation

# Standard library modules
using
    Statistics,
    LinearAlgebra,
    Printf,
    FileIO

# Third-party modules
using
    Adapt,
    FFTW,
    StaticArrays,
    OffsetArrays,
    JLD2,
    NetCDF

import
    GPUifyLoops

import Base:
    size, length,
    getindex, lastindex, setindex!,
    iterate, similar, *, +, -

# Import CUDA utilities if cuda is detected.
const HAVE_CUDA = try
    using CUDAdrv, CUDAnative, CuArrays
    true
catch
    false
end

import GPUifyLoops: @launch, @loop, @unroll

macro hascuda(ex)
    return HAVE_CUDA ? :($(esc(ex))) : :(nothing)
end

@hascuda begin
    println("CUDA-enabled GPU(s) detected:")
    for (gpu, dev) in enumerate(CUDAnative.devices())
        println(dev)
    end
end

@hascuda CuArrays.allowscalar(false)

abstract type Architecture end
struct CPU <: Architecture end
struct GPU <: Architecture end

device(::CPU) = GPUifyLoops.CPU()
device(::GPU) = GPUifyLoops.CUDA()

abstract type ConstantsCollection end
abstract type EquationOfState end
abstract type Grid{T} end
abstract type AbstractModel end
abstract type Field{A, G} end
abstract type FaceField{A, G} <: Field{A, G} end
abstract type OutputWriter end
abstract type Diagnostic end
abstract type PoissonSolver end

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
