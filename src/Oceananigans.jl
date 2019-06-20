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
    underlying_data,
    set!,
    fill_halo_regions!,

    # FieldSets (collections of related fields)
    FieldSet,
    VelocityFields,
    TracerFields,
    PressureFields,
    SourceTerms,
    StepperTemporaryFields,

    # Forcing functions
    Forcing,

    # Equation of state
    LinearEquationOfState,
    δρ,

    # Boundary conditions
    BoundaryCondition,
    CoordinateBoundaryConditions,
    FieldBoundaryConditions,
    ModelBoundaryConditions,
    Periodic,
    FreeSlip,
    Flux,
    Gradient,
    Value,
    getbc,
    setbc!,

    # Time stepping
    time_step!,

    # Poisson solver
    PoissonSolver,
    PoissonSolverGPU,
    solve_poisson_3d_ppn_planned!,
    solve_poisson_3d_ppn_gpu_planned!,

    # Model helper structs, e.g. clock, etc.
    Clock,
    Model,

    # Model output writers
    OutputWriter,
    Checkpointer,
    restore_from_checkpoint,
    BinaryOutputWriter,
    NetCDFOutputWriter,
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
    TurbulenceClosures

# Standard library modules
using
    Statistics,
    LinearAlgebra,
    Printf

# Third-party modules
using
    FFTW,
    JLD,
    NetCDF,
    StaticArrays,
    OffsetArrays

import
    Adapt,
    GPUifyLoops

# Adapt an offset CuArray to work nicely with CUDA kernels.
Adapt.adapt_structure(to, x::OffsetArray) = OffsetArray(Adapt.adapt(to, parent(x)), x.offsets)

# Import CUDA utilities if cuda is detected.
const HAVE_CUDA = try
    using CUDAdrv, CUDAnative, CuArrays
    true
catch
    false
end

macro hascuda(ex)
    return HAVE_CUDA ? :($(esc(ex))) : :(nothing)
end

abstract type Architecture end
struct CPU <: Architecture end
struct GPU <: Architecture end

device(::CPU) = GPUifyLoops.CPU()
device(::GPU) = GPUifyLoops.CUDA()

@hascuda begin
    println("CUDA-enabled GPU(s) detected:")
    for (gpu, dev) in enumerate(CUDAnative.devices())
        println(dev)
    end
end

@hascuda CuArrays.allowscalar(false)

abstract type Metadata end
abstract type ConstantsCollection end
abstract type EquationOfState end
abstract type Grid{T} end
abstract type Field end
abstract type FaceField <: Field end
abstract type FieldSet end
abstract type OutputWriter end
abstract type Diagnostic end
abstract type PoissonSolver end

include("utils.jl")

include("clock.jl")
include("planetary_constants.jl")
include("grids.jl")
include("fields.jl")
include("fieldsets.jl")
include("forcing.jl")

include("operators/operators.jl")
include("closures/turbulence_closures.jl")

include("boundary_conditions.jl")
include("equation_of_state.jl")
include("poisson_solvers.jl")
include("models.jl")
include("time_steppers.jl")

include("output_writers.jl")
include("diagnostics.jl")

end # module
