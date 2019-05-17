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
    set!,

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
    BoundaryConditions,
    BoundaryCondition,
    Default,
    Flux,
    Gradient,

    # Time stepping
    time_step!,
    time_step_kernel!,

    # Poisson solver
    PoissonSolver,
    PoissonSolverGPU,
    init_poisson_solver,
    solve_poisson_3d_ppn,
    solve_poisson_3d_ppn!,
    solve_poisson_3d_ppn_planned!,
    solve_poisson_3d_ppn_gpu!,
    solve_poisson_3d_ppn_gpu_planned!,

    # Model helper structs, e.g. configuration, clock, etc.
    ModelConfiguration,
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
    Nusselt_wT,
    Nusselt_Chi,

    # Package utilities
    prettytime

# Standard library modules
using
    Statistics,
    Printf

# Third-party modules
using
    FFTW,
    JLD,
    NetCDF,
    StaticArrays

import
    Adapt,
    GPUifyLoops

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

# @hascuda CuArrays.allowscalar(false)

abstract type Metadata end
abstract type ConstantsCollection end
abstract type EquationOfState end
abstract type Grid{T} end
abstract type Field end
abstract type FaceField <: Field end
abstract type FieldSet end
abstract type OutputWriter end
abstract type Diagnostic end
abstract type AbstractPoissonSolver end

include("utils.jl")

include("model_configuration.jl")
include("clock.jl")
include("planetary_constants.jl")
include("grids.jl")
include("fields.jl")
include("fieldsets.jl")
include("forcing.jl")

include("operators/operators.jl")

include("boundary_conditions.jl")
include("equation_of_state.jl")
include("poisson_solvers.jl")
include("models.jl")
include("time_steppers.jl")

include("output_writers.jl")
include("diagnostics.jl")

end # module
