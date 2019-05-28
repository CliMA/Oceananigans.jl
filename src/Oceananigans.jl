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
    Value,
    getbc,
    setbc!,

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
    VelocityDivergenceChecker,
    Nusselt_wT,
    Nusselt_Chi,

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

# A slightly complex (but fully-featured?) implementation:

struct ThreadBlockLayout{NT, NB}
    threads :: NTuple{NT, Int}
     blocks :: NTuple{NB, Int}
end

XYZThreadBlockLayout(threads, grid) = ThreadBlockLayout(threads, 
    (floor(Int, threads[1]/grid.Nx), floor(Int, threads[2]/grid.Ny), grid.Nz) )

XYThreadBlockLayout(threads, grid) = ThreadBlockLayout(threads,
    (floor(Int, threads[1]/grid.Nx), floor(Int, threads[2]/grid.Ny)) )

XZThreadBlockLayout(threads, grid) = ThreadBlockLayout(threads,
    (floor(Int, threads[1]/grid.Nx), grid.Nz) )

YZThreadBlockLayout(threads, grid) = ThreadBlockLayout(threads, 
    (floor(Int, threads[2]/grid.Ny), grid.Nz) )

struct GPU{XYZ, XY, XZ, YZ} <: Architecture
    xyz :: XYZ
     xy :: XY
     xz :: XZ
     yz :: YZ
end

GPU(grid; threads=(16, 16)) = GPU(
    XYZThreadBlockLayout(threads, grid), XYThreadBlockLayout(threads, grid),
     XZThreadBlockLayout(threads, grid), YZThreadBlockLayout(threads, grid) )

GPU() = GPU(nothing, nothing, nothing, nothing) # stopgap while code is unchanged.

# Functions permitting generalization:
threads(geom, arch) = nothing
 blocks(geom, arch) = nothing

threads(geom, arch::GPU) = getproperty(getproperty(arch, geom), :threads)
 blocks(geom, arch::GPU) = getproperty(getproperty(arch, geom), :blocks)

# @launch looks like xyz_kernel(args..., threads=threads(:xyz, arch), blocks=blocks(:xyz, arch))
# etc.

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

include("closures/turbulence_closures.jl")

include("boundary_conditions.jl")
include("equation_of_state.jl")
include("poisson_solvers.jl")
include("models.jl")
include("time_steppers.jl")

include("output_writers.jl")
include("diagnostics.jl")

end # module
