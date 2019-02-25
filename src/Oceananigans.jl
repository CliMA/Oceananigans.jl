module Oceananigans

export
    HAVE_CUDA,
    hascuda,

    ConstantsCollection,

    PlanetaryConstants,
    Earth,
    EarthStationary,

    Grid,
    RegularCartesianGrid,

    Field,
    FaceField,
    CellField,
    FaceFieldX,
    FaceFieldY,
    FaceFieldZ,
    EdgeField,
    set!,

    FieldSet,
    VelocityFields,
    TracerFields,
    PressureFields,
    SourceTerms,
    ForcingFields,
    OperatorTemporaryFields,
    StepperTemporaryFields,

    LinearEquationOfState,
    ρ!,
    δρ!,
    ∫δρgdz!,

    BoundaryConditions,

    time_step!,
    time_step_kernel!,

    SpectralSolverParameters,
    SpectralSolverParametersGPU,
    solve_poisson_3d_ppn,
    solve_poisson_3d_ppn!,
    solve_poisson_3d_ppn_planned!,
    solve_poisson_3d_ppn_gpu!,
    solve_poisson_3d_ppn_gpu_planned!,

    ModelMetadata,
    _ModelMetadata,
    ModelConfiguration,
    _ModelConfiguration,
    Clock,
    Model,

    OutputWriter,
    Checkpointer,
    restore_from_checkpoint,
    BinaryFieldWriter,
    NetCDFFieldWriter,
    write_output,
    read_output,

    Diagnostic,
    run_diagnostic,
    FieldSummary,
    Nusselt_wT,
    Nusselt_Chi,

    prettytime

using Statistics, Printf
using FFTW, JLD, NetCDF

const HAVE_CUDA = try
    using CUDAdrv, CUDAnative, CuArrays
    using GPUifyLoops
    true
catch
    false
end

macro hascuda(ex)
    return HAVE_CUDA ? :($(esc(ex))) : :(nothing)
end

abstract type Metadata end
abstract type ConstantsCollection end
abstract type EquationOfState end
abstract type Grid end
abstract type Field end
abstract type FaceField <: Field end
abstract type FieldSet end
abstract type OutputWriter end
abstract type Diagnostic end

include("model_metadata.jl")
include("model_configuration.jl")
include("clock.jl")
include("planetary_constants.jl")
include("grids.jl")
include("fields.jl")
include("fieldsets.jl")

include("operators/operators.jl")

include("boundary_conditions.jl")
include("equation_of_state.jl")
include("spectral_solvers.jl")
include("model.jl")
include("time_steppers.jl")

include("output_writers.jl")
include("diagnostics.jl")

include("utils.jl")

end # module
