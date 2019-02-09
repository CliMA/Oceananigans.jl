module Oceananigans

export
    ConstantsCollection,

    PlanetaryConstants,
    EarthConstants,

    Grid,
    RegularCartesianGrid,

    Field,
    FaceField,
    CellField,
    FaceFieldX,
    FaceFieldY,
    FaceFieldZ,
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

    TimeStepper,
    time_stepping!,

    SpectralSolverParameters,
    solve_poisson_1d_pbc,
    solve_poisson_1d_nbc,
    solve_poisson_2d_pbc,
    solve_poisson_2d_mbc,
    solve_poisson_3d_pbc,
    solve_poisson_3d_mbc,
    solve_poisson_3d_ppn,
    solve_poisson_3d_ppn!,
    solve_poisson_3d_ppn_planned!,
    solve_poisson_3d_ppn_gpu!,

    ModelMetadata,
    Model,

    SavedFields

using FFTW

if Base.find_package("CuArrays") !== nothing
    using CUDAdrv, CUDAnative, CuArrays
end

abstract type Metadata end
abstract type ConstantsCollection end
abstract type EquationOfState end
abstract type Grid end
abstract type Field end
abstract type FaceField <: Field end
abstract type FieldSet end
abstract type TimeStepper end

# We must play this annoying game of organizing the definitions to avoid
# mutually circular type declarations.
# See: https://github.com/JuliaLang/julia/issues/269

include("model_metadata.jl")
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

end # module
