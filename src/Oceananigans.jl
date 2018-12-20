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

    TimeStepper,

    laplacian3d_ppn,

    SpectralSolverParameters,
    solve_poisson_1d_pbc,
    solve_poisson_1d_nbc,
    solve_poisson_2d_pbc,
    solve_poisson_2d_mbc,
    solve_poisson_3d_pbc,
    solve_poisson_3d_mbc,
    solve_poisson_3d_ppn,
    solve_poisson_3d_ppn!,
    solve_poisson_3d_ppn_planned!

using
    FFTW

abstract type ConstantsCollection end
abstract type EquationOfStateParameters <: ConstantsCollection end
abstract type Grid end
abstract type Field end
abstract type FaceField <: Field end
abstract type FieldSet end
abstract type TimeStepper end

include("planetary_constants.jl")
include("grid.jl")
include("field.jl")
include("fieldset.jl")

include("operators/operators.jl")

include("equation_of_state.jl")
include("time_steppers.jl")
include("spectral_solvers.jl")

end # module
