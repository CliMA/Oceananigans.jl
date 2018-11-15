module OceanDispatch

export
    ConstantsCollection,
    Grid,
    Field,
    FieldCollection,
    TimeStepper,

    PlanetaryConstants,
    EarthConstants,

    RegularCartesianGrid,

    ZoneField,
    FaceField,
    Fields,
    SourceTermFields,
    ForcingFields

abstract type ConstantsCollection end
abstract type EquationOfStateParameters <: ConstantsCollection end
abstract type Grid end
abstract type Field end
# abstract type Field <: AbstractArray end
abstract type FieldCollection end
abstract type TimeStepper end

include("planetary_constants.jl")
include("grid.jl")
include("fields.jl")
include("equation_of_state_future.jl")
include("time_steppers.jl")

end # module
