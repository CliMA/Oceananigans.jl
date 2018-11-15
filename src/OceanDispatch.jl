module OceanDispatch

export
    ConstantsCollection,
    Grid,

    Field,
    CellField,
    FaceField,
    FieldCollection,
    set!,

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
abstract type Grid{T} end
abstract type Field{G<:Grid} end
abstract type FieldCollection end
abstract type TimeStepper end

const dim = 3

include("planetary_constants.jl")
include("grid.jl")
include("field.jl")
include("equation_of_state_future.jl")
include("time_steppers.jl")

end # module
