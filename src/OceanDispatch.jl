module OceanDispatch

export
    EarthConstants,

    RegularCartesianGrid,

    ZoneField,
    FaceField,
    Fields,
    SourceTerms

abstract type ConstantsCollection end
abstract type Grid end
abstract type Field end
abstract type FieldCollection end
abstract type TimeStepper end

include("planetary_constants.jl")
include("grid.jl")
include("fields.jl")
include("time_steppers.jl")

end # module
