module OceanDispatch

export
    RegularCartesianGrid,

    ZoneField,
    FaceField,
    VelocityFields,
    Fields,
    SourceTerms,
abstract type Grid end
abstract type Field end
abstract type FieldCollection end

include("grid.jl")
include("fields.jl")

end # module
