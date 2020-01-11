module Fields

export
    Face, Cell,
    AbstractField, AbstractLocatedField,
    Field, CellField, FaceFieldX, FaceFieldY, FaceFieldZ,
    interior, interiorparent, set!,
    VelocityFields, TracerFields, PressureFields, Tendencies

include("field_utils.jl")
include("field.jl")
include("set!.jl")
include("field_tuples.jl")
include("show_fields.jl")

end
