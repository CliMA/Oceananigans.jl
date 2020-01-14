module Fields

export
    Face, Cell,
    AbstractField, AbstractLocatedField,
    Field, CellField, FaceFieldX, FaceFieldY, FaceFieldZ,
    interior, interiorparent,
    xnode, ynode, znode, location,
    set!,
    VelocityFields, TracerFields, tracernames, PressureFields, Tendencies

include("field_utils.jl")
include("field.jl")
include("set!.jl")
include("field_tuples.jl")
include("show_fields.jl")

end
