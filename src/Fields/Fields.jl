module Fields

export
    Face, Cell,
    AbstractField, AbstractLocatedField,
    Field, CellField, FaceFieldX, FaceFieldY, FaceFieldZ,
    interior, interiorparent, set!

include("field_utils.jl")
include("field.jl")
include("set!.jl")
include("show_fields.jl")

end
