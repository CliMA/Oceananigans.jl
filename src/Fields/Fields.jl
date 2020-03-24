module Fields

export
    Face, Cell,
    AbstractField, Field, CellField, XFaceField, YFaceField, ZFaceField,
    interior, interiorparent, data,
    xnode, ynode, znode, location,
    set!,
    VelocityFields, TracerFields, tracernames, PressureFields, TendencyFields

using Oceananigans.Grids
using Oceananigans.BoundaryConditions

include("field.jl")
include("set!.jl")
include("field_tuples.jl")
include("show_fields.jl")

end
