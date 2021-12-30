module Fields

export Face, Center
export AbstractField, Field, Average, Integral, Reduction
export CenterField, XFaceField, YFaceField, ZFaceField
export BackgroundField
export interior, data, xnode, ynode, znode, location
export set!, compute!, @compute, regrid!
export VelocityFields, TracerFields, tracernames, PressureFields, TendencyFields
export interpolate, FieldSlicer

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.BoundaryConditions

include("abstract_field.jl")
include("field.jl")
include("field_reductions.jl")
include("zero_field.jl")
include("averaged_field.jl")
include("kernel_computed_field.jl")
include("function_field.jl")
include("regridding_fields.jl")
include("set!.jl")
include("tracer_names.jl")
include("validate_field_tuple_grid.jl")
include("field_tuples.jl")
include("background_fields.jl")
include("interpolate.jl")
include("field_slicer.jl")
include("show_fields.jl")
include("broadcasting_abstract_fields.jl")

end
