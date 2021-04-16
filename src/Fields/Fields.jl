module Fields

export
    Face, Center,
    AbstractField, Field,
    CenterField, XFaceField, YFaceField, ZFaceField,
    ReducedField, AveragedField, ComputedField, KernelComputedField, BackgroundField,
    interior, interiorparent, data,
    xnode, ynode, znode, location,
    set!, compute!, @compute,
    VelocityFields, TracerFields, tracernames, PressureFields, TendencyFields,
    interpolate, FieldSlicer

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.BoundaryConditions

import Base: zeros

zeros(FT, ::CPU, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz)
zeros(FT, ::GPU, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz) |> CuArray
zeros(arch, grid, Nx, Ny, Nz) = zeros(eltype(grid), arch, Nx, Ny, Nz)

include("new_data.jl")
include("abstract_field.jl")
include("field.jl")
include("zero_field.jl")
include("reduced_field.jl")
include("averaged_field.jl")
include("computed_field.jl")
include("kernel_computed_field.jl")
include("pressure_field.jl")
include("function_field.jl")
include("set!.jl")
include("tracer_names.jl")
include("validate_field_tuple_grid.jl")
include("field_tuples.jl")
include("background_fields.jl")
include("interpolate.jl")
include("field_slicer.jl")
include("show_fields.jl")
include("broadcasting.jl")

end
