module Fields

export
    Face, Cell,
    AbstractField, Field, CellField, XFaceField, YFaceField, ZFaceField,
    ReducedField, AveragedField, ComputedField,
    interior, interiorparent, data,
    xnode, ynode, znode, location,
    set!, compute!, @compute,
    VelocityFields, TracerFields, tracernames, PressureFields, TendencyFields

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.BoundaryConditions

Base.zeros(FT, ::CPU, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz)
Base.zeros(FT, ::GPU, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz) |> CuArray
Base.zeros(arch, grid, Nx, Ny, Nz) = zeros(eltype(grid), arch, Nx, Ny, Nz)

assumed_field_location(name) = name === :u ? (Face, Cell, Cell) :
                               name === :v ? (Cell, Face, Cell) :
                               name === :w ? (Cell, Cell, Face) :
                                             (Cell, Cell, Cell)

include("new_data.jl")
include("abstract_field.jl")
include("field.jl")
include("reduced_field.jl")
include("averaged_field.jl")
include("computed_field.jl")
include("pressure_field.jl")
include("function_field.jl")
include("set!.jl")
include("tracer_names.jl")
include("validate_field_tuple_grid.jl")
include("field_tuples.jl")
include("show_fields.jl")

end
