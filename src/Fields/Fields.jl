module Fields

export
    Face, Cell,
    AbstractField, Field, CellField, XFaceField, YFaceField, ZFaceField,
    interior, interiorparent, data,
    xnode, ynode, znode, location,
    set!,
    VelocityFields, TracerFields, tracernames, PressureFields, TendencyFields

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.BoundaryConditions

Base.zeros(FT, ::CPU, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz)
Base.zeros(FT, ::GPU, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz) |> CuArray
Base.zeros(arch, grid, Nx, Ny, Nz) = zeros(eltype(grid), arch, Nx, Ny, Nz)

include("new_data.jl")
include("field.jl")
include("set!.jl")
include("field_tuples.jl")
include("show_fields.jl")

end
