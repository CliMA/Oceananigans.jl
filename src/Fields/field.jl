using Base: @propagate_inbounds

using CUDA
using OffsetArrays

using Oceananigans.Architectures
using Oceananigans.Utils
using Oceananigans.Grids: total_length, interior_indices, interior_parent_indices

import Base: size, length, iterate, getindex, setindex!, lastindex
import Adapt

import Oceananigans.Architectures: architecture
import Oceananigans.Utils: datatuple
import Oceananigans.Grids: total_size, topology, nodes, xnodes, ynodes, znodes, xnode, ynode, znode

"""
    AbstractField{X, Y, Z, A, G}

Abstract supertype for fields located at `(X, Y, Z)` with data stored in a container
of type `A`. The field is defined on a grid `G`.
"""
abstract type AbstractField{X, Y, Z, A, G} end

"""
    Field{X, Y, Z, A, G, B} <: AbstractField{X, Y, Z, A, G}

A field defined at the location (`X`, `Y`, `Z`), each of which can be either `Cell`
or `Face`, and with data stored in a container of type `A` (typically an array).
The field is defined on a grid `G` and has field boundary conditions `B`.
"""
struct Field{X, Y, Z, A, G, B} <: AbstractField{X, Y, Z, A, G}
                   data :: A
                   grid :: G
    boundary_conditions :: B

    function Field{X, Y, Z}(data, grid, bcs) where {X, Y, Z}
        Tx, Ty, Tz = total_size((X, Y, Z), grid)

        if size(data) != (Tx, Ty, Tz)
            e = "Cannot construct field at ($X, $Y, $Z) with size(data)=$(size(data)). " *
                "`data` must have size ($Tx, $Ty, $Tz)."
            throw(ArgumentError(e))
        end

        return new{X, Y, Z, typeof(data), typeof(grid), typeof(bcs)}(data, grid, bcs)
    end
end

"""
    Field(X, Y, Z, arch, grid, [  bcs = FieldBoundaryConditions(grid, (X, Y, Z)),
                                 data = zeros(arch, grid, (X, Y, Z)) ] )

Construct a `Field` on `grid` with `data` on architecture `arch` with
boundary conditions `bcs`. Each of `(X, Y, Z)` is either `Cell` or `Face` and determines
the field's location in `(x, y, z)`.

Example
=======

julia> ω = Field(Face, Face, Cell, CPU(), RegularCartesianmodel.grid)

"""
function Field(X, Y, Z, arch, grid,
                bcs = FieldBoundaryConditions(grid, (X, Y, Z)),
               data = zeros(eltype(grid), arch, grid, (X, Y, Z)))

    return Field{X, Y, Z}(data, grid, bcs)
end

#####
##### Convenience constructor for Field that uses a 3-tuple of locations rather than a list of locations:
#####

# Type "destantiation": convert Face() to Face and Cell() to Cell if needed.
destantiate(X) = typeof(X)
destantiate(X::DataType) = X

"""
    Field(L::Tuple, arch, grid, data, bcs)

Construct a `Field` at the location defined by the 3-tuple `L`,
whose elements are `Cell` or `Face`.
"""
Field(L::Tuple, args...) = Field(destantiate.(L)..., args...)

#####
##### Special constructors for tracers and velocity fields
#####

"""
    CellField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
              [  bcs = TracerBoundaryConditions(grid),
                data = zeros(FT, arch, grid, (Cell, Cell, Cell) ] )

Return a `Field{Cell, Cell, Cell}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function CellField(FT::DataType, arch, grid,
                    bcs = TracerBoundaryConditions(grid),
                   data = zeros(FT, arch, grid, (Cell, Cell, Cell)))

    return Field{Cell, Cell, Cell}(data, grid, bcs)
end

"""
    XFaceField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
               [  bcs = UVelocityBoundaryConditions(grid),
                 data = zeros(FT, arch, grid, (Face, Cell, Cell) ] )

Return a `Field{Face, Cell, Cell}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function XFaceField(FT::DataType, arch, grid,
                     bcs = UVelocityBoundaryConditions(grid),
                    data = zeros(FT, arch, grid, (Face, Cell, Cell)))

    return Field{Face, Cell, Cell}(data, grid, bcs)
end

"""
    YFaceField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
               [  bcs = VVelocityBoundaryConditions(grid),
                 data = zeros(FT, arch, grid, (Cell, Face, Cell)) ] )

Return a `Field{Cell, Face, Cell}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function YFaceField(FT::DataType, arch, grid,
                     bcs = VVelocityBoundaryConditions(grid),
                    data = zeros(FT, arch, grid, (Cell, Face, Cell)))

    return Field{Cell, Face, Cell}(data, grid, bcs)
end

"""
    ZFaceField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
               [  bcs = WVelocityBoundaryConditions(grid),
                 data = zeros(FT, arch, grid, (Cell, Cell, Face)) ] )

Return a `Field{Cell, Cell, Face}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function ZFaceField(FT::DataType, arch, grid,
                     bcs = WVelocityBoundaryConditions(grid),
                    data = zeros(FT, arch, grid, (Cell, Cell, Face)))

    return Field{Cell, Cell, Face}(data, grid, bcs)
end

 CellField(arch::AbstractArchitecture, grid, args...) =  CellField(eltype(grid), arch, grid, args...)
XFaceField(arch::AbstractArchitecture, grid, args...) = XFaceField(eltype(grid), arch, grid, args...)
YFaceField(arch::AbstractArchitecture, grid, args...) = YFaceField(eltype(grid), arch, grid, args...)
ZFaceField(arch::AbstractArchitecture, grid, args...) = ZFaceField(eltype(grid), arch, grid, args...)

#####
##### Functions for querying fields
#####

location(a) = nothing

"Returns the location `(X, Y, Z)` of an `AbstractField{X, Y, Z}`."
location(::AbstractField{X, Y, Z}) where {X, Y, Z} = (X, Y, Z) # note no instantiation
location(f, i) = location(f)[i]

"Returns the architecture where the field data `f.data` is stored."
architecture(f::Field) = architecture(f.data)
architecture(o::OffsetArray) = architecture(o.parent)

"Returns the length of a field's `data`."
@inline length(f::Field) = length(f.data)


"Returns the topology of a fields' `grid`."
@inline topology(f, args...) = topology(f.grid, args...)

"""
    size(loc, grid)

Returns the size of a field at `loc` on `grid`.
This is a 3-tuple of integers corresponding to the number of interior nodes
of `f` along `x, y, z`.
"""
@inline size(loc, grid) = (total_length(loc[1], topology(grid, 1), grid.Nx),
                           total_length(loc[2], topology(grid, 2), grid.Ny),
                           total_length(loc[3], topology(grid, 3), grid.Nz))

"""
    size(f::AbstractField{X, Y, Z}) where {X, Y, Z}

Returns the size of an `AbstractField{X, Y, Z}` located at `X, Y, Z`.
This is a 3-tuple of integers corresponding to the number of interior nodes
of `f` along `x, y, z`.
"""
@inline size(f::AbstractField) = size(location(f), f.grid)

"""
    total_size(loc, grid)

Returns the "total" size of a field at `loc` on `grid`.
This is a 3-tuple of integers corresponding to the number of grid points
contained by `f` along `x, y, z`.
"""
@inline total_size(loc, grid) = (total_length(loc[1], topology(grid, 1), grid.Nx, grid.Hx),
                                 total_length(loc[2], topology(grid, 2), grid.Ny, grid.Hy),
                                 total_length(loc[3], topology(grid, 3), grid.Nz, grid.Hz))

@inline total_size(f::AbstractField) = total_size(location(f), f.grid)

@propagate_inbounds getindex(f::Field, inds...) = @inbounds getindex(f.data, inds...)
@propagate_inbounds setindex!(f::Field, v, inds...) = @inbounds setindex!(f.data, v, inds...)

@inline lastindex(f::Field) = lastindex(f.data)
@inline lastindex(f::Field, dim) = lastindex(f.data, dim)

"Returns `f.data` for `f::Field` or `f` for `f::AbstractArray."
@inline data(a) = a # fallback
@inline data(f::Field) = f.data

@inline cpudata(a) = data(a)

const OffsetCuArray = OffsetArray{T, D, <:CuArray} where {T, D}

@hascuda @inline cpudata(f::Field{X, Y, Z, <:OffsetCuArray}) where {X, Y, Z} =
    OffsetArray(Array(parent(f)), f.grid, location(f))

# Endpoint for recursive `datatuple` function:
@inline datatuple(obj::AbstractField) = obj #data(obj)

"Returns `f.data.parent` for `f::Field`."
@inline Base.parent(f::Field) = f.data.parent

"Returns a view of `f` that excludes halo points."
@inline interior(f::Field{X, Y, Z}) where {X, Y, Z} = view(f.data, interior_indices(X, topology(f, 1), f.grid.Nx),
                                                                   interior_indices(Y, topology(f, 2), f.grid.Ny),
                                                                   interior_indices(Z, topology(f, 3), f.grid.Nz))

"Returns a reference (not a view) to the interior points of `field.data.parent.`"
@inline interiorparent(f::Field{X, Y, Z}) where {X, Y, Z} =
    @inbounds f.data.parent[interior_parent_indices(X, topology(f, 1), f.grid.Nx, f.grid.Hx),
                            interior_parent_indices(Y, topology(f, 2), f.grid.Ny, f.grid.Hy),
                            interior_parent_indices(Z, topology(f, 3), f.grid.Nz, f.grid.Hz)]

iterate(f::Field, state=1) = iterate(f.data, state)

@inline xnode(i, ψ::Field{X, Y, Z}) where {X, Y, Z} = xnode(X, i, ψ.grid)
@inline ynode(j, ψ::Field{X, Y, Z}) where {X, Y, Z} = ynode(Y, j, ψ.grid)
@inline znode(k, ψ::Field{X, Y, Z}) where {X, Y, Z} = znode(Z, k, ψ.grid)

xnodes(ψ::AbstractField) = xnodes(location(ψ, 1), ψ.grid)
ynodes(ψ::AbstractField) = ynodes(location(ψ, 2), ψ.grid)
znodes(ψ::AbstractField) = znodes(location(ψ, 3), ψ.grid)

nodes(ψ::AbstractField; kwargs...) = nodes(location(ψ), ψ.grid; kwargs...)

#####
##### Creating offset arrays for field data by dispatching on architecture.
#####

"""
Return a range of indices for a field located at `Cell` centers
`along a grid dimension of length `N` and with halo points `H`.
"""
offset_indices(loc, topo, N, H=0) = 1 - H : N + H

"""
Return a range of indices for a field located at cell `Face`s
`along a grid dimension of length `N` and with halo points `H`.
"""
offset_indices(::Type{Face}, ::Type{Bounded}, N, H=0) = 1 - H : N + H + 1

"""
    OffsetArray(underlying_data, grid, loc)

Returns an `OffsetArray` that maps to `underlying_data` in memory,
with offset indices appropriate for the `data` of a field on
a `grid` of `size(grid)` and located at `loc`.
"""
function OffsetArray(underlying_data, grid, loc)
    ii = offset_indices(loc[1], topology(grid, 1), grid.Nx, grid.Hx)
    jj = offset_indices(loc[2], topology(grid, 2), grid.Ny, grid.Hy)
    kk = offset_indices(loc[3], topology(grid, 3), grid.Nz, grid.Hz)

    return OffsetArray(underlying_data, ii, jj, kk)
end

"""
    zeros([FT=Float64], ::CPU, grid, loc)

Returns an `OffsetArray` of zeros of float type `FT`, with
parent data in CPU memory and indices corresponding to a field on a
`grid` of `size(grid)` and located at `loc`.
"""
function Base.zeros(FT, ::CPU, grid, loc=(Cell, Cell, Cell))
    underlying_data = zeros(FT, total_length(loc[1], topology(grid, 1), grid.Nx, grid.Hx),
                                total_length(loc[2], topology(grid, 2), grid.Ny, grid.Hy),
                                total_length(loc[3], topology(grid, 3), grid.Nz, grid.Hz))

    return OffsetArray(underlying_data, grid, loc)
end

"""
    zeros([FT=Float64], ::GPU, grid, loc)

Returns an `OffsetArray` of zeros of float type `FT`, with
parent data in GPU memory and indices corresponding to a field on a `grid`
of `size(grid)` and located at `loc`.
"""
function Base.zeros(FT, ::GPU, grid, loc=(Cell, Cell, Cell))
    underlying_data = CuArray{FT}(undef, total_length(loc[1], topology(grid, 1), grid.Nx, grid.Hx),
                                         total_length(loc[2], topology(grid, 2), grid.Ny, grid.Hy),
                                         total_length(loc[3], topology(grid, 3), grid.Nz, grid.Hz))

    underlying_data .= 0 # Ensure data is initially 0.

    return OffsetArray(underlying_data, grid, loc)
end

# Default to type of Grid
Base.zeros(arch, grid, loc=(Cell, Cell, Cell)) = zeros(eltype(grid), arch, grid, loc)

Base.zeros(FT, ::CPU, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz)
Base.zeros(FT, ::GPU, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz) |> CuArray

Base.zeros(arch, grid, Nx, Ny, Nz) = zeros(eltype(grid), arch, Nx, Ny, Nz)
