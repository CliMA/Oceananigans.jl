import Base: size, length, iterate, getindex, setindex!, lastindex
using Base: @propagate_inbounds

import Adapt
using OffsetArrays

import Oceananigans.Architectures: architecture
import Oceananigans.Utils: datatuple
import Oceananigans.Grids: topology, x_topology, y_topology, z_topology

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Utils

@hascuda using CuArrays

"""
    AbstractField{A, G}

Abstract supertype for fields stored on an architecture `A` and defined on a grid `G`.
"""
abstract type AbstractField{A, G} end

"""
    AbstractLocatedField{X, Y, Z, A, G}

Abstract supertype for fields located at `(X, Y, Z)`, stored on an architecture `A`,
and defined on a grid `G`.
"""
abstract type AbstractLocatedField{X, Y, Z, A, G} <: AbstractField{A, G} end

"""
    Cell

A type describing the location at the center of a grid cell.
"""
struct Cell end

"""
	Face

A type describing the location at the face of a grid cell.
"""
struct Face end

"""
    Field{X, Y, Z, A, G} <: AbstractLocatedField{X, Y, Z, A, G}

A field defined at the location (`X`, `Y`, `Z`) which can be either `Cell` or `Face`.
"""
struct Field{X, Y, Z, A, G} <: AbstractLocatedField{X, Y, Z, A, G}
    data :: A
    grid :: G
    function Field{X, Y, Z}(data, grid) where {X, Y, Z}
        return new{X, Y, Z, typeof(data), typeof(grid)}(data, grid)
    end
end

Adapt.adapt_structure(to, field::Field{X, Y, Z}) where {X, Y, Z} =
    Field{X, Y, Z}(adapt(to, data), field.grid)

"""
	Field(L::Tuple, data::AbstractArray, grid)

Construct a `Field` on `grid` using the array `data` with location defined by the tuple `L`
of length 3 whose elements are `Cell` or `Face`.
"""
Field(L::Tuple, data::AbstractArray, grid) = Field{L[1], L[2], L[3]}(data, grid)

"""
    Field(L::Tuple, arch, grid)

Construct a `Field` on architecture `arch` and `grid` at location `L`,
where `L` is a tuple of `Cell` or `Face` types.
"""
Field(L::Tuple, arch, grid) = Field{L[1], L[2], L[3]}(zeros(arch, grid, L), grid)

"""
    Field(X, Y, Z, arch, grid)

Construct a `Field` on architecture `arch` and `grid` at location `X`, `Y`, `Z`,
where each of `X, Y, Z` is `Cell` or `Face`.
"""
Field(X, Y, Z, arch, grid) =  Field((X, Y, Z), arch, grid)

"""
    CellField([FT=eltype(grid)], arch, grid)

Return a `Field{Cell, Cell, Cell}` on architecture `arch` and `grid`.
Used for tracers and pressure fields.
"""
CellField(FT, arch, grid) = Field{Cell, Cell, Cell}(zeros(FT, arch, grid, (Cell, Cell, Cell)), grid)

"""
    FaceFieldX([FT=eltype(grid)], arch, grid)

Return a `Field{Face, Cell, Cell}` on architecture `arch` and `grid`.
Used for the x-velocity field.
"""
FaceFieldX(FT, arch, grid) = Field{Face, Cell, Cell}(zeros(FT, arch, grid, (Face, Cell, Cell)), grid)

"""
    FaceFieldY([FT=eltype(grid)], arch, grid)

Return a `Field{Cell, Face, Cell}` on architecture `arch` and `grid`.
Used for the y-velocity field.
"""
FaceFieldY(FT, arch, grid) = Field{Cell, Face, Cell}(zeros(FT, arch, grid, (Cell, Face, Cell)), grid)

"""
    FaceFieldZ([FT=eltype(grid)], arch, grid)

Return a `Field{Cell, Cell, Face}` on architecture `arch` and `grid`.
Used for the z-velocity field.
"""
FaceFieldZ(FT, arch, grid) = Field{Cell, Cell, Face}(zeros(FT, arch, grid, (Cell, Cell, Face)), grid)

 CellField(arch, grid) = Field((Cell, Cell, Cell), arch, grid)
FaceFieldX(arch, grid) = Field((Face, Cell, Cell), arch, grid)
FaceFieldY(arch, grid) = Field((Cell, Face, Cell), arch, grid)
FaceFieldZ(arch, grid) = Field((Cell, Cell, Face), arch, grid)

#####
##### Functions for querying fields
#####

location(a) = nothing

"Returns the location `(X, Y, Z)` of an `AbstractLocatedField{X, Y, Z}`."
location(::AbstractLocatedField{X, Y, Z}) where {X, Y, Z} = (X, Y, Z) # note no instantiation

x_location(::AbstractLocatedField{X}) where X               = X
y_location(::AbstractLocatedField{X, Y}) where {X, Y}       = Y
z_location(::AbstractLocatedField{X, Y, Z}) where {X, Y, Z} = Z

"Returns the architecture where the field data `f.data` is stored."
architecture(f::Field) = architecture(f.data)

"Returns the length of a field's `data`."    
@inline length(f::Field) = length(f.data)

"Returns the architecture where offset array data `o.parent` is stored."
architecture(o::OffsetArray) = architecture(o.parent)

"Returns the topology of a fields' `grid`."
@inline topology(f) = topology(f.grid)
@inline x_topology(f) = x_topology(f.grid)
@inline y_topology(f) = y_topology(f.grid)
@inline z_topology(f) = z_topology(f.grid)

"""
Returns the length of a field located at `Cell` centers along a grid 
dimension of length `N` and with halo points `H`.
"""
length(loc, topo, N, H=0) = N + 2H

"""
Returns the length of a field located at cell `Face`s along a grid 
dimension of length `N` and with halo points `H`.
"""
length(::Type{Face}, ::Bounded, N, H=0) = N + 1 + 2H

"Returns the size of `f.grid`."
@inline size(f::AbstractField) = size(f.grid)

"""
    size(f::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}

Returns the size of an `AbstractLocatedField{X, Y, Z}` located at `X, Y, Z`.
This is a 3-tuple of integers corresponding to the number of interior nodes
of `f` along `x, y, z`.
"""
@inline size(f::AbstractLocatedField{X, Y, Z}) where {X, Y, Z} = (length(X, x_topology(f), f.grid.Nx), 
                                                                  length(Y, y_topology(f), f.grid.Ny), 
                                                                  length(Z, z_topology(f), f.grid.Nz))

@propagate_inbounds getindex(f::Field, inds...) = @inbounds getindex(f.data, inds...)
@propagate_inbounds setindex!(f::Field, v, inds...) = @inbounds setindex!(f.data, v, inds...)

@inline lastindex(f::Field) = lastindex(f.data)
@inline lastindex(f::Field, dim) = lastindex(f.data, dim)

"Returns `f.data` for `f::Field` or `f` for `f::AbstractArray."
@inline data(a) = a
@inline data(f::Field) = f.data

# Endpoint for recursive `datatuple` function:
@inline datatuple(obj::AbstractField) = data(obj)

"Returns `f.data.parent` for `f::Field`."
@inline Base.parent(f::Field) = f.data.parent

@inline interior_indices(loc, topo, N) = 1:N
@inline interior_indices(::Type{Face}, ::Bounded, N) = 1:N+1

@inline interior_parent_indices(loc, topo, N, H) = 1+H:N+H
@inline interior_parent_indices(::Type{Face}, ::Bounded, N, H) = 1+H:N+1+H

"Returns a view of `f` that excludes halo points."
@inline interior(f::Field{X, Y, Z}) where {X, Y, Z} = view(f.data, interior_indices(X, x_topology(f), f.grid.Nx), 
                                                                   interior_indices(Y, y_topology(f), f.grid.Ny), 
                                                                   interior_indices(Z, z_topology(f), f.grid.Nz))

"Returns a reference (not a view) to the interior points of `field.data.parent.`"
@inline interiorparent(f::Field) = @inbounds f.data.parent[interior_parent_indices(X, x_topology(f), f.grid.Nx, f.grid.Hx),
                                                           interior_parent_indices(Y, y_topology(f), f.grid.Ny, f.grid.Hy),
                                                           interior_parent_indices(Z, z_topology(f), f.grid.Nz, f.grid.Hz)]

iterate(f::Field, state=1) = iterate(f.data, state)

@inline xnode(::Type{Cell}, i, grid) = @inbounds grid.xC[i]
@inline xnode(::Type{Face}, i, grid) = @inbounds grid.xF[i]

@inline ynode(::Type{Cell}, j, grid) = @inbounds grid.yC[j]
@inline ynode(::Type{Face}, j, grid) = @inbounds grid.yF[j]

@inline znode(::Type{Cell}, k, grid) = @inbounds grid.zC[k]
@inline znode(::Type{Face}, k, grid) = @inbounds grid.zF[k]

@inline xnode(i, ψ::Field{X, Y, Z}) where {X, Y, Z} = xnode(X, i, ψ.grid)
@inline ynode(j, ψ::Field{X, Y, Z}) where {X, Y, Z} = ynode(Y, j, ψ.grid)
@inline znode(k, ψ::Field{X, Y, Z}) where {X, Y, Z} = znode(Z, k, ψ.grid)

# Dispatch insanity
xnodes(::Type{Cell}, topo, grid) = reshape(ψ.grid.xC, ψ.grid.Nx, 1, 1)
ynodes(::Type{Cell}, topo, grid) = reshape(ψ.grid.yC, 1, ψ.grid.Ny, 1)
znodes(::Type{Cell}, topo, grid) = reshape(ψ.grid.zC, 1, 1, ψ.grid.Nz)

xnodes(::Type{Face}, topo, grid) = reshape(ψ.grid.xF[1:end-1], ψ.grid.Nx, 1, 1)
ynodes(::Type{Face}, topo, grid) = reshape(ψ.grid.yF[1:end-1], 1, ψ.grid.Ny, 1)
znodes(::Type{Face}, topo, grid) = reshape(ψ.grid.zF[1:end-1], 1, 1, ψ.grid.Nz)

xnodes(::Type{Face}, ::Bounded, grid) = reshape(ψ.grid.xF, ψ.grid.Nx+1, 1, 1)
ynodes(::Type{Face}, ::Bounded, grid) = reshape(ψ.grid.yF, 1, ψ.grid.Ny+1, 1)
znodes(::Type{Face}, ::Bounded, grid) = reshape(ψ.grid.zF, 1, 1, ψ.grid.Nz+1)

xnodes(ψ::AbstractField) = xnodes(x_location(ψ), x_topology(ψ), ψ.grid)
ynodes(ψ::AbstractField) = ynodes(y_location(ψ), y_topology(ψ), ψ.grid)
znodes(ψ::AbstractField) = znodes(z_location(ψ), z_topology(ψ), ψ.grid)

nodes(ψ::AbstractField) = (xnodes(ψ), ynodes(ψ), znodes(ψ))

# Nicites (but what for?)
const AbstractCPUField =
    AbstractField{A, G} where {A<:OffsetArray{FT, D, <:Array} where {FT, D}, G}

@hascuda const AbstractGPUField =
    AbstractField{A, G} where {A<:OffsetArray{FT, D, <:CuArray} where {FT, D}, G}

#####
##### Creating fields by dispatching on architecture
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
offset_indices(::Type{Face}, ::Bounded, N, H=0) = 1 - H : N + H + 1

"""
    OffsetArray(underlying_data, grid, loc)

Returns an `OffsetArray` that maps to `underlying_data` in memory,
with offset indices appropriate for the `data` of a field on 
a `grid` of `size(grid)` and located at `loc`.
"""
function OffsetArray(underlying_data, grid, loc)
    ii = offset_indices(loc[1], x_topology(grid), grid.Nx, grid.Hx)
    jj = offset_indices(loc[2], y_topology(grid), grid.Ny, grid.Hy)
    kk = offset_indices(loc[3], z_topology(grid), grid.Nz, grid.Hz)

    return OffsetArray(underlying_data, ii, jj, kk)
end

"""
    zeros([FT=Float64], ::CPU, grid, loc)

Returns an `OffsetArray` of zeros of float type `FT`, with
parent data in CPU memory and indices corresponding to a field on a 
`grid` of `size(grid)` and located at `loc`.
"""
function Base.zeros(FT, ::CPU, grid, loc)
    underlying_data = zeros(FT, length(loc[1], x_topology(grid), grid.Nx, grid.Hx),
                                length(loc[2], y_topology(grid), grid.Ny, grid.Hy),
                                length(loc[3], z_topology(grid), grid.Nz, grid.Hz))

    return OffsetArray(underlying_data, grid, loc)
end

"""
    zeros([FT=Float64], ::GPU, grid, loc)

Returns an `OffsetArray` of zeros of float type `FT`, with
parent data in GPU memory and indices corresponding to a field on a `grid`
of `size(grid)` and located at `loc`.
"""
function Base.zeros(FT, ::GPU, grid, loc)
    underlying_data = CuArray{FT}(undef, length(loc[1], x_topology(grid), grid.Nx, grid.Hx),
                                         length(loc[2], y_topology(grid), grid.Ny, grid.Hy),
                                         length(loc[3], z_topology(grid), grid.Nz, grid.Hz))

    underlying_data .= 0 # Ensure data is initially 0.

    return OffsetArray(underlying_data, grid, loc)
end

# Default to type of Grid
Base.zeros(arch, grid, loc) = zeros(eltype(grid), arch, grid, loc)






#Base.zeros(arch, grid::AbstractGrid{FT}, Nx, Ny, Nz, loc=(Cell, Cell, Cell)) where FT = 
#    zeros(FT, arch, grid, Nx, Ny, Nz)
           

#=
function Base.zeros(FT, ::CPU, grid)
    underlying_data = zeros(FT, grid.Nx + 2grid.Hx, grid.Ny + 2grid.Hy, grid.Nz + 2grid.Hz)
    return OffsetArray(underlying_data, grid)
end

function Base.zeros(FT, ::GPU, grid)
    underlying_data = CuArray{FT}(undef, grid.Nx + 2grid.Hx, grid.Ny + 2grid.Hy, grid.Nz + 2grid.Hz)
    underlying_data .= 0  # Gotta do this otherwise you might end up with a few NaN values!
    return OffsetArray(underlying_data, grid, loc)
end

Base.zeros(FT, ::CPU, grid, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz)
Base.zeros(FT, ::GPU, grid, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz) |> CuArray
=#


