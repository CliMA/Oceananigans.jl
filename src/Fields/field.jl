using Base: @propagate_inbounds

import Adapt
using OffsetArrays

import Oceananigans.Utils: datatuple
using Oceananigans: AbstractGrid
using Oceananigans.Architectures: @hascuda, CPU, GPU

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
Field(L::Tuple, arch, grid) = Field{L[1], L[2], L[3]}(zeros(arch, grid), grid)

"""
    Field(X, Y, Z, arch, grid)

Construct a `Field` on architecture `arch` and `grid` at location `X`, `Y`, `Z`,
where each of `X, Y, Z` is `Cell` or `Face`.
"""
Field(X, Y, Z, arch, grid) =  Field((X, Y, Z), arch, grid)

"""
    CellField([T=eltype(grid)], arch, grid)

Return a `Field{Cell, Cell, Cell}` on architecture `arch` and `grid`.
Used for tracers and pressure fields.
"""
CellField(T, arch, grid) = Field{Cell, Cell, Cell}(zeros(T, arch, grid), grid)

"""
    FaceFieldX([T=eltype(grid)], arch, grid)

Return a `Field{Face, Cell, Cell}` on architecture `arch` and `grid`.
Used for the x-velocity field.
"""
FaceFieldX(T, arch, grid) = Field{Face, Cell, Cell}(zeros(T, arch, grid), grid)

"""
    FaceFieldY([T=eltype(grid)], arch, grid)

Return a `Field{Cell, Face, Cell}` on architecture `arch` and `grid`.
Used for the y-velocity field.
"""
FaceFieldY(T, arch, grid) = Field{Cell, Face, Cell}(zeros(T, arch, grid), grid)

"""
    FaceFieldZ([T=eltype(grid)], arch, grid)

Return a `Field{Cell, Cell, Face}` on architecture `arch` and `grid`.
Used for the z-velocity field.
"""
FaceFieldZ(T, arch, grid) = Field{Cell, Cell, Face}(zeros(T, arch, grid), grid)

 CellField(arch, grid) = Field((Cell, Cell, Cell), arch, grid)
FaceFieldX(arch, grid) = Field((Face, Cell, Cell), arch, grid)
FaceFieldY(arch, grid) = Field((Cell, Face, Cell), arch, grid)
FaceFieldZ(arch, grid) = Field((Cell, Cell, Face), arch, grid)

location(a) = nothing
location(::AbstractLocatedField{X, Y, Z}) where {X, Y, Z} = (X, Y, Z)

architecture(f::Field) = architecture(f.data)
architecture(o::OffsetArray) = architecture(o.parent)

@inline size(f::AbstractField) = size(f.grid)
@inline length(f::Field) = length(f.data)

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

"Returns a view over the interior points of the `field.data`."
@inline interior(f::Field) = view(f.data, 1:f.grid.Nx, 1:f.grid.Ny, 1:f.grid.Nz)

"Returns a reference to the interior points of `field.data.parent.`"
@inline interiorparent(f::Field) = @inbounds f.data.parent[1+f.grid.Hx:f.grid.Nx+f.grid.Hx,
                                                           1+f.grid.Hy:f.grid.Ny+f.grid.Hy,
                                                           1+f.grid.Hz:f.grid.Nz+f.grid.Hz]

iterate(f::Field, state=1) = iterate(f.data, state)

@inline xnode(::Type{Cell}, i, grid) = @inbounds grid.xC[i]
@inline xnode(::Type{Face}, i, grid) = @inbounds grid.xF[i]

@inline ynode(::Type{Cell}, j, grid) = @inbounds grid.yC[j]
@inline ynode(::Type{Face}, j, grid) = @inbounds grid.yF[j]

@inline znode(::Type{Cell}, k, grid) = @inbounds grid.zC[k]
@inline znode(::Type{Face}, k, grid) = @inbounds grid.zF[k]

@inline xnode(i, ϕ::Field{X, Y, Z}) where {X, Y, Z} = xnode(X, i, ϕ.grid)
@inline ynode(j, ϕ::Field{X, Y, Z}) where {X, Y, Z} = ynode(Y, j, ϕ.grid)
@inline znode(k, ϕ::Field{X, Y, Z}) where {X, Y, Z} = znode(Z, k, ϕ.grid)

xnodes(ϕ::AbstractField) = reshape(ϕ.grid.xC, ϕ.grid.Nx, 1, 1)
ynodes(ϕ::AbstractField) = reshape(ϕ.grid.yC, 1, ϕ.grid.Ny, 1)
znodes(ϕ::AbstractField) = reshape(ϕ.grid.zC, 1, 1, ϕ.grid.Nz)

xnodes(ϕ::Field{Face})                    = reshape(ϕ.grid.xF[1:end-1], ϕ.grid.Nx, 1, 1)
ynodes(ϕ::Field{X, Face}) where X         = reshape(ϕ.grid.yF[1:end-1], 1, ϕ.grid.Ny, 1)
znodes(ϕ::Field{X, Y, Face}) where {X, Y} = reshape(ϕ.grid.zF[1:end-1], 1, 1, ϕ.grid.Nz)

nodes(ϕ) = (xnodes(ϕ), ynodes(ϕ), znodes(ϕ))

# Niceties
const AbstractCPUField =
    AbstractField{A, G} where {A<:OffsetArray{T, D, <:Array} where {T, D}, G}

@hascuda const AbstractGPUField =
    AbstractField{A, G} where {A<:OffsetArray{T, D, <:CuArray} where {T, D}, G}

#####
##### Creating fields by dispatching on architecture
#####

function OffsetArray(underlying_data, grid)
    # Starting and ending indices for the offset array.
    i1, i2 = 1 - grid.Hx, grid.Nx + grid.Hx
    j1, j2 = 1 - grid.Hy, grid.Ny + grid.Hy
    k1, k2 = 1 - grid.Hz, grid.Nz + grid.Hz

    return OffsetArray(underlying_data, i1:i2, j1:j2, k1:k2)
end

function Base.zeros(T, ::CPU, grid)
    underlying_data = zeros(T, grid.Tx, grid.Ty, grid.Tz)
    return OffsetArray(underlying_data, grid)
end

function Base.zeros(T, ::GPU, grid)
    underlying_data = CuArray{T}(undef, grid.Tx, grid.Ty, grid.Tz)
    underlying_data .= 0  # Gotta do this otherwise you might end up with a few NaN values!
    return OffsetArray(underlying_data, grid)
end

Base.zeros(T, ::CPU, grid, Nx, Ny, Nz) = zeros(T, Nx, Ny, Nz)
Base.zeros(T, ::GPU, grid, Nx, Ny, Nz) = zeros(T, Nx, Ny, Nz) |> CuArray

# Default to type of Grid
Base.zeros(arch, grid::AbstractGrid{T}) where T = zeros(T, arch, grid)
Base.zeros(arch, grid::AbstractGrid{T}, Nx, Ny, Nz) where T = zeros(T, arch, grid, Nx, Ny, Nz)
