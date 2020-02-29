import Base: size, length, iterate, getindex, setindex!, lastindex
using Base: @propagate_inbounds

import Adapt
using OffsetArrays

import Oceananigans.Architectures: architecture
import Oceananigans.Utils: datatuple
import Oceananigans.Grids: total_size, topology

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Utils

@hascuda using CuArrays

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
    Field(L::Tuple, arch, grid, bcs, [data=zeros(arch, grid)])

Construct a `Field` on some architecture `arch` and a `grid` with some `data`.
The field's location is defined by a tuple `L` of length 3 whose elements are
`Cell` or `Face` and has field boundary conditions `bcs`.
"""
function Field(L::Tuple, arch, grid, bcs, 
               data=zeros(eltype(grid), arch, grid, (typeof(L[1]), typeof(L[2]), typeof(L[3]))))
      
    return Field{typeof(L[1]), typeof(L[2]), typeof(L[3])}(data, grid, bcs)
end

function Field(L::NTuple{3, DataType}, arch, grid, bcs, 
               data=zeros(eltype(grid), arch, grid, (L[1], L[2], L[3])))

    return Field{L[1], L[2], L[3]}(data, grid, bcs)
end

"""
    Field(X, Y, Z, arch, grid, [data=zeros(arch, grid)], bcs)

Construct a `Field` on some architecture `arch` and a `grid` with some `data`.
The field's location is defined by `X`, `Y`, `Z` where each is either `Cell` or `Face`
and has field boundary conditions `bcs`.
"""
Field(X, Y, Z, arch, grid, bcs, data=zeros(eltype(grid), arch, grid, (X, Y, Z))) =
    Field((X, Y, Z), arch, grid, bcs, data)

"""
    CellField([FT=eltype(grid)], arch::AbstractArchitecture, grid, bcs=TracerBoundaryConditions(grid), 
              data=zeros(FT, arch, grid, (Cell, Cell, Cell)))

Return a `Field{Cell, Cell, Cell}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function CellField(arch::AbstractArchitecture, grid, 
                   bcs=TracerBoundaryConditions(grid),
                   data=zeros(eltype(grid), arch, grid, (Cell, Cell, Cell)))

    return Field(Cell, Cell, Cell, arch, grid, bcs, data)
end

"""
    XFaceField([FT=eltype(grid)], arch::AbstractArchitecture, grid, bcs=UVelocityBoundaryConditions(grid), 
              data=zeros(FT, arch, grid, (Face, Cell, Cell)))

Return a `Field{Face, Cell, Cell}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function XFaceField(arch::AbstractArchitecture, grid, 
                    bcs=UVelocityBoundaryConditions(grid),
                    data=zeros(eltype(grid), arch, grid, (Face, Cell, Cell))) 

    return Field(Face, Cell, Cell, arch, grid, bcs, data)
end
  
"""
    YFaceField([FT=eltype(grid)], arch::AbstractArchitecture, grid, bcs=VVelocityBoundaryConditions(grid), 
              data=zeros(FT, arch, grid, (Cell, Face, Cell)))

Return a `Field{Cell, Face, Cell}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function YFaceField(arch::AbstractArchitecture, grid, 
                    bcs=VVelocityBoundaryConditions(grid), 
                    data=zeros(eltype(grid), arch, grid, (Cell, Face, Cell))) 

    return Field(Cell, Face, Cell, arch, grid, bcs, data)
end
  
"""
    ZFaceField([FT=eltype(grid)], arch::AbstractArchitecture, grid, bcs=WVelocityBoundaryConditions(grid), 
              data=zeros(FT, arch, grid, (Cell, Cell, Face))

Return a `Field{Cell, Cell, Face}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function ZFaceField(arch::AbstractArchitecture, grid, 
                    bcs=WVelocityBoundaryConditions(grid), 
                    data=zeros(eltype(grid), arch, grid, (Cell, Cell, Face))) 

    return Field(Cell, Cell, Face, arch, grid, bcs, data)
end

CellField(FT::DataType, arch, grid, bcs=TracerBoundaryConditions(grid)) = 
    CellField(arch, grid, bcs, zeros(FT, arch, grid, (Cell, Cell, Cell)))

XFaceField(FT::DataType, arch, grid, bcs=UVelocityBoundaryConditions(grid)) =
    XFaceField(arch, grid, bcs, zeros(FT, arch, grid, (Face, Cell, Cell)))

YFaceField(FT::DataType, arch, grid, bcs=VVelocityBoundaryConditions(grid)) =
    YFaceField(arch, grid, bcs, zeros(FT, arch, grid, (Cell, Face, Cell)))

ZFaceField(FT::DataType, arch, grid, bcs=WVelocityBoundaryConditions(grid)) =
    ZFaceField(arch, grid, bcs, zeros(FT, arch, grid, (Cell, Cell, Face)))

#####
##### Functions for querying fields
#####

location(a) = nothing

"Returns the location `(X, Y, Z)` of an `AbstractField{X, Y, Z}`."
location(::AbstractField{X, Y, Z}) where {X, Y, Z} = (X, Y, Z) # note no instantiation

x_location(::AbstractField{X}) where X               = X
y_location(::AbstractField{X, Y}) where {X, Y}       = Y
z_location(::AbstractField{X, Y, Z}) where {X, Y, Z} = Z

"Returns the architecture where the field data `f.data` is stored."
architecture(f::Field) = architecture(f.data)
architecture(o::OffsetArray) = architecture(o.parent)

"Returns the length of a field's `data`."    
@inline length(f::Field) = length(f.data)


"Returns the topology of a fields' `grid`."
@inline topology(f, args...) = topology(f.grid, args...)

"""
Returns the length of a field located at `Cell` centers along a grid 
dimension of length `N` and with halo points `H`.
"""
dimension_length(loc, topo, N, H=0) = N + 2H

"""
Returns the length of a field located at cell `Face`s along a grid 
dimension of length `N` and with halo points `H`.
"""
dimension_length(::Type{Face}, ::Bounded, N, H=0) = N + 1 + 2H

"""
    size(loc, grid)

Returns the size of a field at `loc` on `grid`.
This is a 3-tuple of integers corresponding to the number of interior nodes
of `f` along `x, y, z`.
"""
@inline size(loc, grid) = (dimension_length(loc[1], topology(grid, 1), grid.Nx), 
                           dimension_length(loc[2], topology(grid, 2), grid.Ny), 
                           dimension_length(loc[3], topology(grid, 3), grid.Nz))

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
@inline total_size(loc, grid) = (dimension_length(loc[1], topology(grid, 1), grid.Nx, grid.Hx), 
                                 dimension_length(loc[2], topology(grid, 2), grid.Ny, grid.Hy), 
                                 dimension_length(loc[3], topology(grid, 3), grid.Nz, grid.Hz))

@inline total_size(f::AbstractField) = total_size(location(f), f.grid)

@propagate_inbounds getindex(f::Field, inds...) = @inbounds getindex(f.data, inds...)
@propagate_inbounds setindex!(f::Field, v, inds...) = @inbounds setindex!(f.data, v, inds...)

@inline lastindex(f::Field) = lastindex(f.data)
@inline lastindex(f::Field, dim) = lastindex(f.data, dim)

"Returns `f.data` for `f::Field` or `f` for `f::AbstractArray."
@inline data(a) = a # fallback
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
@inline interior(f::Field{X, Y, Z}) where {X, Y, Z} = view(f.data, interior_indices(X, topology(f, 1), f.grid.Nx), 
                                                                   interior_indices(Y, topology(f, 2), f.grid.Ny), 
                                                                   interior_indices(Z, topology(f, 3), f.grid.Nz))

"Returns a reference (not a view) to the interior points of `field.data.parent.`"
@inline interiorparent(f::Field{X, Y, Z}) where {X, Y, Z} = 
    @inbounds f.data.parent[interior_parent_indices(X, topology(f, 1), f.grid.Nx, f.grid.Hx),
                            interior_parent_indices(Y, topology(f, 2), f.grid.Ny, f.grid.Hy),
                            interior_parent_indices(Z, topology(f, 3), f.grid.Nz, f.grid.Hz)]

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
xnodes(::Type{Cell}, topo, grid) = reshape(grid.xC, grid.Nx, 1, 1)
ynodes(::Type{Cell}, topo, grid) = reshape(grid.yC, 1, grid.Ny, 1)
znodes(::Type{Cell}, topo, grid) = reshape(grid.zC, 1, 1, grid.Nz)

xnodes(::Type{Face}, topo, grid) = reshape(grid.xF[1:end-1], grid.Nx, 1, 1)
ynodes(::Type{Face}, topo, grid) = reshape(grid.yF[1:end-1], 1, grid.Ny, 1)
znodes(::Type{Face}, topo, grid) = reshape(grid.zF[1:end-1], 1, 1, grid.Nz)

xnodes(::Type{Face}, ::Bounded, grid) = reshape(grid.xF, grid.Nx+1, 1, 1)
ynodes(::Type{Face}, ::Bounded, grid) = reshape(grid.yF, 1, grid.Ny+1, 1)
znodes(::Type{Face}, ::Bounded, grid) = reshape(grid.zF, 1, 1, grid.Nz+1)

xnodes(ψ::AbstractField) = xnodes(x_location(ψ), topology(ψ, 1), ψ.grid)
ynodes(ψ::AbstractField) = ynodes(y_location(ψ), topology(ψ, 2), ψ.grid)
znodes(ψ::AbstractField) = znodes(z_location(ψ), topology(ψ, 3), ψ.grid)

nodes(ψ::AbstractField) = (xnodes(ψ), ynodes(ψ), znodes(ψ))

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
offset_indices(::Type{Face}, ::Bounded, N, H=0) = 1 - H : N + H + 1

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
    underlying_data = zeros(FT, dimension_length(loc[1], topology(grid, 1), grid.Nx, grid.Hx),
                                dimension_length(loc[2], topology(grid, 2), grid.Ny, grid.Hy),
                                dimension_length(loc[3], topology(grid, 3), grid.Nz, grid.Hz))

    return OffsetArray(underlying_data, grid, loc)
end

"""
    zeros([FT=Float64], ::GPU, grid, loc)

Returns an `OffsetArray` of zeros of float type `FT`, with
parent data in GPU memory and indices corresponding to a field on a `grid`
of `size(grid)` and located at `loc`.
"""
function Base.zeros(FT, ::GPU, grid, loc=(Cell, Cell, Cell))
    underlying_data = CuArray{FT}(undef, dimension_length(loc[1], topology(grid, 1), grid.Nx, grid.Hx),
                                         dimension_length(loc[2], topology(grid, 2), grid.Ny, grid.Hy),
                                         dimension_length(loc[3], topology(grid, 3), grid.Nz, grid.Hz))

    underlying_data .= 0 # Ensure data is initially 0.

    return OffsetArray(underlying_data, grid, loc)
end

# Default to type of Grid
Base.zeros(arch, grid, loc=(Cell, Cell, Cell)) = zeros(eltype(grid), arch, grid, loc)

Base.zeros(FT, ::CPU, grid, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz)
Base.zeros(FT, ::GPU, grid, Nx, Ny, Nz) = zeros(FT, Nx, Ny, Nz) |> CuArray

Base.zeros(arch, grid, args...) = zeros(eltype(grid), args...)
