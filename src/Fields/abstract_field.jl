using Base: @propagate_inbounds
using CUDA
using Adapt
using OffsetArrays
using Statistics

using Oceananigans.Architectures
using Oceananigans.Utils
using Oceananigans.Grids: interior_indices, interior_parent_indices

import Base: minimum, maximum, extrema
import Statistics: mean
import Oceananigans: location, instantiated_location
import Oceananigans.Architectures: architecture
import Oceananigans.Grids: interior_x_indices, interior_y_indices, interior_z_indices
import Oceananigans.Grids: total_size, topology, nodes, xnodes, ynodes, znodes, xnode, ynode, znode
import Oceananigans.Utils: datatuple

const ArchOrNothing = Union{AbstractArchitecture, Nothing}
const GridOrNothing = Union{AbstractGrid, Nothing}

"""
    AbstractField{X, Y, Z, A, G, T, N}

Abstract supertype for fields located at `(X, Y, Z)` on architecture `A`
and defined on a grid `G` with eltype `T` and `N` dimensions.
"""
abstract type AbstractField{X, Y, Z, A <: ArchOrNothing, G <: GridOrNothing, T, N} <: AbstractArray{T, N} end
abstract type AbstractOperation{X, Y, Z, A, G, T} <: AbstractField{X, Y, Z, A, G, T, 3} end

Base.IndexStyle(::AbstractField) = IndexCartesian()

function validate_field_data(loc, data, grid)
    Tx, Ty, Tz = total_size(loc, grid)

    if size(data) != (Tx, Ty, Tz)
        LX, LY, LZ = loc    
        e = "Cannot construct field at ($LX, $LY, $LZ) with size(data)=$(size(data)). " *
            "`data` must have size ($Tx, $Ty, $Tz)."
        throw(ArgumentError(e))
    end

    return nothing
end

# Endpoint for recursive `datatuple` function:
@inline datatuple(obj::AbstractField) = data(obj)


#####
##### AbstractField functionality
#####

@inline location(a) = (Nothing, Nothing, Nothing)

"Returns the location `(LX, LY, LZ)` of an `AbstractField{LX, LY, LZ}`."
@inline location(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = (LX, LY, LZ) # note no instantiation
@inline location(f, i) = location(f)[i]

@inline instantiated_location(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = (LX(), LY(), LZ())

"Returns the architecture of on which `f` is defined."
architecture(f::AbstractField) = architecture(f.grid)

"Returns the topology of a fields' `grid`."
@inline topology(f, args...) = topology(f.grid, args...)

"""
    size(f::AbstractField)

Returns the size of an `AbstractField{X, Y, Z}` located at `X, Y, Z`.
This is a 3-tuple of integers corresponding to the number of interior nodes
of `f` along `x, y, z`.
"""
Base.size(f::AbstractField) = size(location(f), f.grid)

"Returns the length of a field's `data`."
@inline Base.length(f::AbstractField) = prod(size(f))

"""
    total_size(field::AbstractField)

Returns a 3-tuple that gives the "total" size of a field including
both interior points and halo points.
"""
total_size(f::AbstractField) = total_size(location(f), f.grid)

#####
##### Accessing wrapped arrays
#####

"Returns `f.data` for `f::Field` or `f` for `f::AbstractArray."
data(a) = nothing # fallback
cpudata(a) = data(a)

cpudata(f::AbstractField{X, Y, Z, <:GPU}) where {X, Y, Z} =
    offset_data(Array(parent(f)), f.grid, location(f))

"Returns `f.data.parent` for `f::Field`."
@inline Base.parent(f::AbstractField) = parent(data(f))

@inline interior(f::AbstractField) = f

@inline interior_copy(f::AbstractField{X, Y, Z}) where {X, Y, Z} =
    parent(f)[interior_parent_indices(X, topology(f, 1), f.grid.Nx, f.grid.Hx),
              interior_parent_indices(Y, topology(f, 2), f.grid.Ny, f.grid.Hy),
              interior_parent_indices(Z, topology(f, 3), f.grid.Nz, f.grid.Hz)]

#####
##### Coordinates of fields
#####

@propagate_inbounds xnode(i, ψ::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = xnode(LX(), i, ψ.grid)
@propagate_inbounds ynode(j, ψ::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = ynode(LY(), j, ψ.grid)
@propagate_inbounds znode(k, ψ::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = znode(LZ(), k, ψ.grid)

xnodes(ψ::AbstractField) = xnodes(location(ψ, 1), ψ.grid)
ynodes(ψ::AbstractField) = ynodes(location(ψ, 2), ψ.grid)
znodes(ψ::AbstractField) = znodes(location(ψ, 3), ψ.grid)

nodes(ψ::AbstractField; kwargs...) = nodes(location(ψ), ψ.grid; kwargs...)

#####
##### Some conveniences
#####

for f in (:+, :-)
    @eval Base.$f(ϕ::AbstractArray, ψ::AbstractField) = $f(ϕ, interior(ψ))
    @eval Base.$f(ϕ::AbstractField, ψ::AbstractArray) = $f(interior(ϕ), ψ)
end

function Statistics.norm(a::AbstractField)
    arch = architecture(a)
    grid = a.grid
    r = zeros(arch, grid, 1)
    Base.mapreducedim!(x -> x * x, +, r, a)
    return CUDA.@allowscalar sqrt(r[1])
end

