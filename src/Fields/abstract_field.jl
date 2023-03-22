using Base: @propagate_inbounds
using CUDA
using Adapt
using OffsetArrays
using Statistics

using Oceananigans.Architectures
using Oceananigans.Utils
using Oceananigans.Grids: interior_indices, interior_parent_indices

import Base: minimum, maximum, extrema
import Oceananigans: location, instantiated_location
import Oceananigans.Architectures: architecture
import Oceananigans.Grids: interior_x_indices, interior_y_indices, interior_z_indices
import Oceananigans.Grids: total_size, topology, nodes, xnodes, ynodes, znodes, xnode, ynode, znode
import Oceananigans.Utils: datatuple

const ArchOrNothing = Union{AbstractArchitecture, Nothing}
const GridOrNothing = Union{AbstractGrid, Nothing}

"""
    AbstractField{LX, LY, LZ, G, T, N}

Abstract supertype for fields located at `(LX, LY, LZ)`
and defined on a grid `G` with eltype `T` and `N` dimensions.

Note: we need the parameter `T` to subtype AbstractArray.
"""
abstract type AbstractField{LX, LY, LZ, G <: GridOrNothing, T, N} <: AbstractArray{T, N} end

Base.IndexStyle(::AbstractField) = IndexCartesian()

#####
##### AbstractField functionality
#####

"Returns the location `(LX, LY, LZ)` of an `AbstractField{LX, LY, LZ}`."
@inline location(a) = (Nothing, Nothing, Nothing) # used in AbstractOperations for location inference
@inline location(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = (LX, LY, LZ) # note no instantiation
@inline location(f::AbstractField, i) = location(f)[i]
@inline instantiated_location(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = (LX(), LY(), LZ())

"Returns the architecture of on which `f` is defined."
architecture(f::AbstractField) = architecture(f.grid)

"Returns the topology of a fields' `grid`."
@inline topology(f::AbstractField, args...) = topology(f.grid, args...)

"""
    size(f::AbstractField)

Returns the size of an `AbstractField{LX, LY, LZ}` located at `LX, LY, LZ`.
This is a 3-tuple of integers corresponding to the number of interior nodes
of `f` along `x, y, z`.
"""
Base.size(f::AbstractField) = size(location(f), f.grid)
Base.length(f::AbstractField) = prod(size(f))
Base.parent(f::AbstractField) = f

"""
    total_size(field::AbstractField)

Returns a 3-tuple that gives the "total" size of a field including
both interior points and halo points.
"""
total_size(f::AbstractField) = total_size(location(f), f.grid)

interior(f::AbstractField) = f

#####
##### Coordinates of fields
#####

@propagate_inbounds xnode(i, j, k, ψ::AbstractField) = xnode(i, j, k, ψ.grid, instantiated_location(ψ)...)
@propagate_inbounds ynode(i, j, k, ψ::AbstractField) = ynode(i, j, k, ψ.grid, instantiated_location(ψ)...)
@propagate_inbounds znode(i, j, k, ψ::AbstractField) = znode(i, j, k, ψ.grid, instantiated_location(ψ)...)

xnodes(ψ::AbstractField; kwargs...) = xnodes(ψ.grid, instantiated_location(ψ)...; kwargs...)
ynodes(ψ::AbstractField; kwargs...) = ynodes(ψ.grid, instantiated_location(ψ)...; kwargs...)
znodes(ψ::AbstractField; kwargs...) = znodes(ψ.grid, instantiated_location(ψ)...; kwargs...)

nodes(ψ::AbstractField; kwargs...) = nodes(ψ.grid, instantiated_location(ψ); kwargs...)

#####
##### Some conveniences
#####

for f in (:+, :-)
    @eval Base.$f(ϕ::AbstractArray, ψ::AbstractField) = $f(ϕ, interior(ψ))
    @eval Base.$f(ϕ::AbstractField, ψ::AbstractArray) = $f(interior(ϕ), ψ)
end

