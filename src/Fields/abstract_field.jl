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
import Oceananigans.Architectures: architecture, child_architecture
import Oceananigans.Grids: interior_x_indices, interior_y_indices, interior_z_indices
import Oceananigans.Grids: total_size, topology, nodes, xnodes, ynodes, znodes, node, xnode, ynode, znode
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
@inline location(a, i) = location(a)[i]
@inline location(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = (LX, LY, LZ) # note no instantiation
@inline instantiated_location(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = (LX(), LY(), LZ())
Base.eltype(::AbstractField{<:Any, <:Any, <:Any, <:Any, T}) where T = T

"Returns the architecture of on which `f` is defined."
architecture(f::AbstractField) = architecture(f.grid)
child_architecture(f::AbstractField) = child_architecture(architecture(f))

"Returns the topology of a fields' `grid`."
@inline topology(f::AbstractField, args...) = topology(f.grid, args...)

"""
    size(f::AbstractField)

Returns the size of an `AbstractField{LX, LY, LZ}` located at `LX, LY, LZ`.
This is a 3-tuple of integers corresponding to the number of interior nodes
of `f` along `x, y, z`.
"""
Base.size(f::AbstractField) = size(f.grid, location(f))
Base.length(f::AbstractField) = prod(size(f))
Base.parent(f::AbstractField) = f

const Abstract3DField = AbstractField{<:Any, <:Any, <:Any, <:Any, <:Any, 3}
const Abstract4DField = AbstractField{<:Any, <:Any, <:Any, <:Any, <:Any, 4}

# TODO: to omit boundaries on Face fields, we have to return 2:N
# when topo=Bounded, and loc=Face
@inline axis(::Colon, N) = Base.OneTo(N)
@inline axis(index::UnitRange, N) = index

@inline function Base.axes(f::Abstract3DField)
    Nx, Ny, Nz = size(f)
    ix, iy, iz = indices(f)

    ax = axis(ix, Nx)
    ay = axis(iy, Ny)
    az = axis(iz, Nz)

    return (ax, ay, az)
end

@inline function Base.axes(f::Abstract4DField)
    Nx, Ny, Nz, Nt = size(f)
    ix, iy, iz = indices(f)

    ax = axis(ix, Nx)
    ay = axis(iy, Ny)
    az = axis(iz, Nz)
    at = Base.OneTo(Nt)

    return (ax, ay, az, at)
end



"""
    total_size(field::AbstractField)

Returns a 3-tuple that gives the "total" size of a field including
both interior points and halo points.
"""
total_size(f::AbstractField) = total_size(f.grid, location(f))

interior(f::AbstractField) = f

#####
##### Coordinates of fields
#####

@propagate_inbounds node(i, j, k, ψ::AbstractField) = node(i, j, k, ψ.grid, instantiated_location(ψ)...)
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

