using Base: @propagate_inbounds
using CUDA
using OffsetArrays
using Oceananigans.Architectures
using Oceananigans.Utils
using Oceananigans.Grids: interior_indices, interior_parent_indices

import Oceananigans.Utils: datatuple
import Oceananigans.Architectures: architecture
import Oceananigans.Grids: total_size, topology, nodes, xnodes, ynodes, znodes, xnode, ynode, znode

"""
    AbstractField{X, Y, Z, A, G}

Abstract supertype for fields located at `(X, Y, Z)` with data stored in a container
of type `A`. The field is defined on a grid `G`.
"""
abstract type AbstractField{X, Y, Z, A, G} end

function validate_field_data(X, Y, Z, data, grid)
    Tx, Ty, Tz = total_size((X, Y, Z), grid)

    if size(data) != (Tx, Ty, Tz)
        e = "Cannot construct field at ($X, $Y, $Z) with size(data)=$(size(data)). " *
            "`data` must have size ($Tx, $Ty, $Tz)."
        throw(ArgumentError(e))
    end

    return nothing
end

#####
##### AbstractField functionality
#####

# Overload compute! for custom fields to produce non-default behavior
compute!(f::AbstractField) = nothing

@inline location(a) = nothing

"Returns the location `(X, Y, Z)` of an `AbstractField{X, Y, Z}`."
@inline location(::AbstractField{X, Y, Z}) where {X, Y, Z} = (X, Y, Z) # note no instantiation
@inline location(f, i) = location(f)[i]

"Returns the architecture where the field data `f.data` is stored."
architecture(f::AbstractField) = architecture(f.data)
architecture(o::OffsetArray) = architecture(o.parent)

"Returns the length of a field's `data`."
@inline Base.length(f::AbstractField) = length(f.data)

"Returns the topology of a fields' `grid`."
@inline topology(f, args...) = topology(f.grid, args...)

"""
    size(f::AbstractField)

Returns the size of an `AbstractField{X, Y, Z}` located at `X, Y, Z`.
This is a 3-tuple of integers corresponding to the number of interior nodes
of `f` along `x, y, z`.
"""
Base.size(f::AbstractField) = size(location(f), f.grid)

"""
    total_size(field::AbstractField)

Returns a 3-tuple that gives the "total" size of a field including 
both interior points and halo points.
"""
total_size(f::AbstractField) = total_size(location(f), f.grid)

# Endpoint for recursive `datatuple` function:
@inline datatuple(obj::AbstractField) = data(obj)

"Returns `f.data.parent` for `f::Field`."
@inline Base.parent(f::AbstractField) = f.data.parent

"Returns a view of `f` that excludes halo points."
@inline interior(f::AbstractField{X, Y, Z}) where {X, Y, Z} =
    view(f.data, interior_indices(X, topology(f, 1), f.grid.Nx),
                 interior_indices(Y, topology(f, 2), f.grid.Ny),
                 interior_indices(Z, topology(f, 3), f.grid.Nz))

"Returns a reference (not a view) to the interior points of `field.data.parent.`"
@inline interiorparent(f::AbstractField{X, Y, Z}) where {X, Y, Z} =
    @inbounds f.data.parent[interior_parent_indices(X, topology(f, 1), f.grid.Nx, f.grid.Hx),
                            interior_parent_indices(Y, topology(f, 2), f.grid.Ny, f.grid.Hy),
                            interior_parent_indices(Z, topology(f, 3), f.grid.Nz, f.grid.Hz)]

Base.iterate(f::AbstractField, state=1) = iterate(f.data, state)

@inline xnode(i, ψ::AbstractField{X, Y, Z}) where {X, Y, Z} = xnode(X, i, ψ.grid)
@inline ynode(j, ψ::AbstractField{X, Y, Z}) where {X, Y, Z} = ynode(Y, j, ψ.grid)
@inline znode(k, ψ::AbstractField{X, Y, Z}) where {X, Y, Z} = znode(Z, k, ψ.grid)

@hascuda @inline cpudata(f::AbstractField{X, Y, Z, <:OffsetCuArray}) where {X, Y, Z} =
    OffsetArray(Array(parent(f)), f.grid, location(f))

@inline Base.lastindex(f::AbstractField) = lastindex(f.data)
@inline Base.lastindex(f::AbstractField, dim) = lastindex(f.data, dim)

"Returns `f.data` for `f::Field` or `f` for `f::AbstractArray."
@inline data(a) = a # fallback
@inline data(f::AbstractField) = f.data

@inline cpudata(a) = data(a)

const OffsetCuArray = OffsetArray{T, D, <:CuArray} where {T, D}

xnodes(ψ::AbstractField) = xnodes(location(ψ, 1), ψ.grid)
ynodes(ψ::AbstractField) = ynodes(location(ψ, 2), ψ.grid)
znodes(ψ::AbstractField) = znodes(location(ψ, 3), ψ.grid)

nodes(ψ::AbstractField; kwargs...) = nodes(location(ψ), ψ.grid; kwargs...)
