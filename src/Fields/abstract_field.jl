using Base: @propagate_inbounds
using CUDA
using Adapt
using OffsetArrays

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

"""
    AbstractField{X, Y, Z, A, G}

Abstract supertype for fields located at `(X, Y, Z)` with data stored in a container
of type `A`. The field is defined on a grid `G`.
"""
abstract type AbstractField{X, Y, Z,
                            A <: Union{AbstractArchitecture, Nothing},
                            G <: Union{AbstractGrid, Nothing}} end

function validate_field_data(X, Y, Z, data, grid)
    Tx, Ty, Tz = total_size((X, Y, Z), grid)

    if size(data) != (Tx, Ty, Tz)
        e = "Cannot construct field at ($X, $Y, $Z) with size(data)=$(size(data)). " *
            "`data` must have size ($Tx, $Ty, $Tz)."
        throw(ArgumentError(e))
    end

    return nothing
end

# Endpoint for recursive `datatuple` function:
@inline datatuple(obj::AbstractField) = data(obj)

#####
##### Computing AbstractField
#####

# Note: overload compute! for custom fields to produce non-default behavior

"""
    compute!(field)

Computes `field.data`.
"""
compute!(field) = nothing

"""
    compute_at!(field, time)

Computes `field.data` at `time`. Falls back to compute!(field).
"""
compute_at!(field, time) = compute!(field)

mutable struct FieldStatus{T}
    time :: T
end

Adapt.adapt_structure(to, status::FieldStatus) = (time = status.time,)

"""
    conditional_compute!(field, time)

Computes `field.data` if `time != field.status.time`.
"""
function conditional_compute!(field, time)

    if time == zero(time) || time != field.status.time
        compute!(field, time)
        field.status.time = time
    end

    return nothing
end

# This edge case occurs if `fetch_output` is called with `model::Nothing`.
# We do the safe thing here and always compute.
conditional_compute!(field, ::Nothing) = compute!(field, nothing)

"""
    @compute(exprs...)

Call compute! on fields after defining them.
"""
macro compute(def)
    expr = Expr(:block)
    field = def.args[1]
    push!(expr.args, :($(esc(def))))
    push!(expr.args, :(compute!($(esc(field)))))
    return expr
end

#####
##### AbstractField functionality
#####

@inline location(a) = nothing

"Returns the location `(X, Y, Z)` of an `AbstractField{X, Y, Z}`."
@inline location(::AbstractField{X, Y, Z}) where {X, Y, Z} = (X, Y, Z) # note no instantiation
@inline location(f, i) = location(f)[i]

@inline instantiated_location(::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = (LX(), LY(), LZ())

"Returns the architecture where the field data `f.data` is stored."
architecture(f::AbstractField) = f.architecture

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

#####
##### Accessing wrapped arrays
#####

"Returns `f.data` for `f::Field` or `f` for `f::AbstractArray."
@inline data(a) = a # fallback
@inline data(f::AbstractField) = f.data

@inline cpudata(a) = data(a)

@inline cpudata(f::AbstractField{X, Y, Z, <:AbstractGPUArchitecture}) where {X, Y, Z} =
    offset_data(Array(parent(f)), f.grid, location(f))

"Returns `f.data.parent` for `f::Field`."
@inline Base.parent(f::AbstractField) = parent(data(f))

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

#####
##### getindex
#####

@propagate_inbounds Base.getindex(f::AbstractField, inds...) = @inbounds getindex(f.data, inds...)

#####
##### setindex
#####

@propagate_inbounds Base.setindex!(f::AbstractField, a, inds...) = @inbounds setindex!(f.data, a, inds...)

#####
##### Coordinates of fields
#####

@inline xnode(i, ψ::AbstractField{X, Y, Z}) where {X, Y, Z} = xnode(X, i, ψ.grid)
@inline ynode(j, ψ::AbstractField{X, Y, Z}) where {X, Y, Z} = ynode(Y, j, ψ.grid)
@inline znode(k, ψ::AbstractField{X, Y, Z}) where {X, Y, Z} = znode(Z, k, ψ.grid)

@inline Base.lastindex(f::AbstractField) = lastindex(f.data)
@inline Base.lastindex(f::AbstractField, dim) = lastindex(f.data, dim)

xnodes(ψ::AbstractField) = xnodes(location(ψ, 1), ψ.grid)
ynodes(ψ::AbstractField) = ynodes(location(ψ, 2), ψ.grid)
znodes(ψ::AbstractField) = znodes(location(ψ, 3), ψ.grid)

nodes(ψ::AbstractField; kwargs...) = nodes(location(ψ), ψ.grid; kwargs...)

Base.iterate(f::AbstractField, state=1) = iterate(f.data, state)

#####
##### Field reductions
#####

"""
    minimum(field::AbstractField; dims=:)

Compute the minimum value of an Oceananigans `field` over the given dimensions (not including halo points).
By default all dimensions are included.
"""
minimum(field::AbstractField; dims=:) = minimum(interiorparent(field); dims=dims)

"""
    minimum(f, field::AbstractField; dims=:)

Returns the smallest result of calling the function `f` on each element of an Oceananigans `field`
(not including halo points) over the given dimensions. By default all dimensions are included.
"""
minimum(f, field::AbstractField; dims=:) = minimum(f, interiorparent(field); dims=dims)

"""
    maximum(field::AbstractField; dims=:)

Compute the maximum value of an Oceananigans `field` over the given dimensions (not including halo points).
By default all dimensions are included.
"""
maximum(field::AbstractField; dims=:) = maximum(interiorparent(field); dims=dims)

"""
    maximum(f, field::AbstractField; dims=:)

Returns the largest result of calling the function `f` on each element of an Oceananigans `field`
(not including halo points) over the given dimensions. By default all dimensions are included.
"""
maximum(f, field::AbstractField; dims=:) = maximum(f, interiorparent(field); dims=dims)

"""
    mean(field::AbstractField; dims=:)

Compute the mean of an Oceananigans `field` over the given dimensions (not including halo points).
By default all dimensions are included.
"""
mean(field::AbstractField; dims=:) = mean(interiorparent(field); dims=dims)

"""
    mean(f::Function, field::AbstractField; dims=:)

Apply the function `f` to each element of an Oceananigans `field` and take the mean over dimensions `dims`
(not including halo points). By default all dimensions are included.
"""
mean(f::Function, field::AbstractField; dims=:) = mean(f, interiorparent(field); dims=dims)
