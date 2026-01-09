module OceananigansConservativeRegriddingExt

using ConservativeRegridding
using Oceananigans
using Oceananigans.Architectures: architecture, array_type
using Oceananigans.Fields: AbstractField, location
using Oceananigans.Grids: ξnode, ηnode
using Oceananigans.Utils: launch!

using KernelAbstractions: @index, @kernel

import Oceananigans.Fields: regrid!
import ConservativeRegridding: Regridder

#####
##### Cell matrix computation
#####

instantiate(L) = L()

flip(::Face) = Center()
flip(::Center) = Face()

left_index(i, ::Center) = i
left_index(i, ::Face) = i - 1
right_index(i, ::Center) = i + 1
right_index(i, ::Face) = i

@kernel function _compute_cell_vertices!(cell_vertices, Fx, ℓx, ℓy, grid)
    i, j = @index(Global, NTuple)

    vx = flip(ℓx)
    vy = flip(ℓy)

    isw = left_index(i, ℓx)
    jsw = left_index(j, ℓy)

    inw = left_index(i, ℓx)
    jnw = right_index(j, ℓy)

    ine = right_index(i, ℓx)
    jne = right_index(j, ℓy)

    ise = right_index(i, ℓx)
    jse = left_index(j, ℓy)

    xsw = ξnode(isw, jsw, 1, grid, vx, vy, nothing)
    ysw = ηnode(isw, jsw, 1, grid, vx, vy, nothing)

    xnw = ξnode(inw, jnw, 1, grid, vx, vy, nothing)
    ynw = ηnode(inw, jnw, 1, grid, vx, vy, nothing)

    xne = ξnode(ine, jne, 1, grid, vx, vy, nothing)
    yne = ηnode(ine, jne, 1, grid, vx, vy, nothing)

    xse = ξnode(ise, jse, 1, grid, vx, vy, nothing)
    yse = ηnode(ise, jse, 1, grid, vx, vy, nothing)

    linear_idx = i + (j - 1) * Fx
    @inbounds begin
        cell_vertices[1, linear_idx] = (xsw, ysw)
        cell_vertices[2, linear_idx] = (xnw, ynw)
        cell_vertices[3, linear_idx] = (xne, yne)
        cell_vertices[4, linear_idx] = (xse, yse)
        cell_vertices[5, linear_idx] = (xsw, ysw)
    end
end

"""
    compute_cell_vertices(field::AbstractField)

Compute a matrix of cell vertices for `field`, suitable for use with
`ConservativeRegridding.Regridder`.

Returns a matrix of size `(5, Nx*Ny)` where each column contains the vertices
of a cell in the order `[sw, nw, ne, se, sw]` (closing the polygon).
"""
function compute_cell_vertices(field::AbstractField)
    Fx, Fy, _ = size(field)
    LX, LY, _ = location(field)
    ℓx, ℓy = LX(), LY()

    if isnothing(ℓx) || isnothing(ℓy)
        error("cell_vertices can only be computed for fields with non-nothing horizontal location.")
    end

    grid = field.grid
    arch = architecture(grid)
    FT = eltype(grid)

    vertices_per_cell = 5 # convention: [sw, nw, ne, se, sw]
    ArrayType = array_type(arch)
    cell_vertices = ArrayType{Tuple{FT, FT}}(undef, vertices_per_cell, Fx * Fy)

    launch!(arch, grid, (Fx, Fy), _compute_cell_vertices!, cell_vertices, Fx, ℓx, ℓy, grid)

    return cell_vertices
end

#####
##### Regridder construction from Oceananigans fields
#####

"""
    Regridder(dst_field::AbstractField, src_field::AbstractField; kwargs...)

Construct a `ConservativeRegridding.Regridder` for regridding from `src_field` to `dst_field`.

The regridder computes intersection areas between cells of the source and destination grids,
enabling conservative (mean-preserving) regridding.

Keyword arguments are passed to `ConservativeRegridding.Regridder`.
"""
function Regridder(dst_field::AbstractField, src_field::AbstractField; kwargs...)
    dst_vertices = compute_cell_vertices(dst_field)
    src_vertices = compute_cell_vertices(src_field)
    return ConservativeRegridding.Regridder(dst_vertices, src_vertices; kwargs...)
end

#####
##### Regridding Oceananigans fields
#####

"""
    regrid!(dst_field::AbstractField, regridder::ConservativeRegridding.Regridder, src_field::AbstractField)

Regrid `src_field` onto `dst_field` using the conservative `regridder`.

This performs mean-preserving regridding, where the regridded field values are
weighted by the intersection areas between source and destination grid cells.
"""
function regrid!(dst_field::AbstractField, regridder::ConservativeRegridding.Regridder, src_field::AbstractField)
    dst_data = vec(interior(dst_field))
    src_data = vec(interior(src_field))
    ConservativeRegridding.regrid!(dst_data, regridder, src_data)
    return dst_field
end

end # module

