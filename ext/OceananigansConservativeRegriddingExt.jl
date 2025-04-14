module OceananigansConservativeRegriddingExt

import GeoInterface as GI, GeometryOps as GO

using ConservativeRegridding

using Oceananigans
using Oceananigans.Grids: ξnode, ηnode
using Oceananigans.Fields: AbstractField
using KernelAbstractions: @index, @kernel

instantiate(L) = L()

function compute_cell_matrix(field::AbstractField)
    Fx, Fy, _ = size(field)
    LX, LY, _ = Oceananigans.Fields.location(field)
    ℓx, ℓy = LX(), LY()

    if isnothing(ℓx) || isnothing(ℓy)
        throw(ArgumentError("cell_matrix can only be computed for fields with non-nothing horizontal location."))
    end

    grid = field.grid
    arch = grid.architecture
    FT = eltype(grid)

    vertices_per_cell = 5 # convention: [sw, nw, ne, se, sw]
    ArrayType = Oceananigans.Architectures.array_type(arch)
    cell_matrix = ArrayType{Tuple{FT, FT}}(undef, vertices_per_cell, Fx*Fy)

    arch = grid.architecture
    Oceananigans.Utils.launch!(arch, grid, (Fx, Fy), _compute_cell_matrix!, cell_matrix, Fx, ℓx, ℓy, grid)

    return cell_matrix
end

flip(::Face) = Center()
flip(::Center) = Face()

left_index(i, ::Center) = i
left_index(i, ::Face) = i-1
right_index(i, ::Center) = i + 1
right_index(i, ::Face) = i

@kernel function _compute_cell_matrix!(cell_matrix, Fx, ℓx, ℓy, grid)
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
        cell_matrix[1, linear_idx] = (xsw, ysw)
        cell_matrix[2, linear_idx] = (xnw, ynw)
        cell_matrix[3, linear_idx] = (xne, yne)
        cell_matrix[4, linear_idx] = (xse, yse)
        cell_matrix[5, linear_idx] = (xsw, ysw)
    end
end

function ConservativeRegridding.Regridder(dst_field, src_field)
    src_cells = compute_cell_matrix(src_field)
    dst_cells = compute_cell_matrix(dst_field)

    src_polygons = GI.Polygon.(GI.LinearRing.(eachcol(src_cells))) .|> GO.fix
    dst_polygons = GI.Polygon.(GI.LinearRing.(eachcol(dst_cells))) .|> GO.fix

    return ConservativeRegridding.Regridder(src_polygons, dst_polygons)
end

ConservativeRegridding.regrid!(dst_field::Field, regridder::ConservativeRegridding.regridder, src_field::AbstractField) =
    regrid!(vec(interior(dst_field)), regridder, vec(interior(src_field)))

function ConservativeRegridding.regrid!(dst_field::Field, src_field::AbstractField)
    regridder = ConservativeRegridding.Regridder(dst_field, src_field)
    return regrid!(dst_field), regridder, src_field)
end

end # OceananigansConservativeRegriddingExt
