module OceananigansConservativeRegriddingExt

using ConservativeRegridding
import ConservativeRegridding: Regridder, regrid!

using Oceananigans
using Oceananigans.Grids: ξnode, ηnode, architecture
using Oceananigans.Fields: AbstractField
using KernelAbstractions: @index, @kernel

# TODO: extend regridding to more cases
const RegriddableField{LX, LY} = Field{LX, LY, Nothing}

instantiate(L) = L()

function compute_cell_matrix(field::AbstractField)
    LX, LY, _ = Oceananigans.Fields.location(field)
    return compute_cell_matrix(field.grid, LX(), LY(), field.indices)
end

function compute_cell_matrix(grid::AbstractGrid, ℓx, ℓy, indices=default_indices(3))

    if isnothing(ℓx) || isnothing(ℓy)
        throw(ArgumentError("cell_matrix can only be computed for fields with non-nothing horizontal location."))
    end

    arch = architecture(grid)
    FT = eltype(grid)

    Fx = size(grid, ℓx, indices)
    Fy = size(grid, ℓy, indices)
    
    vertices_per_cell = 5 # convention: [sw, nw, ne, se, sw]
    ArrayType = Oceananigans.Architectures.array_type(arch)
    cell_matrix = ArrayType{Tuple{FT, FT}}(undef, vertices_per_cell, Fx*Fy)

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

function Regridder(dst_field::RegriddableField, src_field::RegriddableField)
    src_cells = compute_cell_matrix(src_field)
    dst_cells = compute_cell_matrix(dst_field)

    # TODO: eliminate this memory allocation
    # src_polygons = GI.Polygon.(GI.LinearRing.(eachcol(src_cells))) .|> GO.fix
    # dst_polygons = GI.Polygon.(GI.LinearRing.(eachcol(dst_cells))) .|> GO.fix
    #return ConservativeRegridding.Regridder(src_polygons, dst_polygons)

    return ConservativeRegridding.Regridder(src_cells, dst_cells)
end

"""
    regrid!(a, b)

    regrid!(a, regridder, b)

Regrid field `b` onto the grid of field `a`. 

Example
=======

Generate a tracer field on a vertically stretched grid and regrid it on a regular grid.

```jldoctest
using Oceananigans

Nz, Lz = 2, 1.0
topology = (Flat, Flat, Bounded)

input_grid = RectilinearGrid(size=Nz, z = [0, Lz/3, Lz], topology=topology, halo=1)
input_field = CenterField(input_grid)
input_field[1, 1, 1:Nz] = [2, 3]

output_grid = RectilinearGrid(size=Nz, z=(0, Lz), topology=topology, halo=1)
output_field = CenterField(output_grid)

regrid!(output_field, input_field)

output_field[1, 1, :]

# output
4-element OffsetArray(::Vector{Float64}, 0:3) with eltype Float64 with indices 0:3:
 0.0
 2.333333333333333
 3.0
 0.0
```
"""
function regrid!(dst_field::RegriddableField,
                 regridder::ConservativeRegridding.Regridder,
                 src_field::RegriddableField)

    dst_vec = vec(interior(dst_field))
    src_vec = vec(interior(src_field))

    return regrid!(dst_vec, regridder, src_vec)
end

function regrid!(dst_field::RegriddableField, src_field::RegriddableField)
    regridder = ConservativeRegridding.Regridder(dst_field, src_field)
    return regrid!(dst_field, regridder, src_field)
end

end # OceananigansConservativeRegriddingExt
