using Oceananigans.Fields: Field, fill_halo_regions!
using Oceananigans.Grids: Grids, AbstractStaticGrid, constructor_arguments, XFlatGrid, YFlatGrid
using Oceananigans.Utils: prettysummary, KernelParameters

import Oceananigans.Grids: peripheral_node
import Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶜᶜᶠ, Δrᶜᶠᶜ, Δrᶜᶠᶠ, Δrᶠᶜᶜ, Δrᶠᶜᶠ, Δrᶠᶠᶜ, Δrᶠᶠᶠ,
                               Δzᶜᶜᶜ, Δzᶜᶜᶠ, Δzᶜᶠᶜ, Δzᶜᶠᶠ, Δzᶠᶜᶜ, Δzᶠᶜᶠ, Δzᶠᶠᶜ, Δzᶠᶠᶠ

#####
##### CutCellBottom
#####

struct CutCellBottom{H, W, S, C, E, K} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    west_bottom_height :: W
    south_bottom_height :: S
    corner_bottom_height :: C
    minimum_fractional_cell_height :: E
    bottom_cell_index :: K
end

const CCBIBG{FT, TX, TY, TZ} = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:Any, <:CutCellBottom} where {FT, TX, TY, TZ}

XFlatCCBIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:CutCellBottom}
YFlatCCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:CutCellBottom}
XYFlatCCBIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Flat, <:Any, <:Any, <:CutCellBottom}

# Vertically-static, cut cell bottom, immersed boundary grid
VSCCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractStaticGrid, <:CutCellBottom}

"""
    CutCellBottom(bottom_height; minimum_fractional_cell_height=0.2)

Return a `CutCellBottom` representing an immersed boundary with cut bottom cells, inspired by
the shaved cells of [Adcroft et al. (1997)](@cite AdcroftHillMarshall1997).

The bottom of each column is a plane through the height of the column, as steep as it can be while each of its sides
stays between that height and the mean of the two corners on that side, so that it is flat at a local extremum. Where
the planes of two columns do not meet at their common face a vertical wall joins them, and the face opens down to the
higher of the two. A `Field` or `Array` sampled at cell centers gives the height of each column, a column at or above the
top of the grid is dry, and each corner is the mean of the wet columns around it. A function of `(x, y)` is evaluated at
the cell corners, a `Field{Face, Face, Nothing}` gives the corners directly, and the height of a column is the mean of
its four corners, so that each side of its plane sits at the mean of the two corners on that side.

In each level a cell is as tall as the mean opening its plane leaves at the lateral faces of the cell, so that a column
is exactly as deep as its height. The opening the height of a column leaves in its level is rounded to the nearest of
zero and

```
minimum_fractional_cell_height * Δz,
```

where `Δz` is the height of that level, and the slopes of a plane are reduced until no side of it leaves a thinner
opening, so that no face opens less than `minimum_fractional_cell_height * Δz`.
"""
CutCellBottom(bottom_height; minimum_fractional_cell_height=0.2) = CutCellBottom(bottom_height, nothing, nothing, nothing, minimum_fractional_cell_height, nothing)

function Base.summary(ib::CutCellBottom)
    bottom_interior = bottom_height_interior(ib.bottom_height)
    zmax = maximum(bottom_interior)
    zmin = minimum(bottom_interior)
    zmean = sum(bottom_interior) / length(bottom_interior)

    summary1 = "CutCellBottom("

    summary2 = string("mean(zb)=", prettysummary(zmean),
                      ", min(zb)=", prettysummary(zmin),
                      ", max(zb)=", prettysummary(zmax),
                      ", ϵ=", prettysummary(ib.minimum_fractional_cell_height))

    summary3 = ")"

    return summary1 * summary2 * summary3
end

Base.summary(ib::CutCellBottom{<:Function}) = @sprintf("CutCellBottom(%s, ϵ=%.1f)",
                                                          prettysummary(ib.bottom_height, false),
                                                          ib.minimum_fractional_cell_height)

function Base.show(io::IO, ib::CutCellBottom)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "└── minimum_fractional_cell_height: ", prettysummary(ib.minimum_fractional_cell_height))
end

Adapt.adapt_structure(to, ib::CutCellBottom) = CutCellBottom(adapt(to, ib.bottom_height),
                                                                  adapt(to, ib.west_bottom_height),
                                                                  adapt(to, ib.south_bottom_height),
                                                                  adapt(to, ib.corner_bottom_height),
                                                                  ib.minimum_fractional_cell_height,
                                                                  adapt(to, ib.bottom_cell_index))

Architectures.on_architecture(to, ib::CutCellBottom) = CutCellBottom(on_architecture(to, ib.bottom_height),
                                                                          on_architecture(to, ib.west_bottom_height),
                                                                          on_architecture(to, ib.south_bottom_height),
                                                                          on_architecture(to, ib.corner_bottom_height),
                                                                          on_architecture(to, ib.minimum_fractional_cell_height),
                                                                          on_architecture(to, ib.bottom_cell_index))

function Grids.constructor_arguments(grid::CCBIBG)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid.underlying_grid)
    cut_cell_bottom_args = Dict(:bottom_height => grid.immersed_boundary.bottom_height,
                                   :corner_bottom_height => grid.immersed_boundary.corner_bottom_height,
                                   :minimum_fractional_cell_height => grid.immersed_boundary.minimum_fractional_cell_height)
    return underlying_grid_args, underlying_grid_kwargs, cut_cell_bottom_args
end

function Base.:(==)(scb1::CutCellBottom, scb2::CutCellBottom)
    return bottom_heights_equal(scb1.bottom_height, scb2.bottom_height) &&
           bottom_heights_equal(scb1.west_bottom_height, scb2.west_bottom_height) &&
           bottom_heights_equal(scb1.south_bottom_height, scb2.south_bottom_height) &&
           bottom_heights_equal(scb1.corner_bottom_height, scb2.corner_bottom_height) &&
           scb1.minimum_fractional_cell_height == scb2.minimum_fractional_cell_height
end

@inline x_index(i, grid, offset) = i + offset
@inline x_index(i, grid::XFlatGrid, offset) = i
@inline y_index(j, grid, offset) = j + offset
@inline y_index(j, grid::YFlatGrid, offset) = j

#####
##### Immersed bottom construction (called at setup)
#####

# Staggered fields carry a boundary point beyond the last cell in Bounded directions.
function staggered_bottom_parameters(grid)
    Nx, Ny, _ = size(grid)
    TX, TY, _ = topology(grid)

    Ix = TX === Flat ? (1:1) : (1:Nx+1)
    Iy = TY === Flat ? (1:1) : (1:Ny+1)

    return KernelParameters(Ix, Iy)
end

# Heights sampled at cell centers carry the wet/dry mask of the columns; corner heights given directly carry none.
center_bottom_height(bottom_height, grid) = nothing
center_bottom_height(bottom_height::Field{Face, Face, Nothing}, grid) = nothing

function center_bottom_height(bottom_height::AbstractArray, grid)
    center_field = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(center_field, bottom_height)
    fill_halo_regions!(center_field)
    return center_field
end

set_corner_bottom_height!(corner_field, grid, ib::CutCellBottom, ::Nothing, parameters) = set_bottom_height!(corner_field, ib.bottom_height)
set_corner_bottom_height!(corner_field, grid, ib::CutCellBottom, center_field, parameters) = launch!(architecture(grid), grid, parameters, _interpolate_bottom_height_to_corners!, corner_field, center_field, grid)

# A boundary that carries its corners is rebuilt from their interior, so that rebuilding is idempotent on any halo.
set_corner_bottom_height!(corner_field, grid, ib::CutCellBottom{<:Any, <:Any, <:Any, <:AbstractArray}, center_field::Field, parameters) = set_bottom_height!(corner_field, ib.corner_bottom_height)

@inline dry_column(i, j, grid, ::Nothing) = false
@inline dry_column(i, j, grid, center_field) = @inbounds center_field[i, j, 1] ≥ rnode(i, j, grid.Nz+1, grid, c, c, f)

# A corner is the mean of its wet neighboring centers, or sits at the surface when none is wet.
@kernel function _interpolate_bottom_height_to_corners!(corner_field, center_field, grid)
    i, j = @index(Global, NTuple)
    iᵂ = x_index(i, grid, -1)
    jˢ = y_index(j, grid, -1)
    rᵗ = rnode(i, j, grid.Nz+1, grid, c, c, f)

    Σr = zero(rᵗ)
    n = 0
    for (i′, j′) in ((iᵂ, jˢ), (i, jˢ), (iᵂ, j), (i, j))
        wet = !dry_column(i′, j′, grid, center_field)
        Σr += ifelse(wet, @inbounds(center_field[i′, j′, 1]), zero(rᵗ))
        n += wet
    end

    @inbounds corner_field[i, j, 1] = ifelse(n > 0, Σr / max(n, 1), rᵗ)
end

function materialize_immersed_boundary(grid, ib::CutCellBottom)
    FT = eltype(grid)
    # A positive floor keeps the surface inside a level even for ϵ = 0.
    ϵ = max(convert(FT, ib.minimum_fractional_cell_height), sqrt(eps(FT)))
    arch = architecture(grid)
    parameters = staggered_bottom_parameters(grid)

    center_field = center_bottom_height(ib.bottom_height, grid)

    corner_field = Field{Face, Face, Nothing}(grid)
    @apply_regionally set_corner_bottom_height!(corner_field, grid, ib, center_field, parameters)
    @apply_regionally launch!(arch, grid, parameters, _clamp_bottom_height_to_domain!, corner_field, grid)
    fill_halo_regions!(corner_field)

    bottom_field = Field{Center, Center, Nothing}(grid)
    @apply_regionally launch!(arch, grid, :xy, _compute_column_heights!, bottom_field, corner_field, center_field, grid, ϵ)
    fill_halo_regions!(bottom_field)

    west_field = Field{Face, Center, Nothing}(grid)
    south_field = Field{Center, Face, Nothing}(grid)
    @apply_regionally launch!(arch, grid, parameters, _compute_bottom_planes!, west_field, south_field, bottom_field, corner_field, grid, ϵ)
    fill_halo_regions!(west_field)
    fill_halo_regions!(south_field)

    ib = CutCellBottom(bottom_field.data, west_field.data, south_field.data, corner_field.data, ϵ, nothing)
    bottom_cell_index = Field{Center, Center, Nothing}(grid, index_type(grid))
    @apply_regionally launch!(arch, grid, :xy, _compute_bottom_cell_index!, bottom_cell_index, grid, ib)
    fill_halo_regions!(bottom_cell_index)

    return CutCellBottom(bottom_field.data, west_field.data, south_field.data, corner_field.data, ϵ, bottom_cell_index.data)
end

@kernel function _clamp_bottom_height_to_domain!(bottom_field, grid)
    i, j = @index(Global, NTuple)
    rᵈ = rnode(i, j, 1, grid, c, c, f)
    rᵗ = rnode(i, j, grid.Nz+1, grid, c, c, f)
    @inbounds bottom_field[i, j, 1] = clamp(bottom_field[i, j, 1], rᵈ, rᵗ)
end

# The height of a column sampled at its center, within the domain, or the mean of its four corners
@inline function column_height(i, j, grid, corner_field, center_field)
    rᵈ = rnode(i, j, 1, grid, c, c, f)
    rᵗ = rnode(i, j, grid.Nz+1, grid, c, c, f)
    return @inbounds clamp(center_field[i, j, 1], rᵈ, rᵗ)
end

@inline function column_height(i, j, grid, corner_field, ::Nothing)
    iᴱ = x_index(i, grid, +1)
    jᴺ = y_index(j, grid, +1)
    return @inbounds (corner_field[i, j, 1] + corner_field[iᴱ, j, 1] + corner_field[i, jᴺ, 1] + corner_field[iᴱ, jᴺ, 1]) / 4
end

@kernel function _compute_column_heights!(bottom_field, corner_field, center_field, grid, ϵ)
    i, j = @index(Global, NTuple)
    rᵗ = rnode(i, j, grid.Nz+1, grid, c, c, f)
    rᵇ = column_height(i, j, grid, corner_field, center_field)
    @inbounds bottom_field[i, j, 1] = ifelse(dry_column(i, j, grid, center_field), rᵗ, rounded_bottom_height(i, j, grid, rᵇ, ϵ))
end

# The opening a height leaves in the level it falls in is rounded to the nearest of zero and [ϵ Δr, Δr]
@inline function rounded_bottom_height(i, j, grid, rᵇ, ϵ)
    for k in 1:grid.Nz
        r⁻ = rnode(i, j, k,   grid, c, c, f)
        r⁺ = rnode(i, j, k+1, grid, c, c, f)
        Δr = Δrᶜᶜᶜ(i, j, k, grid)
        sliver = r⁺ - rᵇ < ϵ * Δr / 2
        rᵇ = ifelse(r⁻ ≤ rᵇ < r⁺, ifelse(sliver, r⁺, clamp(rᵇ, r⁻, r⁺ - ϵ * Δr)), rᵇ)
    end
    return rᵇ
end

@inline minmod(a, b) = ifelse(a * b > 0, sign(a) * min(abs(a), abs(b)), zero(a))

# A side of the plane through rᵇ that would leave an opening thinner than ϵ Δr in its level moves toward rᵇ: the upper side
# down to an opening of ϵ Δr, the lower side up to the top of its level. The plane falls flat when that does not settle.
@inline function rounded_half_slope(i, j, grid, rᵇ, a, ϵ)
    for pass in 1:4, k in 1:grid.Nz
        r⁺ = rnode(i, j, k+1, grid, c, c, f)
        ϵΔr = ϵ * Δrᶜᶜᶜ(i, j, k, grid)
        a = ifelse(r⁺ - ϵΔr < rᵇ + a < r⁺, r⁺ - ϵΔr - rᵇ, a)
        a = ifelse(r⁺ - ϵΔr < rᵇ - a < r⁺, rᵇ - r⁺, a)
    end

    sliver = false
    for k in 1:grid.Nz
        r⁺ = rnode(i, j, k+1, grid, c, c, f)
        ϵΔr = ϵ * Δrᶜᶜᶜ(i, j, k, grid)
        sliver = sliver | (r⁺ - ϵΔr < rᵇ + a < r⁺) | (r⁺ - ϵΔr < rᵇ - a < r⁺)
    end

    return ifelse(sliver, zero(a), a)
end

# The plane of a column is as steep as it can be while each of its sides stays between the height of the column and
# the mean of the two corners on that side; it is stored by its height at the west and south sides of the cell.
# Against a wall of the domain only the side toward the interior bounds it, and beyond the last cell it is flat.
@kernel function _compute_bottom_planes!(west_field, south_field, bottom_field, corner_field, grid, ϵ)
    i, j = @index(Global, NTuple)
    Nx, Ny, _ = size(grid)
    TX, TY, _ = topology(grid)
    iᴱ = x_index(i, grid, +1)
    jᴺ = y_index(j, grid, +1)
    rᵈ = rnode(i, j, 1, grid, c, c, f)
    rᵗ = rnode(i, j, grid.Nz+1, grid, c, c, f)
    rᵇ = @inbounds bottom_field[i, j, 1]

    Δˣ⁻ = @inbounds 2rᵇ - corner_field[i, j, 1] - corner_field[i, jᴺ, 1]
    Δˣ⁺ = @inbounds corner_field[iᴱ, j, 1] + corner_field[iᴱ, jᴺ, 1] - 2rᵇ
    Δʸ⁻ = @inbounds 2rᵇ - corner_field[i, j, 1] - corner_field[iᴱ, j, 1]
    Δʸ⁺ = @inbounds corner_field[i, jᴺ, 1] + corner_field[iᴱ, jᴺ, 1] - 2rᵇ

    sˣ = ifelse((TX === Bounded) & (i == 1), Δˣ⁺, ifelse((TX === Bounded) & (i == Nx), Δˣ⁻, minmod(Δˣ⁻, Δˣ⁺)))
    sʸ = ifelse((TY === Bounded) & (j == 1), Δʸ⁺, ifelse((TY === Bounded) & (j == Ny), Δʸ⁻, minmod(Δʸ⁻, Δʸ⁺)))

    dry = (rᵇ ≥ rᵗ) | (i > Nx) | (j > Ny)
    aˣ = ifelse(dry, zero(rᵇ), rounded_half_slope(i, j, grid, rᵇ, min(abs(sˣ) / 2, rᵗ - rᵇ, rᵇ - rᵈ), ϵ))
    aʸ = ifelse(dry, zero(rᵇ), rounded_half_slope(i, j, grid, rᵇ, min(abs(sʸ) / 2, rᵗ - rᵇ, rᵇ - rᵈ), ϵ))

    @inbounds west_field[i, j, 1] = rᵇ - sign(sˣ) * aˣ
    @inbounds south_field[i, j, 1] = rᵇ - sign(sʸ) * aʸ
end

@kernel function _compute_bottom_cell_index!(bottom_cell_index, grid, ib)
    i, j = @index(Global, NTuple)
    @inbounds bottom_cell_index[i, j, 1] = bottom_cell(i, j, grid, ib)
end

# Before the bottom index is stored, a cell is immersed when its plane leaves no opening in its level; land stays dry
@inline function _immersed_cell(i, j, k, grid, ib::CutCellBottom{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing})
    r⁺ = rnode(i, j, k + 1, grid, c, c, f)
    Δr = Δrᶜᶜᶜ(i, j, k, grid)
    dry = @inbounds ib.bottom_height[i, j, 1] ≥ rnode(i, j, grid.Nz + 1, grid, c, c, f)
    return (mean_face_opening(i, j, k, grid, ib, r⁺, Δr) == 0) | dry
end

# The first active level of a column, or Nz + 1 for a dry column, searched before the index is stored
@inline function bottom_cell(i, j, grid, ib::CutCellBottom{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing})
    kᵇ = grid.Nz + 1
    for k in grid.Nz:-1:1
        kᵇ = ifelse(_immersed_cell(i, j, k, grid, ib), kᵇ, k)
    end
    return kᵇ
end

#####
##### Cut-cell methods (called during stepping)
#####

# Open height of a face in level k, from the bottom height sampled at that face
@inline face_open_height(rᵇ, r⁺, Δr) = clamp(r⁺ - rᵇ, zero(Δr), Δr)

# The mean opening the plane of a cell leaves at its lateral faces in level k: the plane is at rᵂ on the west face and at
# 2rᵇ - rᵂ on the east face. In a Flat direction the two faces are one phantom face.
@inline function mean_face_opening(i, j, k, grid, ib, r⁺, Δr)
    TX, TY, _ = topology(grid)
    wˣ = ifelse((TX === Flat) & (TY !== Flat), 0, 1)
    wʸ = ifelse((TY === Flat) & (TX !== Flat), 0, 1)
    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    rᵂ = @inbounds ib.west_bottom_height[i, j, 1]
    rˢ = @inbounds ib.south_bottom_height[i, j, 1]
    oˣ = face_open_height(rᵂ, r⁺, Δr) + face_open_height(2rᵇ - rᵂ, r⁺, Δr)
    oʸ = face_open_height(rˢ, r⁺, Δr) + face_open_height(2rᵇ - rˢ, r⁺, Δr)
    return (wˣ * oˣ + wʸ * oʸ) / (2 * (wˣ + wʸ))
end

# A vertical wall joins two planes that do not meet at a face, so the face opens down to the higher of the two
@inline bottom_heightᶠᶜᵃ(i, j, grid, ib) = @inbounds max(2 * ib.bottom_height[x_index(i, grid, -1), j, 1] - ib.west_bottom_height[x_index(i, grid, -1), j, 1], ib.west_bottom_height[i, j, 1])
@inline bottom_heightᶜᶠᵃ(i, j, grid, ib) = @inbounds max(2 * ib.bottom_height[i, y_index(j, grid, -1), 1] - ib.south_bottom_height[i, y_index(j, grid, -1), 1], ib.south_bottom_height[i, j, 1])

@inline _immersed_cell(i, j, k, underlying_grid, ib::CutCellBottom) = k < bottom_cell(i, j, underlying_grid, ib)
@inline bottom_cell(i, j, grid, ib::CutCellBottom) = @inbounds ib.bottom_cell_index[i, j, 1]

# An active cell is as tall as the mean opening of its plane; immersed cells keep the full level height
@inline function Δrᶜᶜᶜ(i, j, k, ibg::CCBIBG)
    grid = ibg.underlying_grid
    Δr = Δrᶜᶜᶜ(i, j, k, grid)
    opening = mean_face_opening(i, j, k, grid, ibg.immersed_boundary, rnode(i, j, k + 1, grid, c, c, f), Δr)
    return ifelse(_immersed_cell(i, j, k, grid, ibg.immersed_boundary), Δr, opening)
end

@inline function Δrᶠᶜᶜ(i, j, k, ibg::CCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = bottom_heightᶠᶜᵃ(i, j, underlying_grid, ibg.immersed_boundary)
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)
    Δr = Δrᶠᶜᶜ(i, j, k, underlying_grid)
    cut = clamp(r⁺ - rᵇ, zero(Δr), Δr)
    return ifelse(r⁺ ≤ rᵇ, Δr, cut)
end

@inline function Δrᶜᶠᶜ(i, j, k, ibg::CCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = bottom_heightᶜᶠᵃ(i, j, underlying_grid, ibg.immersed_boundary)
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)
    Δr = Δrᶜᶠᶜ(i, j, k, underlying_grid)
    cut = clamp(r⁺ - rᵇ, zero(Δr), Δr)
    return ifelse(r⁺ ≤ rᵇ, Δr, cut)
end

@inline Δrᶠᶠᶜ(i, j, k, ibg::CCBIBG) = min(Δrᶠᶜᶜ(i, j-1, k, ibg), Δrᶠᶜᶜ(i, j, k, ibg))

# The center of a cut cell sits half its height below the interface above it.
@inline function cut_center_spacing(i, j, k, underlying_grid, rᵇ, full_Δr)
    rᶜ = rnode(i, j, k, underlying_grid, c, c, c)
    rᶠ = rnode(i, j, k, underlying_grid, c, c, f)
    above_bottom = rnode(i, j, k-1, underlying_grid, c, c, f) ≤ rᵇ < rᶠ
    return ifelse(above_bottom, rᶜ - rᶠ + (rᶠ - rᵇ) / 2, full_Δr)
end

@inline Δrᶜᶜᶠ(i, j, k, ibg::CCBIBG) = (Δrᶜᶜᶜ(i, j, k-1, ibg) + Δrᶜᶜᶜ(i, j, k, ibg)) / 2

@inline function Δrᶠᶜᶠ(i, j, k, ibg::CCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = bottom_heightᶠᶜᵃ(i, j, underlying_grid, ibg.immersed_boundary)
    return cut_center_spacing(i, j, k, underlying_grid, rᵇ, Δrᶠᶜᶠ(i, j, k, underlying_grid))
end

@inline function Δrᶜᶠᶠ(i, j, k, ibg::CCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = bottom_heightᶜᶠᵃ(i, j, underlying_grid, ibg.immersed_boundary)
    return cut_center_spacing(i, j, k, underlying_grid, rᵇ, Δrᶜᶠᶠ(i, j, k, underlying_grid))
end

@inline Δrᶠᶠᶠ(i, j, k, ibg::CCBIBG) = min(Δrᶠᶜᶠ(i, j-1, k, ibg), Δrᶠᶜᶠ(i, j, k, ibg))

# Flat topologies collapse the staggered metrics onto the centered ones.
@inline Δrᶠᶜᶜ(i, j, k, ibg::XFlatCCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶜᶠ(i, j, k, ibg::XFlatCCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶜᶠᶜ(i, j, k, ibg::YFlatCCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶜᶠᶠ(i, j, k, ibg::YFlatCCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XFlatCCBIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::YFlatCCBIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶠ(i, j, k, ibg::XFlatCCBIBG) = Δrᶜᶠᶠ(i, j, k, ibg)
@inline Δrᶠᶠᶠ(i, j, k, ibg::YFlatCCBIBG) = Δrᶠᶜᶠ(i, j, k, ibg)

@inline Δrᶠᶠᶜ(i, j, k, ibg::XYFlatCCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶠ(i, j, k, ibg::XYFlatCCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)

@inline Δzᶜᶜᶜ(i, j, k, ibg::VSCCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δzᶠᶜᶜ(i, j, k, ibg::VSCCBIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶜ(i, j, k, ibg::VSCCBIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶜᶠ(i, j, k, ibg::VSCCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶜ(i, j, k, ibg::VSCCBIBG) = Δrᶠᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶠ(i, j, k, ibg::VSCCBIBG) = Δrᶜᶠᶠ(i, j, k, ibg)
@inline Δzᶠᶜᶠ(i, j, k, ibg::VSCCBIBG) = Δrᶠᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶠ(i, j, k, ibg::VSCCBIBG) = Δrᶠᶠᶠ(i, j, k, ibg)

@inline top_cell(i, j, grid, ::CutCellBottom) = size(grid, 3)
@inline bottom_height(i, j, ibg::CCBIBG) = @inbounds ibg.immersed_boundary.bottom_height[i, j, 1]

# A face column reaches down to the bottom height at that face, and is dry next to a dry column
@inline function face_column_depth(i, j, i⁻, j⁻, ibg, ib::CutCellBottom)
    underlying_grid = ibg.underlying_grid
    rᵇ = ifelse(i⁻ == i, bottom_heightᶜᶠᵃ(i, j, underlying_grid, ib), bottom_heightᶠᶜᵃ(i, j, underlying_grid, ib))
    rᵗ = rnode(i, j, ibg.Nz+1, ibg, c, c, f)
    dry = (bottom_height(i, j, ibg) ≥ rᵗ) | (bottom_height(i⁻, j⁻, ibg) ≥ rᵗ)
    return ifelse(dry, zero(rᵗ), rᵗ - rᵇ)
end

@inline Operators.Δr⁻¹ᶜᶜᶜ(i, j, k, ibg::CCBIBG) = 1 / Δrᶜᶜᶜ(i, j, k, ibg)
@inline Operators.Δr⁻¹ᶜᶜᶠ(i, j, k, ibg::CCBIBG) = 1 / Δrᶜᶜᶠ(i, j, k, ibg)
@inline Operators.Δr⁻¹ᶠᶜᶜ(i, j, k, ibg::CCBIBG) = 1 / Δrᶠᶜᶜ(i, j, k, ibg)
@inline Operators.Δr⁻¹ᶜᶠᶜ(i, j, k, ibg::CCBIBG) = 1 / Δrᶜᶠᶜ(i, j, k, ibg)
@inline Operators.Δr⁻¹ᶠᶠᶜ(i, j, k, ibg::CCBIBG) = 1 / Δrᶠᶠᶜ(i, j, k, ibg)
@inline Operators.Δr⁻¹ᶠᶜᶠ(i, j, k, ibg::CCBIBG) = 1 / Δrᶠᶜᶠ(i, j, k, ibg)
@inline Operators.Δr⁻¹ᶜᶠᶠ(i, j, k, ibg::CCBIBG) = 1 / Δrᶜᶠᶠ(i, j, k, ibg)
@inline Operators.Δr⁻¹ᶠᶠᶠ(i, j, k, ibg::CCBIBG) = 1 / Δrᶠᶠᶠ(i, j, k, ibg)

@inline Operators.Δz⁻¹ᶜᶜᶜ(i, j, k, ibg::VSCCBIBG) = Δr⁻¹ᶜᶜᶜ(i, j, k, ibg)
@inline Operators.Δz⁻¹ᶜᶜᶠ(i, j, k, ibg::VSCCBIBG) = Δr⁻¹ᶜᶜᶠ(i, j, k, ibg)
@inline Operators.Δz⁻¹ᶠᶜᶜ(i, j, k, ibg::VSCCBIBG) = Δr⁻¹ᶠᶜᶜ(i, j, k, ibg)
@inline Operators.Δz⁻¹ᶜᶠᶜ(i, j, k, ibg::VSCCBIBG) = Δr⁻¹ᶜᶠᶜ(i, j, k, ibg)
@inline Operators.Δz⁻¹ᶠᶠᶜ(i, j, k, ibg::VSCCBIBG) = Δr⁻¹ᶠᶠᶜ(i, j, k, ibg)
@inline Operators.Δz⁻¹ᶠᶜᶠ(i, j, k, ibg::VSCCBIBG) = Δr⁻¹ᶠᶜᶠ(i, j, k, ibg)
@inline Operators.Δz⁻¹ᶜᶠᶠ(i, j, k, ibg::VSCCBIBG) = Δr⁻¹ᶜᶠᶠ(i, j, k, ibg)
@inline Operators.Δz⁻¹ᶠᶠᶠ(i, j, k, ibg::VSCCBIBG) = Δr⁻¹ᶠᶠᶠ(i, j, k, ibg)

# A face the bottom closes is peripheral even between two active cells, as on a crest, and carries no velocity
@inline peripheral_node(i, j, k, ibg::CCBIBG, ::Face, ::Center, ::Center) = inactive_cell(i, j, k, ibg) | inactive_cell(i-1, j, k, ibg) |
                                                                           (rnode(i, j, k+1, ibg.underlying_grid, c, c, f) ≤ bottom_heightᶠᶜᵃ(i, j, ibg.underlying_grid, ibg.immersed_boundary))

@inline peripheral_node(i, j, k, ibg::CCBIBG, ::Center, ::Face, ::Center) = inactive_cell(i, j, k, ibg) | inactive_cell(i, j-1, k, ibg) |
                                                                           (rnode(i, j, k+1, ibg.underlying_grid, c, c, f) ≤ bottom_heightᶜᶠᵃ(i, j, ibg.underlying_grid, ibg.immersed_boundary))

# A Flat direction has a single phantom face, which never closes
@inline peripheral_node(i, j, k, ibg::XFlatCCBIBG, ::Face, ::Center, ::Center) = inactive_cell(i, j, k, ibg) | inactive_cell(i-1, j, k, ibg)
@inline peripheral_node(i, j, k, ibg::YFlatCCBIBG, ::Center, ::Face, ::Center) = inactive_cell(i, j, k, ibg) | inactive_cell(i, j-1, k, ibg)

# Nodes on the edges of a closed face are peripheral, as they are on the edges of a face next to an immersed cell
@inline peripheral_node(i, j, k, ibg::CCBIBG, ::Face, ::Center, ::Face) = peripheral_node(i, j, k, ibg, f, c, c) | peripheral_node(i, j, k-1, ibg, f, c, c)

@inline peripheral_node(i, j, k, ibg::CCBIBG, ::Face, ::Face, ::Center) = peripheral_node(i, j, k, ibg, f, c, c) | peripheral_node(i, j-1, k, ibg, f, c, c) |
                                                                          peripheral_node(i, j, k, ibg, c, f, c) | peripheral_node(i-1, j, k, ibg, c, f, c)

@inline peripheral_node(i, j, k, ibg::CCBIBG, ::Face, ::Face, ::Face) = peripheral_node(i, j, k, ibg, f, f, c) | peripheral_node(i, j, k-1, ibg, f, f, c)

# No difference reaches across a closed face, as none reaches into an immersed cell
@inline conditional_δx_f(::Center, ::Center, i, j, k, ibg::CCBIBG, δx, args...) = ifelse(immersed_peripheral_node(i, j, k, ibg, f, c, c), zero(ibg), δx(i, j, k, ibg, args...))
@inline conditional_δy_f(::Center, ::Center, i, j, k, ibg::CCBIBG, δy, args...) = ifelse(immersed_peripheral_node(i, j, k, ibg, c, f, c), zero(ibg), δy(i, j, k, ibg, args...))
