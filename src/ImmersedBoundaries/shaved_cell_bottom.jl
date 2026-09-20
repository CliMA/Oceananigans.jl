using Oceananigans.Fields: Field, fill_halo_regions!, set!
using Oceananigans.Grids: Grids, AbstractStaticGrid, constructor_arguments, XFlatGrid, YFlatGrid
using Oceananigans.Utils: prettysummary, KernelParameters

import Oceananigans.Grids: peripheral_node
import Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶜᶜᶠ, Δrᶜᶠᶜ, Δrᶜᶠᶠ, Δrᶠᶜᶜ, Δrᶠᶜᶠ, Δrᶠᶠᶜ, Δrᶠᶠᶠ,
                               Δzᶜᶜᶜ, Δzᶜᶜᶠ, Δzᶜᶠᶜ, Δzᶜᶠᶠ, Δzᶠᶜᶜ, Δzᶠᶜᶠ, Δzᶠᶠᶜ, Δzᶠᶠᶠ

#####
##### ShavedCellBottom
#####

struct ShavedCellBottom{H, F, E} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    west_bottom_height :: F
    south_bottom_height :: F
    corner_bottom_height :: F
    minimum_fractional_cell_height :: E
end

const SCBIBG{FT, TX, TY, TZ} = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:Any, <:ShavedCellBottom} where {FT, TX, TY, TZ}

"""
    ShavedCellBottom(bottom_height; minimum_fractional_cell_height=0.2)

Return a `ShavedCellBottom` representing an immersed boundary with "shaved" bottom cells, inspired by
the shaved cells of [Adcroft et al. (1997)](@cite AdcroftHillMarshall1997).

The bottom is a piecewise-bilinear surface reconstructed from `bottom_height`, which may be a
`Field`, `Array`, or function of `(x, y)`. Every lateral face of a bottom cell is cut by that
surface at the horizontal position of the face itself, so the faces of one cell carry different
heights and the bottom slopes through the cell. The height of the cell is the mean of the heights of
its lateral faces, so that the faces bound the cell volume. When `bottom_height` is sampled at cell
centers, a column at or above the top of the grid stays dry and does not enter the surface of its neighbors.

The surface shaves a single level in each column and in each lateral face: the lowest level that is not
immersed. A slope steeper than one level per cell is therefore represented by a staircase of shaved cells.
The height of a shaved cell, and of each of its open lateral faces, is greater than

```
minimum_fractional_cell_height * Δz,
```

where `Δz` is the original height of the bottom cell of the underlying grid.
"""
ShavedCellBottom(bottom_height; minimum_fractional_cell_height=0.2) = ShavedCellBottom(bottom_height, nothing, nothing, nothing, minimum_fractional_cell_height)

function Base.summary(ib::ShavedCellBottom)
    bottom_interior = bottom_height_interior(ib.bottom_height)
    zmax = maximum(bottom_interior)
    zmin = minimum(bottom_interior)
    zmean = sum(bottom_interior) / length(bottom_interior)

    summary1 = "ShavedCellBottom("

    summary2 = string("mean(zb)=", prettysummary(zmean),
                      ", min(zb)=", prettysummary(zmin),
                      ", max(zb)=", prettysummary(zmax),
                      ", ϵ=", prettysummary(ib.minimum_fractional_cell_height))

    summary3 = ")"

    return summary1 * summary2 * summary3
end

Base.summary(ib::ShavedCellBottom{<:Function}) = @sprintf("ShavedCellBottom(%s, ϵ=%.1f)",
                                                          prettysummary(ib.bottom_height, false),
                                                          ib.minimum_fractional_cell_height)

function Base.show(io::IO, ib::ShavedCellBottom)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "└── minimum_fractional_cell_height: ", prettysummary(ib.minimum_fractional_cell_height))
end

@inline x_index(i, grid, offset) = i + offset
@inline x_index(i, grid::XFlatGrid, offset) = i
@inline y_index(j, grid, offset) = j + offset
@inline y_index(j, grid::YFlatGrid, offset) = j

#####
##### Materialization
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

set_corner_bottom_height!(corner_field, grid, ib::ShavedCellBottom, ::Nothing, parameters) = set_bottom_height!(corner_field, ib.bottom_height)
set_corner_bottom_height!(corner_field, grid, ib::ShavedCellBottom, center_field, parameters) = launch!(architecture(grid), grid, parameters, _interpolate_bottom_height_to_corners!, corner_field, center_field, grid)

# A materialized boundary is rebuilt from the interior of its own corners, so that rebuilding is idempotent on any halo.
set_corner_bottom_height!(corner_field, grid, ib::ShavedCellBottom{<:Any, <:AbstractArray}, center_field::Field, parameters) = set_bottom_height!(corner_field, ib.corner_bottom_height)

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

function materialize_immersed_boundary(grid, ib::ShavedCellBottom)
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
    @apply_regionally launch!(arch, grid, :xy, _average_corners_to_centers!, bottom_field, corner_field, center_field, grid, ϵ)
    fill_halo_regions!(bottom_field)

    west_field = Field{Face, Center, Nothing}(grid)
    south_field = Field{Center, Face, Nothing}(grid)
    previous_bottom_field = Field{Center, Center, Nothing}(grid)

    compute_ib = ShavedCellBottom(bottom_field, nothing, nothing, nothing, ϵ)

    TX, TY, _ = topology(grid)
    wˣ = TX === Flat && TY !== Flat ? 0 : 1
    wʸ = TY === Flat && TX !== Flat ? 0 : 1

    # Immersing a cell can close the faces of its neighbors, so faces and cells are recomputed until no level changes.
    converged = false
    while !converged
        set!(previous_bottom_field, bottom_field)

        @apply_regionally launch!(arch, grid, parameters, _compute_shaved_face_bottom_heights!, west_field, south_field, corner_field, grid, compute_ib)

        fill_halo_regions!(west_field)
        fill_halo_regions!(south_field)

        @apply_regionally launch!(arch, grid, :xy, _average_faces_to_centers!, bottom_field, west_field, south_field, grid, compute_ib, wˣ, wʸ)

        fill_halo_regions!(bottom_field)
        converged = maximum(abs, bottom_field - previous_bottom_field) == 0
    end

    return ShavedCellBottom(bottom_field.data, west_field.data, south_field.data, corner_field.data, ϵ)
end

@kernel function _clamp_bottom_height_to_domain!(bottom_field, grid)
    i, j = @index(Global, NTuple)
    rᵈ = rnode(i, j, 1, grid, c, c, f)
    rᵗ = rnode(i, j, grid.Nz+1, grid, c, c, f)
    @inbounds bottom_field[i, j, 1] = clamp(bottom_field[i, j, 1], rᵈ, rᵗ)
end

# The mean of a bilinear surface over a cell is the mean of its four corners; the ϵ-limiter follows, and dry columns stay dry.
@kernel function _average_corners_to_centers!(bottom_field, corner_field, center_field, grid, ϵ)
    i, j = @index(Global, NTuple)

    iᴱ = x_index(i, grid, +1)
    jᴺ = y_index(j, grid, +1)

    rˢ = @inbounds (corner_field[i, j,  1] + corner_field[iᴱ, j,  1]) / 2
    rᴺ = @inbounds (corner_field[i, jᴺ, 1] + corner_field[iᴱ, jᴺ, 1]) / 2
    rᵇ = (rˢ + rᴺ) / 2

    for k in 1:grid.Nz
        r⁻ = rnode(i, j, k,   grid, c, c, f)
        r⁺ = rnode(i, j, k+1, grid, c, c, f)
        Δr = Δrᶜᶜᶜ(i, j, k, grid)
        bottom_cell = r⁻ ≤ rᵇ < r⁺
        rᵇ = ifelse(bottom_cell, min(r⁺ - ϵ * Δr, rᵇ), rᵇ)
    end

    rᵗ = rnode(i, j, grid.Nz+1, grid, c, c, f)
    @inbounds bottom_field[i, j, 1] = ifelse(dry_column(i, j, grid, center_field), rᵗ, rᵇ)
end

# Index of the bottom-most cell of column (i, j) that is not immersed, or Nz + 1 for a dry column.
@inline function bottom_active_index(i, j, grid, ib)
    kᵇ = grid.Nz + 1
    for k in grid.Nz:-1:1
        kᵇ = ifelse(_immersed_cell(i, j, k, grid, ib), kᵇ, k)
    end
    return kᵇ
end

# Snap a face bottom height into the lowest level the face leaves open, keeping it within [ϵ Δr, Δr]
# of that level.
@inline function shaved_face_bottom_height(i, j, kᵇ, grid, rᵇ, ϵ)
    k  = min(kᵇ, grid.Nz)
    r⁻ = rnode(i, j, k,   grid, c, c, f)
    r⁺ = rnode(i, j, k+1, grid, c, c, f)
    Δr = Δrᶜᶜᶜ(i, j, k, grid)
    return clamp(rᵇ, r⁻, r⁺ - ϵ * Δr)
end

# The level a height falls in, or the top level when it lies above the grid
@inline function bottom_level(i, j, grid, rᵇ)
    k = 1
    for kk in 1:grid.Nz
        k = ifelse(shaved_level(i, j, kk, grid, rᵇ), kk, k)
    end
    return k
end

# True when the bottom surface at height rᵇ cuts through level k.
@inline shaved_level(i, j, k, grid, rᵇ) = rnode(i, j, k, grid, c, c, f) ≤ rᵇ < rnode(i, j, k+1, grid, c, c, f)

@kernel function _compute_shaved_face_bottom_heights!(west_field, south_field, corner_field, grid, ib)
    i, j = @index(Global, NTuple)

    iᵂ = x_index(i, grid, -1)
    iᴱ = x_index(i, grid, +1)
    jˢ = y_index(j, grid, -1)
    jᴺ = y_index(j, grid, +1)

    ϵ = ib.minimum_fractional_cell_height

    kᶜ = bottom_active_index(i,  j,  grid, ib)
    kᵂ = bottom_active_index(iᵂ, j,  grid, ib)
    kˢ = bottom_active_index(i,  jˢ, grid, ib)

    rᶠᶜ = @inbounds (corner_field[i, j, 1] + corner_field[i, jᴺ, 1]) / 2
    rᶜᶠ = @inbounds (corner_field[i, j, 1] + corner_field[iᴱ, j, 1]) / 2

    # Cap each face height inside the level it cuts, so that the opening it leaves is at least ϵ Δr tall and columns
    # never become arbitrarily thin (which would wreck the conditioning of the free-surface solve).
    @inbounds west_field[i, j, 1]  = shaved_face_bottom_height(i, j, bottom_level(i, j, grid, rᶠᶜ), grid, rᶠᶜ, ϵ)
    @inbounds south_field[i, j, 1] = shaved_face_bottom_height(i, j, bottom_level(i, j, grid, rᶜᶠ), grid, rᶜᶠ, ϵ)
end

# A face shaved in a higher level is closed in this one, so clipping the faces into the level of the cell makes them bound its volume.
@kernel function _average_faces_to_centers!(bottom_field, west_field, south_field, grid, ib, wˣ, wʸ)
    i, j = @index(Global, NTuple)

    iᴱ = x_index(i, grid, +1)
    jᴺ = y_index(j, grid, +1)

    kᶜ = bottom_active_index(i, j, grid, ib)
    k  = min(kᶜ, grid.Nz)
    r⁻ = rnode(i, j, k,   grid, c, c, f)
    r⁺ = rnode(i, j, k+1, grid, c, c, f)

    rˣ = @inbounds (west_field[i, j, 1]  + west_field[iᴱ, j, 1])  / 2
    rʸ = @inbounds (south_field[i, j, 1] + south_field[i, jᴺ, 1]) / 2
    rᵇ = (wˣ * rˣ + wʸ * rʸ) / (wˣ + wʸ)

    @inbounds bottom_field[i, j, 1] = ifelse(kᶜ > grid.Nz, bottom_field[i, j, 1], rᵇ)
end

Adapt.adapt_structure(to, ib::ShavedCellBottom) = ShavedCellBottom(adapt(to, ib.bottom_height),
                                                                  adapt(to, ib.west_bottom_height),
                                                                  adapt(to, ib.south_bottom_height),
                                                                  adapt(to, ib.corner_bottom_height),
                                                                  ib.minimum_fractional_cell_height)

Architectures.on_architecture(to, ib::ShavedCellBottom) = ShavedCellBottom(on_architecture(to, ib.bottom_height),
                                                                          on_architecture(to, ib.west_bottom_height),
                                                                          on_architecture(to, ib.south_bottom_height),
                                                                          on_architecture(to, ib.corner_bottom_height),
                                                                          on_architecture(to, ib.minimum_fractional_cell_height))

#####
##### Immersed cells and grid spacings
#####

# Open height of a face in level k, from the bottom height sampled at that face
@inline face_open_height(rᵇ, r⁺, Δr) = clamp(r⁺ - rᵇ, zero(Δr), Δr)

# A cell is immersed when neither its centre nor any of its faces leaves an opening in this level: a cell whose centre is
# below the shaved surface but whose faces are open still holds the wedge they leave, and masking it discards that flux.
# Land, whose bottom reaches the surface, stays dry however deep the sea on the other side of its faces is.
@inline function shaved_cell_metrics(i, j, k, grid, ib)
    r⁺ = rnode(i, j, k + 1, grid, c, c, f)
    rᵗ = rnode(i, j, grid.Nz + 1, grid, c, c, f)
    Δr = Δrᶜᶜᶜ(i, j, k, grid)
    ϵ = ib.minimum_fractional_cell_height
    rᵇ = @inbounds ib.bottom_height[i, j, 1]

    opening = mean_face_opening(i, j, k, grid, ib, r⁺, Δr)
    dry = rᵇ ≥ rᵗ
    immersed = ((r⁺ - Δr * ϵ) < rᵇ) & ((opening == 0) | dry)

    return immersed, opening
end

@inline opening_x(i, j, k, grid, ib, r⁺, Δr) = @inbounds (face_open_height(ib.west_bottom_height[i, j, 1], r⁺, Δr) +
                                                          face_open_height(ib.west_bottom_height[x_index(i, grid, +1), j, 1], r⁺, Δr)) / 2

@inline opening_y(i, j, k, grid, ib, r⁺, Δr) = @inbounds (face_open_height(ib.south_bottom_height[i, j, 1], r⁺, Δr) +
                                                          face_open_height(ib.south_bottom_height[i, y_index(j, grid, +1), 1], r⁺, Δr)) / 2

@inline mean_face_opening(i, j, k, grid, ib, r⁺, Δr) = (opening_x(i, j, k, grid, ib, r⁺, Δr) + opening_y(i, j, k, grid, ib, r⁺, Δr)) / 2
@inline mean_face_opening(i, j, k, grid::XFlatGrid, ib, r⁺, Δr) = opening_y(i, j, k, grid, ib, r⁺, Δr)
@inline mean_face_opening(i, j, k, grid::YFlatGrid, ib, r⁺, Δr) = opening_x(i, j, k, grid, ib, r⁺, Δr)

@inline mean_face_opening(i, j, k, grid, ib::ShavedCellBottom{<:Any, Nothing}, r⁺, Δr) = zero(grid)
@inline mean_face_opening(i, j, k, grid::XFlatGrid, ib::ShavedCellBottom{<:Any, Nothing}, r⁺, Δr) = zero(grid)
@inline mean_face_opening(i, j, k, grid::YFlatGrid, ib::ShavedCellBottom{<:Any, Nothing}, r⁺, Δr) = zero(grid)

@inline _immersed_cell(i, j, k, underlying_grid, ib::ShavedCellBottom) = first(shaved_cell_metrics(i, j, k, underlying_grid, ib))

# A cell or a face keeps the full height of its level except in the level the surface cuts, where it
# keeps the part above the surface. Levels below keep the full height: they are masked, and a
# positive height keeps them out of denominators.

# Cells kept active by their faces alone take the mean opening of those faces as their height, so that volume and face areas stay consistent.
# A cell is as tall as the mean of the openings its faces leave: on a flat bottom this is the partial cell height, and
# summing it over a column returns the depth the faces describe.
@inline function Δrᶜᶜᶜ(i, j, k, ibg::SCBIBG)
    grid = ibg.underlying_grid
    immersed, opening = shaved_cell_metrics(i, j, k, grid, ibg.immersed_boundary)
    return ifelse(immersed, Δrᶜᶜᶜ(i, j, k, grid), opening)
end

@inline function Δrᶠᶜᶜ(i, j, k, ibg::SCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.west_bottom_height[i, j, 1]
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)
    Δr = Δrᶠᶜᶜ(i, j, k, underlying_grid)
    cut = clamp(r⁺ - rᵇ, zero(Δr), Δr)
    return ifelse(r⁺ ≤ rᵇ, Δr, cut)
end

@inline function Δrᶜᶠᶜ(i, j, k, ibg::SCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.south_bottom_height[i, j, 1]
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)
    Δr = Δrᶜᶠᶜ(i, j, k, underlying_grid)
    cut = clamp(r⁺ - rᵇ, zero(Δr), Δr)
    return ifelse(r⁺ ≤ rᵇ, Δr, cut)
end

@inline Δrᶠᶠᶜ(i, j, k, ibg::SCBIBG) = min(Δrᶠᶜᶜ(i, j-1, k, ibg), Δrᶠᶜᶜ(i, j, k, ibg))

# The center of a shaved cell sits half a shaved height below the interface above it.
@inline function shaved_center_spacing(i, j, k, underlying_grid, rᵇ, full_Δr)
    rᶜ = rnode(i, j, k, underlying_grid, c, c, c)
    rᶠ = rnode(i, j, k, underlying_grid, c, c, f)
    above_bottom = shaved_level(i, j, k-1, underlying_grid, rᵇ)
    return ifelse(above_bottom, rᶜ - rᶠ + (rᶠ - rᵇ) / 2, full_Δr)
end

@inline function Δrᶜᶜᶠ(i, j, k, ibg::SCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.bottom_height[i, j, 1]
    return shaved_center_spacing(i, j, k, underlying_grid, rᵇ, Δrᶜᶜᶠ(i, j, k, underlying_grid))
end

@inline function Δrᶠᶜᶠ(i, j, k, ibg::SCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.west_bottom_height[i, j, 1]
    return shaved_center_spacing(i, j, k, underlying_grid, rᵇ, Δrᶠᶜᶠ(i, j, k, underlying_grid))
end

@inline function Δrᶜᶠᶠ(i, j, k, ibg::SCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.south_bottom_height[i, j, 1]
    return shaved_center_spacing(i, j, k, underlying_grid, rᵇ, Δrᶜᶠᶠ(i, j, k, underlying_grid))
end

@inline Δrᶠᶠᶠ(i, j, k, ibg::SCBIBG) = min(Δrᶠᶜᶠ(i, j-1, k, ibg), Δrᶠᶜᶠ(i, j, k, ibg))

# Flat topologies collapse the staggered metrics onto the centered ones.
XFlatSCBIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:ShavedCellBottom}
YFlatSCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:ShavedCellBottom}

@inline Δrᶠᶜᶜ(i, j, k, ibg::XFlatSCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶜᶠ(i, j, k, ibg::XFlatSCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶜᶠᶜ(i, j, k, ibg::YFlatSCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶜᶠᶠ(i, j, k, ibg::YFlatSCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XFlatSCBIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::YFlatSCBIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶠ(i, j, k, ibg::XFlatSCBIBG) = Δrᶜᶠᶠ(i, j, k, ibg)
@inline Δrᶠᶠᶠ(i, j, k, ibg::YFlatSCBIBG) = Δrᶠᶜᶠ(i, j, k, ibg)

XYFlatSCBIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Flat, <:Any, <:Any, <:ShavedCellBottom}

@inline Δrᶠᶠᶜ(i, j, k, ibg::XYFlatSCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶠ(i, j, k, ibg::XYFlatSCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)

# Vertically-static, shaved cell bottom, immersed boundary grid
VSSCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractStaticGrid, <:ShavedCellBottom}
@inline Δzᶜᶜᶜ(i, j, k, ibg::VSSCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δzᶠᶜᶜ(i, j, k, ibg::VSSCBIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶜ(i, j, k, ibg::VSSCBIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶜᶠ(i, j, k, ibg::VSSCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶜ(i, j, k, ibg::VSSCBIBG) = Δrᶠᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶠ(i, j, k, ibg::VSSCBIBG) = Δrᶜᶠᶠ(i, j, k, ibg)
@inline Δzᶠᶜᶠ(i, j, k, ibg::VSSCBIBG) = Δrᶠᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶠ(i, j, k, ibg::VSSCBIBG) = Δrᶠᶠᶠ(i, j, k, ibg)

#####
##### Column depths
#####

@inline staggered_column_depthᶠᶜᵃ(i, j, ibg, ib::ShavedCellBottom) = @inbounds rnode(i, j, ibg.Nz+1, ibg, c, c, f) - ib.west_bottom_height[i, j, 1]
@inline staggered_column_depthᶜᶠᵃ(i, j, ibg, ib::ShavedCellBottom) = @inbounds rnode(i, j, ibg.Nz+1, ibg, c, c, f) - ib.south_bottom_height[i, j, 1]

#####
##### Reconstruction and comparison
#####

function Grids.constructor_arguments(grid::SCBIBG)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid.underlying_grid)
    shaved_cell_bottom_args = Dict(:bottom_height => Field{Face, Face, Nothing}(grid.underlying_grid; data=grid.immersed_boundary.corner_bottom_height),
                                   :minimum_fractional_cell_height => grid.immersed_boundary.minimum_fractional_cell_height)
    return underlying_grid_args, underlying_grid_kwargs, shaved_cell_bottom_args
end

function Base.:(==)(scb1::ShavedCellBottom, scb2::ShavedCellBottom)
    return bottom_heights_equal(scb1.bottom_height, scb2.bottom_height) &&
           bottom_heights_equal(scb1.west_bottom_height, scb2.west_bottom_height) &&
           bottom_heights_equal(scb1.south_bottom_height, scb2.south_bottom_height) &&
           bottom_heights_equal(scb1.corner_bottom_height, scb2.corner_bottom_height) &&
           scb1.minimum_fractional_cell_height == scb2.minimum_fractional_cell_height
end


#####
##### Faces cut away by the shaved surface: the flux area vanishes even when both neighbouring cells stay active, so the
##### face is peripheral and carries no velocity.
#####

@inline function closed_face_x(i, j, k, ibg::SCBIBG)
    grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.west_bottom_height[i, j, 1]
    return rnode(i, j, k+1, grid, c, c, f) ≤ rᵇ
end

@inline function closed_face_y(i, j, k, ibg::SCBIBG)
    grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.south_bottom_height[i, j, 1]
    return rnode(i, j, k+1, grid, c, c, f) ≤ rᵇ
end

@inline peripheral_node(i, j, k, ibg::SCBIBG, ::Face, ::Center, ::Center) =
    inactive_cell(i, j, k, ibg) | inactive_cell(i-1, j, k, ibg) | closed_face_x(i, j, k, ibg)

@inline peripheral_node(i, j, k, ibg::SCBIBG, ::Center, ::Face, ::Center) =
    inactive_cell(i, j, k, ibg) | inactive_cell(i, j-1, k, ibg) | closed_face_y(i, j, k, ibg)
