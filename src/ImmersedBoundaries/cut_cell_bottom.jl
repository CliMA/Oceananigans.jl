using Oceananigans.Fields: Field, fill_halo_regions!, set!
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

"""
    CutCellBottom(bottom_height; minimum_fractional_cell_height=0.2)

Return a `CutCellBottom` representing an immersed boundary with cut bottom cells, inspired by
the shaved cells of [Adcroft et al. (1997)](@cite AdcroftHillMarshall1997).

The bottom is a piecewise-bilinear surface reconstructed from `bottom_height`, which may be a
`Field`, `Array`, or function of `(x, y)`. Every lateral face of a bottom cell is cut by that
surface at the horizontal position of the face itself, so the faces of one cell carry different
heights and the bottom slopes through the cell. The height of the cell is the mean of the heights of
its lateral faces, so that the faces bound the cell volume. When `bottom_height` is sampled at cell
centers, a column at or above the top of the grid stays dry and does not enter the surface of its neighbors.

Each lateral face is cut in the lowest level it reaches, and the opening it leaves there is rounded to the nearest of
zero and

```
minimum_fractional_cell_height * Δz,
```

where `Δz` is the height of that level: an opening thinner than half of it closes the face in that level, a thicker
one is at least that tall. A cell is as tall as the mean opening of its `N` lateral faces (four, or two in a grid with
a `Flat` horizontal direction). When that mean is shorter than `minimum_fractional_cell_height * Δz`, the faces cut in
the level of the cell are deepened until it is not, up to `Δz`, so that a cell with `n` open faces is at least as tall
as the smaller of `minimum_fractional_cell_height * Δz` and `n * Δz / N`. A slope steeper than one level per cell cuts
the faces of a column in different levels, so cut cells are stacked in a staircase.
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

const maximum_cutting_iterations = 20

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
    @apply_regionally launch!(arch, grid, :xy, _average_corners_to_centers!, bottom_field, corner_field, center_field, grid, ϵ)
    fill_halo_regions!(bottom_field)

    west_field = Field{Face, Center, Nothing}(grid)
    south_field = Field{Center, Face, Nothing}(grid)
    previous_bottom_field = Field{Center, Center, Nothing}(grid)
    required_opening = Field{Center, Center, Center}(grid)

    compute_ib = CutCellBottom(bottom_field, nothing, nothing, nothing, ϵ, nothing)

    TX, TY, _ = topology(grid)
    wˣ = TX === Flat && TY !== Flat ? 0 : 1
    wʸ = TY === Flat && TX !== Flat ? 0 : 1

    # Immersing a cell can close the faces of its neighbors, and deepening a face moves the cells on both sides of
    # it, so faces and cells are recomputed until no level changes.
    converged = false
    iteration = 0
    while !converged && iteration < maximum_cutting_iterations
        iteration += 1
        set!(previous_bottom_field, bottom_field)

        @apply_regionally launch!(arch, grid, parameters, _compute_cut_face_bottom_heights!, west_field, south_field, corner_field, grid, compute_ib)

        fill_halo_regions!(west_field)
        fill_halo_regions!(south_field)

        # the level a cell is cut in follows from its faces, so the requirement is posted against a boundary that carries them
        face_ib = CutCellBottom(bottom_field, west_field.data, south_field.data, corner_field.data, ϵ, nothing)

        @apply_regionally launch!(arch, grid, :xyz, _require_open_face_depths!, required_opening, west_field, south_field, grid, face_ib, ϵ, wˣ, wʸ)

        fill_halo_regions!(required_opening)

        @apply_regionally launch!(arch, grid, parameters, _deepen_open_faces!, west_field, south_field, required_opening, grid)

        fill_halo_regions!(west_field)
        fill_halo_regions!(south_field)

        @apply_regionally launch!(arch, grid, :xy, _average_faces_to_centers!, bottom_field, west_field, south_field, grid, compute_ib, wˣ, wʸ)

        fill_halo_regions!(bottom_field)
        converged = maximum(abs, bottom_field - previous_bottom_field) == 0
    end

    ib = CutCellBottom(bottom_field.data, west_field.data, south_field.data, corner_field.data, ϵ, nothing)
    bottom_cell_index = Field{Center, Center, Nothing}(grid, index_type(grid))
    @apply_regionally launch!(arch, grid, :xy, _compute_bottom_cell_index!, bottom_cell_index, grid, ib)
    fill_halo_regions!(bottom_cell_index)

    return CutCellBottom(bottom_field.data, west_field.data, south_field.data, corner_field.data, ϵ, bottom_cell_index.data)
end

@kernel function _compute_bottom_cell_index!(bottom_cell_index, grid, ib)
    i, j = @index(Global, NTuple)
    @inbounds bottom_cell_index[i, j, 1] = bottom_cell(i, j, grid, ib)
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

    rᵗ = rnode(i, j, grid.Nz+1, grid, c, c, f)
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

@kernel function _compute_cut_face_bottom_heights!(west_field, south_field, corner_field, grid, ib)
    i, j = @index(Global, NTuple)

    iᴱ = x_index(i, grid, +1)
    jᴺ = y_index(j, grid, +1)

    ϵ = ib.minimum_fractional_cell_height

    rᶠᶜ = @inbounds (corner_field[i, j, 1] + corner_field[i, jᴺ, 1]) / 2
    rᶜᶠ = @inbounds (corner_field[i, j, 1] + corner_field[iᴱ, j, 1]) / 2

    # No face leaves an opening thinner than ϵ Δr, which would wreck the conditioning of the free-surface solve
    @inbounds west_field[i, j, 1]  = rounded_bottom_height(i, j, grid, rᶠᶜ, ϵ)
    @inbounds south_field[i, j, 1] = rounded_bottom_height(i, j, grid, rᶜᶠ, ϵ)
end

# A cell is ϵ Δr tall when each face cut in its level leaves (faces ϵ - full_faces) Δr / (open_faces - full_faces)
@kernel function _require_open_face_depths!(required_opening, west_field, south_field, grid, ib, ϵ, wˣ, wʸ)
    i, j, k = @index(Global, NTuple)

    iᴱ = x_index(i, grid, +1)
    jᴺ = y_index(j, grid, +1)

    r⁺ = rnode(i, j, k+1, grid, c, c, f)
    Δr = Δrᶜᶜᶜ(i, j, k, grid)

    oᵂ = @inbounds face_open_height( west_field[i,  j,  1], r⁺, Δr)
    oᴱ = @inbounds face_open_height( west_field[iᴱ, j,  1], r⁺, Δr)
    oˢ = @inbounds face_open_height(south_field[i,  j,  1], r⁺, Δr)
    oᴺ = @inbounds face_open_height(south_field[i,  jᴺ, 1], r⁺, Δr)

    # in a Flat direction the two faces are one phantom face, which the cell does not have
    faces = 2 * (wˣ + wʸ)
    open_faces = wˣ * ((oᵂ > zero(Δr)) + (oᴱ > zero(Δr))) + wʸ * ((oˢ > zero(Δr)) + (oᴺ > zero(Δr)))
    full_faces = wˣ * ((oᵂ ≥ Δr) + (oᴱ ≥ Δr)) + wʸ * ((oˢ ≥ Δr) + (oᴺ ≥ Δr))
    Σo = wˣ * (oᵂ + oᴱ) + wʸ * (oˢ + oᴺ)

    thin = (Σo < faces * ϵ * Δr) & !_immersed_cell(i, j, k, grid, ib)

    # a face cannot leave more open than the level it sits in, so the guarantee saturates at open_faces * Δr / faces
    opening = min((faces * ϵ - full_faces) * Δr / max(open_faces - full_faces, 1), Δr)

    @inbounds required_opening[i, j, k] = ifelse(thin, opening, zero(Δr))
end

# A face is deepened only in the level it is cut in, as far as the cell on either side of it requires.
@inline function deepened_face_height(i, j, rᵇ, i⁻, j⁻, grid, required_opening)
    for k in 1:grid.Nz
        r⁻ = rnode(i, j, k,   grid, c, c, f)
        r⁺ = rnode(i, j, k+1, grid, c, c, f)
        opening = @inbounds max(required_opening[i⁻, j⁻, k], required_opening[i, j, k])
        rᵇ = ifelse(r⁻ ≤ rᵇ < r⁺, min(rᵇ, max(r⁻, r⁺ - opening)), rᵇ)
    end
    return rᵇ
end

@kernel function _deepen_open_faces!(west_field, south_field, required_opening, grid)
    i, j = @index(Global, NTuple)

    iᵂ = x_index(i, grid, -1)
    jˢ = y_index(j, grid, -1)

    @inbounds west_field[i, j, 1] = deepened_face_height(i, j, west_field[i, j, 1], iᵂ, j, grid, required_opening)
    @inbounds south_field[i, j, 1] = deepened_face_height(i, j, south_field[i, j, 1], i, jˢ, grid, required_opening)
end

# A face cut in a higher level is closed in this one, so clipping the faces into the level of the cell makes them bound its volume.
@kernel function _average_faces_to_centers!(bottom_field, west_field, south_field, grid, ib, wˣ, wʸ)
    i, j = @index(Global, NTuple)

    iᴱ = x_index(i, grid, +1)
    jᴺ = y_index(j, grid, +1)

    kᶜ = bottom_cell(i, j, grid, ib)

    rˣ = @inbounds ( west_field[i, j, 1] +  west_field[iᴱ, j, 1]) / 2
    rʸ = @inbounds (south_field[i, j, 1] + south_field[i, jᴺ, 1]) / 2
    rᵇ = (wˣ * rˣ + wʸ * rʸ) / (wˣ + wʸ)

    @inbounds bottom_field[i, j, 1] = ifelse(kᶜ > grid.Nz, bottom_field[i, j, 1], rᵇ)
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

#####
##### Immersed cells and grid spacings
#####

# Open height of a face in level k, from the bottom height sampled at that face
@inline face_open_height(rᵇ, r⁺, Δr) = clamp(r⁺ - rᵇ, zero(Δr), Δr)

# The mean opening the lateral faces of a cell leave in level k; in a Flat direction its two faces are one phantom face
@inline function mean_face_opening(i, j, k, grid, ib, r⁺, Δr)
    TX, TY, _ = topology(grid)
    wˣ = ifelse((TX === Flat) & (TY !== Flat), 0, 1)
    wʸ = ifelse((TY === Flat) & (TX !== Flat), 0, 1)
    oˣ = @inbounds face_open_height(ib.west_bottom_height[i, j, 1], r⁺, Δr) + face_open_height(ib.west_bottom_height[x_index(i, grid, +1), j, 1], r⁺, Δr)
    oʸ = @inbounds face_open_height(ib.south_bottom_height[i, j, 1], r⁺, Δr) + face_open_height(ib.south_bottom_height[i, y_index(j, grid, +1), 1], r⁺, Δr)
    return (wˣ * oˣ + wʸ * oʸ) / (2 * (wˣ + wʸ))
end

@inline mean_face_opening(i, j, k, grid, ::CutCellBottom{<:Any, Nothing}, r⁺, Δr) = zero(grid)

# Before the bottom index is stored, a cell is immersed when neither its center nor any of its faces leaves an opening in its level;
# land stays dry
@inline function _immersed_cell(i, j, k, grid, ib::CutCellBottom{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing})
    r⁺ = rnode(i, j, k + 1, grid, c, c, f)
    Δr = Δrᶜᶜᶜ(i, j, k, grid)
    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    dry = rᵇ ≥ rnode(i, j, grid.Nz + 1, grid, c, c, f)
    return ((r⁺ - Δr * ib.minimum_fractional_cell_height) < rᵇ) & ((mean_face_opening(i, j, k, grid, ib, r⁺, Δr) == 0) | dry)
end

@inline _immersed_cell(i, j, k, underlying_grid, ib::CutCellBottom) = k < bottom_cell(i, j, underlying_grid, ib)

# The first active level of a column, or Nz + 1 for a dry column: searched before the index is stored, read from it after
@inline function bottom_cell(i, j, grid, ib::CutCellBottom{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing})
    kᵇ = grid.Nz + 1
    for k in grid.Nz:-1:1
        kᵇ = ifelse(_immersed_cell(i, j, k, grid, ib), kᵇ, k)
    end
    return kᵇ
end

@inline bottom_cell(i, j, grid, ib::CutCellBottom) = @inbounds ib.bottom_cell_index[i, j, 1]

# An active cell is as tall as the mean opening of its faces; immersed cells keep the full level height
@inline function Δrᶜᶜᶜ(i, j, k, ibg::CCBIBG)
    grid = ibg.underlying_grid
    Δr = Δrᶜᶜᶜ(i, j, k, grid)
    opening = mean_face_opening(i, j, k, grid, ibg.immersed_boundary, rnode(i, j, k + 1, grid, c, c, f), Δr)
    return ifelse(_immersed_cell(i, j, k, grid, ibg.immersed_boundary), Δr, opening)
end

@inline function Δrᶠᶜᶜ(i, j, k, ibg::CCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.west_bottom_height[i, j, 1]
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)
    Δr = Δrᶠᶜᶜ(i, j, k, underlying_grid)
    cut = clamp(r⁺ - rᵇ, zero(Δr), Δr)
    return ifelse(r⁺ ≤ rᵇ, Δr, cut)
end

@inline function Δrᶜᶠᶜ(i, j, k, ibg::CCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.south_bottom_height[i, j, 1]
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
    rᵇ = @inbounds ibg.immersed_boundary.west_bottom_height[i, j, 1]
    return cut_center_spacing(i, j, k, underlying_grid, rᵇ, Δrᶠᶜᶠ(i, j, k, underlying_grid))
end

@inline function Δrᶜᶠᶠ(i, j, k, ibg::CCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.south_bottom_height[i, j, 1]
    return cut_center_spacing(i, j, k, underlying_grid, rᵇ, Δrᶜᶠᶠ(i, j, k, underlying_grid))
end

@inline Δrᶠᶠᶠ(i, j, k, ibg::CCBIBG) = min(Δrᶠᶜᶠ(i, j-1, k, ibg), Δrᶠᶜᶠ(i, j, k, ibg))

# Flat topologies collapse the staggered metrics onto the centered ones.
XFlatCCBIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:CutCellBottom}
YFlatCCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:CutCellBottom}

@inline Δrᶠᶜᶜ(i, j, k, ibg::XFlatCCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶜᶠ(i, j, k, ibg::XFlatCCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶜᶠᶜ(i, j, k, ibg::YFlatCCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶜᶠᶠ(i, j, k, ibg::YFlatCCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XFlatCCBIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::YFlatCCBIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶠ(i, j, k, ibg::XFlatCCBIBG) = Δrᶜᶠᶠ(i, j, k, ibg)
@inline Δrᶠᶠᶠ(i, j, k, ibg::YFlatCCBIBG) = Δrᶠᶜᶠ(i, j, k, ibg)

XYFlatCCBIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Flat, <:Any, <:Any, <:CutCellBottom}

@inline Δrᶠᶠᶜ(i, j, k, ibg::XYFlatCCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶠ(i, j, k, ibg::XYFlatCCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)

# Vertically-static, cut cell bottom, immersed boundary grid
VSCCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractStaticGrid, <:CutCellBottom}
@inline Δzᶜᶜᶜ(i, j, k, ibg::VSCCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δzᶠᶜᶜ(i, j, k, ibg::VSCCBIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶜ(i, j, k, ibg::VSCCBIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶜᶠ(i, j, k, ibg::VSCCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶜ(i, j, k, ibg::VSCCBIBG) = Δrᶠᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶠ(i, j, k, ibg::VSCCBIBG) = Δrᶜᶠᶠ(i, j, k, ibg)
@inline Δzᶠᶜᶠ(i, j, k, ibg::VSCCBIBG) = Δrᶠᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶠ(i, j, k, ibg::VSCCBIBG) = Δrᶠᶠᶠ(i, j, k, ibg)

#####
##### Column depths and reciprocal spacings
#####

@inline top_cell(i, j, grid, ::CutCellBottom) = size(grid, 3)
@inline bottom_height(i, j, ibg::CCBIBG) = @inbounds ibg.immersed_boundary.bottom_height[i, j, 1]

# A face column reaches down to the bottom height sampled at that face, and is dry next to a dry column
@inline function face_column_depth(i, j, i⁻, j⁻, ibg, ib::CutCellBottom)
    rᵇ = @inbounds ifelse(i⁻ == i, ib.south_bottom_height[i, j, 1], ib.west_bottom_height[i, j, 1])
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

#####
##### Reconstruction and comparison
#####

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


#####
##### A face the bottom closes is peripheral even between two active cells, as on a crest, and carries no velocity
#####

@inline peripheral_node(i, j, k, ibg::CCBIBG, ::Face, ::Center, ::Center) = inactive_cell(i, j, k, ibg) | inactive_cell(i-1, j, k, ibg) |
                                                                           (rnode(i, j, k+1, ibg.underlying_grid, c, c, f) ≤ @inbounds ibg.immersed_boundary.west_bottom_height[i, j, 1])

@inline peripheral_node(i, j, k, ibg::CCBIBG, ::Center, ::Face, ::Center) = inactive_cell(i, j, k, ibg) | inactive_cell(i, j-1, k, ibg) |
                                                                           (rnode(i, j, k+1, ibg.underlying_grid, c, c, f) ≤ @inbounds ibg.immersed_boundary.south_bottom_height[i, j, 1])

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
