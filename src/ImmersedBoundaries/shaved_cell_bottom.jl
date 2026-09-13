using Oceananigans.Fields: Field, fill_halo_regions!, set!
using Oceananigans.Grids: Grids, AbstractStaticGrid, constructor_arguments, XFlatGrid, YFlatGrid
using Oceananigans.Utils: prettysummary, KernelParameters

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

Return a `ShavedCellBottom` representing an immersed boundary with "shaved" bottom cells,
following Adcroft, Hill and Marshall (1997).

The bottom is a piecewise-bilinear surface reconstructed from `bottom_height`, which may be a
`Field`, `Array`, or function of `(x, y)`. Every lateral face of a bottom cell is cut by that
surface at the horizontal position of the face itself, so the faces of one cell carry different
heights and the bottom slopes through the cell. The volume of the cell is the volume underneath the
same surface.

The height of a shaved cell, and of each of its lateral faces, is greater than

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

"""
$(TYPEDSIGNATURES)

Return a `Field` at `(Face, Face, Nothing)` sharing the corner bottom heights of `grid`, the
bilinear surface every shaved cell volume and face area is built from.
"""
corner_bottom_height_field(grid::IBG) = Field{Face, Face, Nothing}(grid.underlying_grid; data = grid.immersed_boundary.corner_bottom_height)

# Staggered fields carry a boundary point beyond the last cell in Bounded directions.
function staggered_bottom_parameters(grid)
    Nx, Ny, _ = size(grid)
    TX, TY, _ = topology(grid)

    Ix = TX === Flat ? (1:1) : (1:Nx+1)
    Iy = TY === Flat ? (1:1) : (1:Ny+1)

    return KernelParameters(Ix, Iy)
end

set_corner_bottom_height!(corner_field, grid, bottom_height, parameters) = set_bottom_height!(corner_field, bottom_height)

# Anything sampled at cell centers is interpolated to the corners.
function set_corner_bottom_height!(corner_field, grid, bottom_height::AbstractArray, parameters)
    center_field = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(center_field, bottom_height)
    fill_halo_regions!(center_field)
    launch!(architecture(grid), grid, parameters, _interpolate_bottom_height_to_corners!, corner_field, center_field, grid)
    return corner_field
end

set_corner_bottom_height!(corner_field, grid, bottom_height::Field{Face, Face, Nothing}, parameters) =
    set_bottom_height!(corner_field, bottom_height)

# Rebuilding a materialized boundary reuses the surface it was built from, so that it is idempotent.
@inline bottom_surface(ib::ShavedCellBottom{<:Any, <:AbstractArray}, grid) = Field{Face, Face, Nothing}(grid; data=ib.corner_bottom_height)
@inline bottom_surface(ib::ShavedCellBottom, grid) = ib.bottom_height

@kernel function _interpolate_bottom_height_to_corners!(corner_field, center_field, grid)
    i, j = @index(Global, NTuple)
    iᵂ = x_index(i, grid, -1)
    jˢ = y_index(j, grid, -1)
    @inbounds corner_field[i, j, 1] = (center_field[iᵂ, jˢ, 1] + center_field[i, jˢ, 1] +
                                       center_field[iᵂ, j,  1] + center_field[i, j,  1]) / 4
end

function materialize_immersed_boundary(grid, ib::ShavedCellBottom)
    ϵ = convert(eltype(grid), ib.minimum_fractional_cell_height)
    arch = architecture(grid)
    parameters = staggered_bottom_parameters(grid)

    corner_field = Field{Face, Face, Nothing}(grid)
    @apply_regionally set_corner_bottom_height!(corner_field, grid, bottom_surface(ib, grid), parameters)
    @apply_regionally launch!(arch, grid, parameters, _clamp_bottom_height_to_domain!, corner_field, grid)
    fill_halo_regions!(corner_field)

    bottom_field = Field{Center, Center, Nothing}(grid)
    @apply_regionally launch!(arch, grid, :xy, _average_corners_to_centers!, bottom_field, corner_field, grid, ϵ)
    fill_halo_regions!(bottom_field)

    west_field = Field{Face, Center, Nothing}(grid)
    south_field = Field{Center, Face, Nothing}(grid)

    compute_ib = ShavedCellBottom(bottom_field, nothing, nothing, nothing, ϵ)

    @apply_regionally launch!(arch, grid, parameters, _compute_shaved_face_bottom_heights!,
                              west_field, south_field, corner_field, grid, compute_ib)

    fill_halo_regions!(west_field)
    fill_halo_regions!(south_field)

    return ShavedCellBottom(bottom_field.data, west_field.data, south_field.data, corner_field.data, ϵ)
end

@kernel function _clamp_bottom_height_to_domain!(bottom_field, grid)
    i, j = @index(Global, NTuple)
    rᵈ = rnode(i, j, 1, grid, c, c, f)
    rᵗ = rnode(i, j, grid.Nz+1, grid, c, c, f)
    @inbounds bottom_field[i, j, 1] = clamp(bottom_field[i, j, 1], rᵈ, rᵗ)
end

# The mean of a bilinear surface over a cell is the mean of its four corners; the ϵ-limiter follows.
@kernel function _average_corners_to_centers!(bottom_field, corner_field, grid, ϵ)
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

    @inbounds bottom_field[i, j, 1] = rᵇ
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
# of that level. The floor stays positive so that the surface lies inside a level even for ϵ = 0.
@inline function shaved_face_bottom_height(i, j, kᵇ, grid, rᵇ, ϵ)
    FT = eltype(grid)
    k  = min(kᵇ, grid.Nz)
    r⁻ = rnode(i, j, k,   grid, c, c, f)
    r⁺ = rnode(i, j, k+1, grid, c, c, f)
    Δr = Δrᶜᶜᶜ(i, j, k, grid)
    return clamp(rᵇ, r⁻, r⁺ - max(ϵ, sqrt(eps(FT))) * Δr)
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

    @inbounds west_field[i, j, 1]  = shaved_face_bottom_height(i, j, max(kᵂ, kᶜ), grid, rᶠᶜ, ϵ)
    @inbounds south_field[i, j, 1] = shaved_face_bottom_height(i, j, max(kˢ, kᶜ), grid, rᶜᶠ, ϵ)
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

"""
A shaved bottom cell in the x-r plane: the surface crosses the cell, so its two lateral faces are cut
at different heights and its volume is the volume underneath the surface.

           i-1/2           i+1/2
             |               |
     r⁺  ----+---------------+----     ↑
             | ╲             |         |
        rᵇᵂ  +   ╲     ∘     |         | Δrᶜᶜᶜ = r⁺ - rᵇᶜᶜ
             |     ╲         + rᵇᴱ     |
     r⁻  ----+-------╲-------+----     ↓
             ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒

    Δrᶠᶜᶜ = r⁺ - rᵇᵂ  at  i-1/2,   r⁺ - rᵇᴱ  at  i+1/2,   and   rᵇᶜᶜ = (rᵇᵂ + rᵇᴱ) / 2

A cell is immersed when `rᵇᶜᶜ > r⁺ - ϵ Δr`.
"""
@inline function _immersed_cell(i, j, k, underlying_grid, ib::ShavedCellBottom)
    r⁺ = rnode(i, j, k + 1, underlying_grid, c, c, f)
    ϵ  = ib.minimum_fractional_cell_height
    Δr = Δrᶜᶜᶜ(i, j, k, underlying_grid)
    r★ = r⁺ - Δr * ϵ
    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    return r★ < rᵇ
end

# A cell or a face keeps the full height of its level except in the level the surface cuts, where it
# keeps the part above the surface. Levels below keep the full height: they are masked, and a
# positive height keeps them out of denominators.

@inline function Δrᶜᶜᶜ(i, j, k, ibg::SCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.bottom_height[i, j, 1]
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)
    return ifelse(shaved_level(i, j, k, underlying_grid, rᵇ), r⁺ - rᵇ, Δrᶜᶜᶜ(i, j, k, underlying_grid))
end

@inline function Δrᶠᶜᶜ(i, j, k, ibg::SCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.west_bottom_height[i, j, 1]
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)
    return ifelse(shaved_level(i, j, k, underlying_grid, rᵇ), r⁺ - rᵇ, Δrᶠᶜᶜ(i, j, k, underlying_grid))
end

@inline function Δrᶜᶠᶜ(i, j, k, ibg::SCBIBG)
    underlying_grid = ibg.underlying_grid
    rᵇ = @inbounds ibg.immersed_boundary.south_bottom_height[i, j, 1]
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)
    return ifelse(shaved_level(i, j, k, underlying_grid, rᵇ), r⁺ - rᵇ, Δrᶜᶠᶜ(i, j, k, underlying_grid))
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

# The staggered column depths follow the surface at the face, matching the sum of the face heights.
@inline staggered_column_depthᶠᶜᵃ(i, j, ibg, ib::ShavedCellBottom) = @inbounds rnode(i, j, ibg.Nz+1, ibg, c, c, f) - ib.west_bottom_height[i, j, 1]
@inline staggered_column_depthᶜᶠᵃ(i, j, ibg, ib::ShavedCellBottom) = @inbounds rnode(i, j, ibg.Nz+1, ibg, c, c, f) - ib.south_bottom_height[i, j, 1]

#####
##### Reconstruction and comparison
#####

function Grids.constructor_arguments(grid::SCBIBG)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid.underlying_grid)
    shaved_cell_bottom_args = Dict(:bottom_height => corner_bottom_height_field(grid),
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
