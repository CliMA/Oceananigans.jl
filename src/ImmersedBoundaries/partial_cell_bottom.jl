using Oceananigans.Fields: Field, fill_halo_regions!, set!
using Oceananigans.Grids: Grids, AbstractStaticGrid, constructor_arguments
using Oceananigans.Utils: prettysummary

import Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶜᶜᶠ, Δrᶜᶠᶜ, Δrᶜᶠᶠ, Δrᶠᶜᶜ, Δrᶠᶜᶠ, Δrᶠᶠᶜ, Δrᶠᶠᶠ,
                               Δzᶜᶜᶜ, Δzᶜᶜᶠ, Δzᶜᶠᶜ, Δzᶜᶠᶠ, Δzᶠᶜᶜ, Δzᶠᶜᶠ, Δzᶠᶠᶜ, Δzᶠᶠᶠ

#####
##### PartialCellVerticalBoundaries
#####

struct PartialCellVerticalBoundaries{H, T, E, SB, ST, L, D} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    top_height :: T
    minimum_fractional_cell_height :: E
    fractional_bottom_index :: SB
    fractional_top_index :: ST
    top_load :: L
    face_column_depths :: D
end

PartialCellVerticalBoundaries(bottom_height, top_height, minimum_fractional_cell_height) =
    PartialCellVerticalBoundaries(bottom_height, top_height, minimum_fractional_cell_height, nothing, nothing, nothing, nothing)

PartialCellVerticalBoundaries(bottom_height, top_height, minimum_fractional_cell_height, top_load) =
    PartialCellVerticalBoundaries(bottom_height, top_height, minimum_fractional_cell_height, nothing, nothing, top_load, nothing)

# A missing top is materialized as a `ConstantField` at the top of the domain, without a `fractional_top_index`
const PartialCellBottom{H, E} = PartialCellVerticalBoundaries{H, <:Union{Nothing, ConstantField}, E, <:Any, Nothing}

# Dispatches on `fractional_top_index` because kernels see no heights, see `Adapt.adapt_structure`
const PartialCellWithTop = PartialCellVerticalBoundaries{<:Any, <:Any, <:Any, <:Any, <:AbstractArray}

PartialCellBottom(bottom_height, minimum_fractional_cell_height) = PartialCellVerticalBoundaries(bottom_height, nothing, minimum_fractional_cell_height)

const PCBIBG{FT, TX, TY, TZ} = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:Any, <:PartialCellVerticalBoundaries} where {FT, TX, TY, TZ}

# PartialCellVerticalBoundaries with a materialized immersed top
const PCBTIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:PartialCellWithTop}

immersed_boundary_name(::PartialCellBottom) = "PartialCellBottom"
immersed_boundary_name(::PartialCellVerticalBoundaries) = "PartialCellVerticalBoundaries"

function Base.summary(ib::PartialCellVerticalBoundaries)
    zmean, zmin, zmax = height_statistics(ib.bottom_height)

    summary1 = immersed_boundary_name(ib) * "("

    summary2 = string("mean(zb)=", prettysummary(zmean),
                      ", min(zb)=", prettysummary(zmin),
                      ", max(zb)=", prettysummary(zmax),
                      top_height_summary(ib.top_height),
                      ", ϵ=", prettysummary(ib.minimum_fractional_cell_height))

    summary3 = ")"

    return summary1 * summary2 * summary3
end

Base.summary(ib::PartialCellVerticalBoundaries{<:Function}) = @sprintf("%s(%s%s, ϵ=%.1f)",
                                                                       immersed_boundary_name(ib),
                                                                       prettysummary(ib.bottom_height, false),
                                                                       top_height_summary(ib.top_height),
                                                                       ib.minimum_fractional_cell_height)

Base.summary(ib::PartialCellVerticalBoundaries{Nothing}) = @sprintf("%s(nothing%s, ϵ=%.1f)",
                                                                    immersed_boundary_name(ib),
                                                                    top_height_summary(ib.top_height),
                                                                    ib.minimum_fractional_cell_height)

function Base.show(io::IO, ib::PartialCellBottom)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "└── minimum_fractional_cell_height: ", prettysummary(ib.minimum_fractional_cell_height))
end

function Base.show(io::IO, ib::PartialCellVerticalBoundaries)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "├── top_height: ", prettysummary(ib.top_height), '\n')
    print(io, "├── top_load: ", prettysummary(ib.top_load), '\n')
    print(io, "└── minimum_fractional_cell_height: ", prettysummary(ib.minimum_fractional_cell_height))
end

"""
    PartialCellVerticalBoundaries(; bottom_height=nothing, top_height=nothing, minimum_fractional_cell_height=0.2, top_load=nothing)

Return `PartialCellVerticalBoundaries` representing an immersed boundary with "partial"
bottom cells. That is, the height of the bottommost cell in each column is reduced
to fit the provided `bottom_height`, which may be a `Field`, `Array`, or function
of `(x, y)`. If `bottom_height` is `nothing`, the bottom is the bottom of the domain.

If `top_height` is provided (e.g. an ice-shelf draft), cells above it are immersed
and the height of the topmost cell in each column is reduced to fit it in the same way.

The height of partial cells is greater than

```
minimum_fractional_cell_height * Δz,
```

where `Δz` is the original height of the cell in the underlying grid.
Columns in which the top and bottom leave less than this height are immersed entirely.

`top_load` is the potential of the weight of the solid top, added to the hydrostatic pressure of
every column: either `nothing` (default, no load), a [`TopLoad`](@ref) computed from a reference
state, or an array, function of `(x, y)` or two-dimensional field such as the one returned by
[`top_load_potential`](@ref). It requires `top_height`.

Example
=======

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 8, 10), x=(0, 100), y=(0, 100), z=(-100, 0));

julia> ImmersedBoundaryGrid(grid, PartialCellVerticalBoundaries(bottom_height=-95, top_height=(x, y) -> -20 - 0.2y))
2×8×10 ImmersedBoundaryGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo:
├── immersed_boundary: PartialCellVerticalBoundaries(mean(zb)=-95.0, min(zb)=-95.0, max(zb)=-95.0, mean(zt)=-29.8125, min(zt)=-38.0, max(zt)=-21.25, ϵ=0.2)
├── underlying_grid: 2×8×10 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── Periodic x ∈ [0.0, 100.0)  regularly spaced with Δx=50.0
├── Periodic y ∈ [0.0, 100.0)  regularly spaced with Δy=12.5
└── Bounded  z ∈ [-100.0, 0.0] regularly spaced with Δz=10.0
```
"""
function PartialCellVerticalBoundaries(; bottom_height=nothing, top_height=nothing, minimum_fractional_cell_height=0.2, top_load=nothing)
    validate_top_load(top_height, top_load)
    return PartialCellVerticalBoundaries(bottom_height, top_height, minimum_fractional_cell_height, top_load)
end

"""
    PartialCellBottom(bottom_height=nothing; top_height=nothing, minimum_fractional_cell_height=0.2, top_load=nothing)

Return a [`PartialCellVerticalBoundaries`](@ref) bounded below by `bottom_height`. See
[`PartialCellVerticalBoundaries`](@ref) for the meaning of the keyword arguments.

Example
=======

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 8, 10), x=(0, 100), y=(0, 100), z=(-100, 0));

julia> ImmersedBoundaryGrid(grid, PartialCellBottom(-95))
2×8×10 ImmersedBoundaryGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo:
├── immersed_boundary: PartialCellBottom(mean(zb)=-95.0, min(zb)=-95.0, max(zb)=-95.0, ϵ=0.2)
├── underlying_grid: 2×8×10 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── Periodic x ∈ [0.0, 100.0)  regularly spaced with Δx=50.0
├── Periodic y ∈ [0.0, 100.0)  regularly spaced with Δy=12.5
└── Bounded  z ∈ [-100.0, 0.0] regularly spaced with Δz=10.0
```
"""
function PartialCellBottom(bottom_height=nothing; top_height=nothing, minimum_fractional_cell_height=0.2, top_load=nothing)
    validate_top_load(top_height, top_load)
    return PartialCellVerticalBoundaries(bottom_height, top_height, minimum_fractional_cell_height, top_load)
end

function materialize_immersed_boundary(grid, ib::PartialCellVerticalBoundaries)
    domain_bottom, domain_top = z_domain(grid)
    bottom = materialize_height(grid, ib.bottom_height, domain_bottom)
    top    = materialize_height(grid, ib.top_height, domain_top)

    minimum_fractional_cell_height = convert(eltype(grid), ib.minimum_fractional_cell_height)
    compute_ib = PartialCellVerticalBoundaries(bottom, top, minimum_fractional_cell_height)

    @apply_regionally compute_numerical_heights!(bottom, top, grid, compute_ib)

    fractional_bottom_index = Field{Center, Center, Nothing}(grid)
    fractional_top_index    = top_index_field(grid, top, eltype(grid))

    @apply_regionally compute_bottom_cell!(grid, compute_ib, fractional_bottom_index, bottom)
    @apply_regionally compute_top_cell!(grid, compute_ib, fractional_top_index, top)

    materialized_ib = PartialCellVerticalBoundaries(height_data(bottom), height_data(top), minimum_fractional_cell_height,
                                                    index_data(fractional_bottom_index), index_data(fractional_top_index), nothing, nothing)

    face_column_depths = materialize_face_column_depths(grid, materialized_ib)

    unloaded_ib = PartialCellVerticalBoundaries(height_data(bottom), height_data(top), minimum_fractional_cell_height,
                                                index_data(fractional_bottom_index), index_data(fractional_top_index), nothing, face_column_depths)

    top_load = materialize_top_load(grid, unloaded_ib, ib.top_load)

    return PartialCellVerticalBoundaries(height_data(bottom), height_data(top), minimum_fractional_cell_height,
                                         index_data(fractional_bottom_index), index_data(fractional_top_index), top_load, face_column_depths)
end

@kernel function _compute_numerical_heights!(bottom, top, grid, ib::PartialCellVerticalBoundaries)
    i, j = @index(Global, NTuple)

    domain_bottom = rnode(i, j, 1, grid, c, c, f)
    domain_top    = rnode(i, j, grid.Nz+1, grid, c, c, f)
    ẑᵇ = clamp(@inbounds(bottom[i, j, 1]), domain_bottom, domain_top)
    ẑᵗ = clamp(@inbounds(top[i, j, 1]), domain_bottom, domain_top)

    ϵ  = ib.minimum_fractional_cell_height
    Δzᵇ = zero(grid)

    for k in 1:grid.Nz
        z⁻ = rnode(i, j, k,   grid, c, c, f)
        z⁺ = rnode(i, j, k+1, grid, c, c, f)
        Δz = Δrᶜᶜᶜ(i, j, k, grid)
        bottom_cell = z⁻ ≤ ẑᵇ < z⁺
        top_cell    = z⁻ < ẑᵗ ≤ z⁺

        # If the size of a bottom or top cell is less than ϵ Δz, we enforce a minimum size of ϵ Δz
        ẑᵇ  = ifelse(bottom_cell, min(z⁺ - ϵ * Δz, ẑᵇ), ẑᵇ)
        ẑᵗ  = ifelse(top_cell,    max(z⁻ + ϵ * Δz, ẑᵗ), ẑᵗ)
        Δzᵇ = ifelse(bottom_cell, Δz, Δzᵇ)
    end

    closed = (ẑᵗ ≤ ẑᵇ) | thin_column(ẑᵇ, ẑᵗ, ϵ * Δzᵇ, top)
    store_column_heights!(bottom, top, i, j, grid, ẑᵇ, ẑᵗ, closed)
end

@inline thin_column(ẑᵇ, ẑᵗ, ϵΔzᵇ, top) = ẑᵗ - ẑᵇ < ϵΔzᵇ

# Without a top, the bottom cell already holds at least ϵ Δz; skipping the test avoids round-off closing it
@inline thin_column(ẑᵇ, ẑᵗ, ϵΔzᵇ, ::ConstantField) = false

# `fractional_bottom_index` is the position of the bottom in index space, sᵇ = kᵇ + 1 - φ, where kᵇ is the first active cell
# of the column and φ ∈ [ϵ, 1] the open fraction of its height; sᵇ = Nz + 1 in columns without active cells
@kernel function _compute_bottom_cell!(fractional_bottom_index, grid, ib::PartialCellVerticalBoundaries, bottom)
    i, j = @index(Global, NTuple)
    rᵇ = @inbounds bottom[i, j, 1]
    ϵ  = ib.minimum_fractional_cell_height

    kᵇ = grid.Nz + 1
    φ  = one(rᵇ)

    for k in grid.Nz:-1:1
        r⁺ = rnode(i, j, k+1, grid, c, c, f)
        Δr = Δrᶜᶜᶜ(i, j, k, grid)
        immersed_cell = r⁺ - Δr * ϵ < rᵇ

        kᵇ = ifelse(immersed_cell, kᵇ, k)
        # `min`: when Δr and r⁺ - r⁻ differ by an ulp, a bottom on a face gives φ > 1 and ⌊sᵇ⌋ = kᵇ - 1
        φ  = ifelse(immersed_cell, φ, min((r⁺ - rᵇ) / Δr, one(rᵇ)))
    end

    # Columns in the halos of `Bounded` directions keep full spacings
    φ = ifelse(inactive_cell(i, j, 1, grid), one(φ), φ)

    @inbounds fractional_bottom_index[i, j, 1] = kᵇ + 1 - φ
end

# `fractional_top_index` is sᵗ = kᵗ + 1 - φ, where kᵗ is the last active cell of the column and φ ∈ [ϵ, 1] the open
# fraction of its height, so that ⌊sᵗ⌋ = kᵗ; sᵗ = Nz in closed columns, or 0 without a stored bottom
@kernel function _compute_top_cell!(fractional_top_index, grid, ib::PartialCellVerticalBoundaries, top)
    i, j = @index(Global, NTuple)
    rᵗ = @inbounds top[i, j, 1]
    ϵ  = ib.minimum_fractional_cell_height

    kᵗ = 0
    φ  = one(rᵗ)

    for k in 1:grid.Nz
        r⁻ = rnode(i, j, k, grid, c, c, f)
        Δr = Δrᶜᶜᶜ(i, j, k, grid)
        # `r⁻ ≥ rᵗ` keeps φ > 0 for ϵ = 0, so that ⌊sᵗ⌋ = kᵗ
        immersed_cell = (r⁻ + Δr * ϵ > rᵗ) | (r⁻ ≥ rᵗ)

        kᵗ = ifelse(immersed_cell, kᵗ, k)
        φ  = ifelse(immersed_cell, φ, min((rᵗ - r⁻) / Δr, one(rᵗ)))
    end

    φ = ifelse(inactive_cell(i, j, 1, grid), one(φ), φ)

    @inbounds fractional_top_index[i, j, 1] = kᵗ + 1 - φ
end

# Kernels read the bottom and top through the fractional indices only, see `bottom_height(i, j, ibg)`
Adapt.adapt_structure(to, ib::PartialCellVerticalBoundaries) =
    PartialCellVerticalBoundaries(nothing,
                                  nothing,
                                  ib.minimum_fractional_cell_height,
                                  adapt(to, ib.fractional_bottom_index),
                                  adapt(to, ib.fractional_top_index),
                                  adapt(to, ib.top_load),
                                  adapt(to, ib.face_column_depths))

Architectures.on_architecture(to, ib::PartialCellVerticalBoundaries) =
    PartialCellVerticalBoundaries(on_architecture(to, ib.bottom_height),
                                  on_architecture(to, ib.top_height),
                                  on_architecture(to, ib.minimum_fractional_cell_height),
                                  on_architecture(to, ib.fractional_bottom_index),
                                  on_architecture(to, ib.fractional_top_index),
                                  on_architecture(to, ib.top_load),
                                  on_architecture(to, ib.face_column_depths))

# kᵇ = ⌊sᵇ⌋: `unsafe_trunc` is a single conversion, while mixed `Int`-`Float` comparisons of `sᵇ` are exact but several times longer
@inline bottom_cell(i, j, ib::PartialCellVerticalBoundaries) = @inbounds unsafe_trunc(Int, ib.fractional_bottom_index[i, j, 1])
@inline top_cell(i, j, ib::PartialCellVerticalBoundaries) = @inbounds unsafe_trunc(Int, ib.fractional_top_index[i, j, 1])

# Measured from the face below kᵇ, so that full bottom cells and columns without active cells are exact
@inline function bottom_height(i, j, ibg::PCBIBG)
    sᵇ = @inbounds ibg.immersed_boundary.fractional_bottom_index[i, j, 1]
    kᵇ = unsafe_trunc(Int, sᵇ)
    return rnode(i, j, kᵇ, ibg.underlying_grid, c, c, f) + (sᵇ - kᵇ) * Δrᶜᶜᶜ(i, j, kᵇ, ibg.underlying_grid)
end

# Measured from the face above kᵗ, so that full top cells and columns without an immersed top are exact
@inline function top_height(i, j, ibg::PCBTIBG)
    sᵗ = @inbounds ibg.immersed_boundary.fractional_top_index[i, j, 1]
    kᵗ = unsafe_trunc(Int, sᵗ)
    return rnode(i, j, kᵗ+1, ibg.underlying_grid, c, c, f) - (sᵗ - kᵗ) * Δrᶜᶜᶜ(i, j, kᵗ, ibg.underlying_grid)
end

@inline _immersed_cell(i, j, k, underlying_grid, ib::PartialCellWithTop) = (k < bottom_cell(i, j, ib)) | (k > top_cell(i, j, ib))

@inline function Δrᶜᶜᶜ(i, j, k, ibg::PCBIBG)
    sᵇ = @inbounds ibg.immersed_boundary.fractional_bottom_index[i, j, 1]
    at_the_bottom = k == unsafe_trunc(Int, sᵇ)

    full_Δr    = Δrᶜᶜᶜ(i, j, k, ibg.underlying_grid)
    partial_Δr = (k + 1 - sᵇ) * full_Δr

    return ifelse(at_the_bottom, partial_Δr, full_Δr)
end

@inline function Δrᶜᶜᶠ(i, j, k, ibg::PCBIBG)
    sᵇ = @inbounds ibg.immersed_boundary.fractional_bottom_index[i, j, 1]
    just_above_bottom = k - 1 == unsafe_trunc(Int, sᵇ)

    rᶜ = rnode(i, j, k, ibg.underlying_grid, c, c, c)
    rᶠ = rnode(i, j, k, ibg.underlying_grid, c, c, f)
    full_Δr    = Δrᶜᶜᶠ(i, j, k, ibg.underlying_grid)
    partial_Δr = rᶜ - rᶠ + (k - sᵇ) * Δrᶜᶜᶜ(i, j, k-1, ibg.underlying_grid) / 2

    return ifelse(just_above_bottom, partial_Δr, full_Δr)
end

# Only active cells are partial, so that immersed cells next to closed columns keep full spacings
@inline function partial_fractions(i, j, k, ib)
    sᵇ = @inbounds ib.fractional_bottom_index[i, j, 1]
    sᵗ = @inbounds ib.fractional_top_index[i, j, 1]
    kᵇ = unsafe_trunc(Int, sᵇ)
    kᵗ = unsafe_trunc(Int, sᵗ)
    active = (kᵇ ≤ k) & (k ≤ kᵗ)
    at_the_bottom = active & (k == kᵇ)
    at_the_top    = active & (k == kᵗ)
    φᵇ = ifelse(at_the_bottom, k + 1 - sᵇ, one(sᵇ))
    φᵗ = ifelse(at_the_top,    k + 1 - sᵗ, one(sᵗ))
    return φᵇ, φᵗ, at_the_bottom, at_the_top
end

@inline function Δrᶜᶜᶜ(i, j, k, ibg::PCBTIBG)
    φᵇ, φᵗ, at_the_bottom, at_the_top = partial_fractions(i, j, k, ibg.immersed_boundary)

    # `min` keeps a cell that is partial on one side only exact
    φ = ifelse(at_the_bottom & at_the_top, φᵇ + φᵗ - 1, min(φᵇ, φᵗ))

    full_Δr    = Δrᶜᶜᶜ(i, j, k, ibg.underlying_grid)
    partial_Δr = φ * full_Δr

    return ifelse(at_the_bottom | at_the_top, partial_Δr, full_Δr)
end

@inline function Δrᶜᶜᶠ(i, j, k, ibg::PCBTIBG)
    φᵇ, _, just_above_bottom, _ = partial_fractions(i, j, k-1, ibg.immersed_boundary)
    _, φᵗ, _, just_below_top    = partial_fractions(i, j, k,   ibg.immersed_boundary)

    rᶜ = rnode(i, j, k, ibg.underlying_grid, c, c, c)
    rᶠ = rnode(i, j, k, ibg.underlying_grid, c, c, f)

    full_Δr   = Δrᶜᶜᶠ(i, j, k, ibg.underlying_grid)
    bottom_Δr = ifelse(just_above_bottom, rᶜ - rᶠ + φᵇ * Δrᶜᶜᶜ(i, j, k-1, ibg.underlying_grid) / 2, full_Δr)

    # Subtracting the missing part keeps faces below full top cells exact
    top_deficit = ifelse(just_below_top, (1 - φᵗ) * Δrᶜᶜᶜ(i, j, k, ibg.underlying_grid) / 2, zero(rᶠ))

    return bottom_Δr - top_deficit
end

@inline Δrᶠᶜᶜ(i, j, k, ibg::PCBIBG) = min(Δrᶜᶜᶜ(i-1, j, k, ibg), Δrᶜᶜᶜ(i, j, k, ibg))
@inline Δrᶜᶠᶜ(i, j, k, ibg::PCBIBG) = min(Δrᶜᶜᶜ(i, j-1, k, ibg), Δrᶜᶜᶜ(i, j, k, ibg))
@inline Δrᶠᶠᶜ(i, j, k, ibg::PCBIBG) = min(Δrᶠᶜᶜ(i, j-1, k, ibg), Δrᶠᶜᶜ(i, j, k, ibg))

@inline Δrᶠᶜᶠ(i, j, k, ibg::PCBIBG) = min(Δrᶜᶜᶠ(i-1, j, k, ibg), Δrᶜᶜᶠ(i, j, k, ibg))
@inline Δrᶜᶠᶠ(i, j, k, ibg::PCBIBG) = min(Δrᶜᶜᶠ(i, j-1, k, ibg), Δrᶜᶜᶠ(i, j, k, ibg))
@inline Δrᶠᶠᶠ(i, j, k, ibg::PCBIBG) = min(Δrᶠᶜᶠ(i, j-1, k, ibg), Δrᶠᶜᶠ(i, j, k, ibg))

# Make sure Δz works for horizontally-Flat topologies.
# (There's no point in using z-Flat with PartialCellVerticalBoundaries).
const XFlatPCBIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:PartialCellVerticalBoundaries}
const YFlatPCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:PartialCellVerticalBoundaries}
const XYFlatPCBIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Flat, <:Any, <:Any, <:PartialCellVerticalBoundaries}

@inline Δrᶠᶜᶜ(i, j, k, ibg::XFlatPCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶜᶠ(i, j, k, ibg::XFlatPCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶜᶠᶜ(i, j, k, ibg::YFlatPCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)

@inline Δrᶜᶠᶠ(i, j, k, ibg::YFlatPCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XFlatPCBIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::YFlatPCBIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XYFlatPCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)

# Vertically-static, partial cell bottom, immersed boundary grid
VSPCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractStaticGrid, <:PartialCellVerticalBoundaries}
@inline Δzᶜᶜᶜ(i, j, k, ibg::VSPCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δzᶠᶜᶜ(i, j, k, ibg::VSPCBIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶜ(i, j, k, ibg::VSPCBIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶜᶠ(i, j, k, ibg::VSPCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶜ(i, j, k, ibg::VSPCBIBG) = Δrᶠᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶠ(i, j, k, ibg::VSPCBIBG) = Δrᶜᶠᶠ(i, j, k, ibg)
@inline Δzᶠᶜᶠ(i, j, k, ibg::VSPCBIBG) = Δrᶠᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶠ(i, j, k, ibg::VSPCBIBG) = Δrᶠᶠᶠ(i, j, k, ibg)

function Grids.constructor_arguments(grid::PCBIBG)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid.underlying_grid)
    partial_cell_bottom_args = Dict(:bottom_height => grid.immersed_boundary.bottom_height,
                                    :minimum_fractional_cell_height => grid.immersed_boundary.minimum_fractional_cell_height)
    return underlying_grid_args, underlying_grid_kwargs, partial_cell_bottom_args
end

function Base.:(==)(pcb1::PartialCellVerticalBoundaries, pcb2::PartialCellVerticalBoundaries)
    return bottom_heights_equal(pcb1.bottom_height, pcb2.bottom_height) &&
           bottom_heights_equal(pcb1.top_height, pcb2.top_height) &&
           bottom_heights_equal(pcb1.top_load, pcb2.top_load) &&
           pcb1.minimum_fractional_cell_height == pcb2.minimum_fractional_cell_height
end
