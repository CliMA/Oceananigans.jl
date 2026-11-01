using Oceananigans.Fields: Field, fill_halo_regions!, set!
using Oceananigans.Grids: Grids, AbstractStaticGrid, constructor_arguments

import Oceananigans.Grids: peripheral_node
using Oceananigans.Utils: prettysummary, KernelParameters

import Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶜᶜᶠ, Δrᶜᶠᶜ, Δrᶜᶠᶠ, Δrᶠᶜᶜ, Δrᶠᶜᶠ, Δrᶠᶠᶜ, Δrᶠᶠᶠ,
                               Δzᶜᶜᶜ, Δzᶜᶜᶠ, Δzᶜᶠᶜ, Δzᶜᶠᶠ, Δzᶠᶜᶜ, Δzᶠᶜᶠ, Δzᶠᶠᶜ, Δzᶠᶠᶠ,
                               Δr⁻¹ᶜᶜᶜ, Δr⁻¹ᶜᶜᶠ, Δr⁻¹ᶜᶠᶜ, Δr⁻¹ᶜᶠᶠ, Δr⁻¹ᶠᶜᶜ, Δr⁻¹ᶠᶜᶠ, Δr⁻¹ᶠᶠᶜ, Δr⁻¹ᶠᶠᶠ,
                               Δz⁻¹ᶜᶜᶜ, Δz⁻¹ᶜᶜᶠ, Δz⁻¹ᶜᶠᶜ, Δz⁻¹ᶜᶠᶠ, Δz⁻¹ᶠᶜᶜ, Δz⁻¹ᶠᶜᶠ, Δz⁻¹ᶠᶠᶜ, Δz⁻¹ᶠᶠᶠ

#####
##### PartialCellVerticalBoundaries
#####

struct PartialCellVerticalBoundaries{H, T, E, SB, ST, RB, RT, L} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    top_height :: T
    minimum_fractional_cell_height :: E
    fractional_bottom_index :: SB
    fractional_top_index :: ST
    # reciprocal spacings of the bottom (top) cell and of the face above (below) it, one per column, or `nothing` without a bottom (top)
    Δr⁻¹ᶜᶜᶜᵇ :: RB
    Δr⁻¹ᶜᶜᶠᵇ :: RB
    Δr⁻¹ᶜᶜᶜᵗ :: RT
    Δr⁻¹ᶜᶜᶠᵗ :: RT
    top_load :: L
end

function PartialCellVerticalBoundaries(bottom_height, top_height, minimum_fractional_cell_height, top_load=nothing)
    return PartialCellVerticalBoundaries(bottom_height, top_height, minimum_fractional_cell_height,
                                         nothing, nothing, nothing, nothing, nothing, nothing, top_load)
end

PartialCellVerticalBoundaries(bottom_height, top_height, minimum_fractional_cell_height, sᵇ, sᵗ, Δr⁻¹ᶜᶜᶜᵇ, Δr⁻¹ᶜᶜᶠᵇ, Δr⁻¹ᶜᶜᶜᵗ, Δr⁻¹ᶜᶜᶠᵗ) =
    PartialCellVerticalBoundaries(bottom_height, top_height, minimum_fractional_cell_height, sᵇ, sᵗ, Δr⁻¹ᶜᶜᶜᵇ, Δr⁻¹ᶜᶜᶠᵇ, Δr⁻¹ᶜᶜᶜᵗ, Δr⁻¹ᶜᶜᶠᵗ, nothing)

const PartialCellBottom{H, E} = PartialCellVerticalBoundaries{H, Nothing, E, <:Any, Nothing}
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

Base.summary(ib::PartialCellVerticalBoundaries{<:Union{Function, Nothing}}) = @sprintf("%s(%s%s, ϵ=%.1f)",
                                                                                       immersed_boundary_name(ib),
                                                                                       prettysummary(ib.bottom_height, false),
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
    PartialCellBottom(bottom_height; minimum_fractional_cell_height=0.2)

Return a [`PartialCellVerticalBoundaries`](@ref) without an immersed top, bounded below by `bottom_height`.
"""
function PartialCellBottom(bottom_height; minimum_fractional_cell_height=0.2)
    return PartialCellVerticalBoundaries(bottom_height, nothing, minimum_fractional_cell_height)
end

function materialize_immersed_boundary(grid, ib::PartialCellVerticalBoundaries)
    bottom = materialize_height(grid, ib.bottom_height)
    top    = materialize_height(grid, ib.top_height)

    minimum_fractional_cell_height = convert(eltype(grid), ib.minimum_fractional_cell_height)
    compute_ib = PartialCellVerticalBoundaries(bottom, top, minimum_fractional_cell_height)

    @apply_regionally compute_numerical_heights!(bottom, top, grid, compute_ib)

    fractional_bottom_index = Field{Center, Center, Nothing}(grid)
    fractional_top_index    = index_field(grid, top, eltype(grid))

    @apply_regionally compute_cell_indices!(grid, compute_ib, fractional_bottom_index, fractional_top_index, bottom, top)

    materialized_ib = PartialCellVerticalBoundaries(stored_data(bottom), stored_data(top), minimum_fractional_cell_height,
                                                    stored_data(fractional_bottom_index), stored_data(fractional_top_index),
                                                    nothing, nothing, nothing, nothing)

    Δr⁻¹ᶜᶜᶜᵇ = reciprocal_spacing_field(grid, bottom)
    Δr⁻¹ᶜᶜᶠᵇ = reciprocal_spacing_field(grid, bottom)
    Δr⁻¹ᶜᶜᶜᵗ = reciprocal_spacing_field(grid, top)
    Δr⁻¹ᶜᶜᶠᵗ = reciprocal_spacing_field(grid, top)

    TX, TY, TZ = topology(grid)
    ibg = ImmersedBoundaryGrid{TX, TY, TZ}(grid, materialized_ib, nothing, nothing)

    @apply_regionally compute_partial_cell_reciprocal_spacings!(Δr⁻¹ᶜᶜᶜᵇ, Δr⁻¹ᶜᶜᶠᵇ, ibg, materialized_ib.fractional_bottom_index, 1)
    @apply_regionally compute_partial_cell_reciprocal_spacings!(Δr⁻¹ᶜᶜᶜᵗ, Δr⁻¹ᶜᶜᶠᵗ, ibg, materialized_ib.fractional_top_index, 0)

    unloaded_ib = PartialCellVerticalBoundaries(stored_data(bottom), stored_data(top), minimum_fractional_cell_height,
                                                stored_data(fractional_bottom_index), stored_data(fractional_top_index),
                                                stored_data(Δr⁻¹ᶜᶜᶜᵇ), stored_data(Δr⁻¹ᶜᶜᶠᵇ), stored_data(Δr⁻¹ᶜᶜᶜᵗ), stored_data(Δr⁻¹ᶜᶜᶠᵗ))

    top_load = materialize_top_load(grid, unloaded_ib, ib.top_load)

    return PartialCellVerticalBoundaries(stored_data(bottom), stored_data(top), minimum_fractional_cell_height,
                                         stored_data(fractional_bottom_index), stored_data(fractional_top_index),
                                         stored_data(Δr⁻¹ᶜᶜᶜᵇ), stored_data(Δr⁻¹ᶜᶜᶠᵇ), stored_data(Δr⁻¹ᶜᶜᶜᵗ), stored_data(Δr⁻¹ᶜᶜᶠᵗ),
                                         top_load)
end

reciprocal_spacing_field(grid, height) = Field{Center, Center, Nothing}(grid)
reciprocal_spacing_field(grid, ::Nothing) = nothing

# Reciprocal spacings of the cell ⌊s⌋ and of the face ⌊s⌋ + face_offset, which is the face of that cell toward the interior
function compute_partial_cell_reciprocal_spacings!(Δr⁻¹ᶜᶜᶜ, Δr⁻¹ᶜᶜᶠ, ibg, s, face_offset)
    grid = ibg.underlying_grid
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    parameters = KernelParameters(1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
    return launch!(architecture(grid), grid, parameters, _compute_partial_cell_reciprocal_spacings!, Δr⁻¹ᶜᶜᶜ, Δr⁻¹ᶜᶜᶠ, ibg, s, face_offset)
end

compute_partial_cell_reciprocal_spacings!(::Nothing, ::Nothing, ibg, s, face_offset) = nothing

@kernel function _compute_partial_cell_reciprocal_spacings!(Δr⁻¹ᶜᶜᶜ, Δr⁻¹ᶜᶜᶠ, ibg, s, face_offset)
    i, j = @index(Global, NTuple)
    k = @inbounds unsafe_trunc(Int, s[i, j, 1])

    @inbounds Δr⁻¹ᶜᶜᶜ[i, j, 1] = 1 / Δrᶜᶜᶜ(i, j, k, ibg)
    @inbounds Δr⁻¹ᶜᶜᶠ[i, j, 1] = 1 / Δrᶜᶜᶠ(i, j, k + face_offset, ibg)
end

@kernel function _compute_numerical_heights!(bottom, top, grid, ib::PartialCellVerticalBoundaries)
    i, j = @index(Global, NTuple)

    domain_bottom = rnode(i, j, 1, grid, c, c, f)
    domain_top    = rnode(i, j, size(grid, 3)+1, grid, c, c, f)
    ẑᵇ = clamp(height_value(i, j, bottom, domain_bottom), domain_bottom, domain_top)
    ẑᵗ = clamp(height_value(i, j, top, domain_top), domain_bottom, domain_top)

    ϵ  = ib.minimum_fractional_cell_height
    Δzᵇ = zero(grid)

    for k in 1:size(grid, 3)
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
@inline thin_column(ẑᵇ, ẑᵗ, ϵΔzᵇ, ::Nothing) = false

# The fractional indices are sᵇ = kᵇ + 1 - φᵇ and sᵗ = kᵗ + 1 - φᵗ, where kᵇ (kᵗ) is the first (last) active cell of the column
# and φ ∈ [ϵ, 1] the open fraction of its height; sᵇ = Nz + 1 and sᵗ = Nz in closed columns, sᵗ = 0 without a stored bottom
@kernel function _compute_cell_indices!(fractional_bottom_index, fractional_top_index, grid, ib::PartialCellVerticalBoundaries, bottom, top)
    i, j = @index(Global, NTuple)
    rᵇ = height_value(i, j, bottom, rnode(i, j, 1, grid, c, c, f))
    rᵗ = height_value(i, j, top,    rnode(i, j, size(grid, 3)+1, grid, c, c, f))
    ϵ  = ib.minimum_fractional_cell_height

    kᵇ = size(grid, 3) + 1
    φᵇ = one(rᵇ)

    for k in size(grid, 3):-1:1
        r⁺ = rnode(i, j, k+1, grid, c, c, f)
        Δr = Δrᶜᶜᶜ(i, j, k, grid)
        immersed_cell = r⁺ - Δr * ϵ < rᵇ

        kᵇ = ifelse(immersed_cell, kᵇ, k)
        # `min`: when Δr and r⁺ - r⁻ differ by an ulp, a bottom on a face gives φ > 1 and ⌊sᵇ⌋ = kᵇ - 1
        φᵇ = ifelse(immersed_cell, φᵇ, min((r⁺ - rᵇ) / Δr, one(rᵇ)))
    end

    kᵗ = 0
    φᵗ = one(rᵗ)

    for k in 1:size(grid, 3)
        r⁻ = rnode(i, j, k, grid, c, c, f)
        Δr = Δrᶜᶜᶜ(i, j, k, grid)
        # `r⁻ ≥ rᵗ` keeps φ > 0 for ϵ = 0, so that ⌊sᵗ⌋ = kᵗ
        immersed_cell = (r⁻ + Δr * ϵ > rᵗ) | (r⁻ ≥ rᵗ)

        kᵗ = ifelse(immersed_cell, kᵗ, k)
        φᵗ = ifelse(immersed_cell, φᵗ, min((rᵗ - r⁻) / Δr, one(rᵗ)))
    end

    # Columns in the halos of `Bounded` directions keep full spacings
    halo_column = inactive_cell(i, j, 1, grid)
    φᵇ = ifelse(halo_column, one(φᵇ), φᵇ)
    φᵗ = ifelse(halo_column, one(φᵗ), φᵗ)

    store_column_value!(fractional_bottom_index, i, j, kᵇ + 1 - φᵇ)
    store_column_value!(fractional_top_index,    i, j, kᵗ + 1 - φᵗ)
end

# Kernels read the bottom and top through the fractional indices only, see `bottom_height(i, j, ibg)`
Adapt.adapt_structure(to, ib::PartialCellVerticalBoundaries) =
    PartialCellVerticalBoundaries(nothing,
                                  nothing,
                                  ib.minimum_fractional_cell_height,
                                  adapt(to, ib.fractional_bottom_index),
                                  adapt(to, ib.fractional_top_index),
                                  adapt(to, ib.Δr⁻¹ᶜᶜᶜᵇ),
                                  adapt(to, ib.Δr⁻¹ᶜᶜᶠᵇ),
                                  adapt(to, ib.Δr⁻¹ᶜᶜᶜᵗ),
                                  adapt(to, ib.Δr⁻¹ᶜᶜᶠᵗ),
                                  adapt(to, ib.top_load))

Architectures.on_architecture(to, ib::PartialCellVerticalBoundaries) =
    PartialCellVerticalBoundaries(on_architecture(to, ib.bottom_height),
                                  on_architecture(to, ib.top_height),
                                  on_architecture(to, ib.minimum_fractional_cell_height),
                                  on_architecture(to, ib.fractional_bottom_index),
                                  on_architecture(to, ib.fractional_top_index),
                                  on_architecture(to, ib.Δr⁻¹ᶜᶜᶜᵇ),
                                  on_architecture(to, ib.Δr⁻¹ᶜᶜᶠᵇ),
                                  on_architecture(to, ib.Δr⁻¹ᶜᶜᶜᵗ),
                                  on_architecture(to, ib.Δr⁻¹ᶜᶜᶠᵗ),
                                  on_architecture(to, ib.top_load))

# kᵇ = ⌊sᵇ⌋: `unsafe_trunc` is a single conversion, while mixed `Int`-`Float` comparisons of `sᵇ` are exact but several times longer
@inline bottom_cell(i, j, grid, ib::PartialCellVerticalBoundaries) = @inbounds unsafe_trunc(Int, ib.fractional_bottom_index[i, j, 1])
@inline    top_cell(i, j, grid, ib::PartialCellVerticalBoundaries) = unsafe_trunc(Int, height_value(i, j, ib.fractional_top_index, size(grid, 3)))

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

@inline face_column_depth(i, j, i⁻, j⁻, ibg, ::PartialCellBottom) = min(static_column_depthᶜᶜᵃ(i⁻, j⁻, ibg), static_column_depthᶜᶜᵃ(i, j, ibg))

@inline _immersed_cell(i, j, k, underlying_grid, ib::PartialCellWithTop) = (k < bottom_cell(i, j, underlying_grid, ib)) | (k > top_cell(i, j, underlying_grid, ib))

# The open fractions φᵇ and φᵗ of cell k hold only where `at_the_bottom` and `at_the_top`
# Only active cells are partial, so that immersed cells next to closed columns keep full spacings
@inline function partial_fractions(i, j, k, ib)
    sᵇ = @inbounds ib.fractional_bottom_index[i, j, 1]
    sᵗ = @inbounds ib.fractional_top_index[i, j, 1]
    kᵇ = unsafe_trunc(Int, sᵇ)
    kᵗ = unsafe_trunc(Int, sᵗ)
    active = (kᵇ ≤ k) & (k ≤ kᵗ)
    at_the_bottom = active & (k == kᵇ)
    at_the_top    = active & (k == kᵗ)
    return k + 1 - sᵇ, k + 1 - sᵗ, at_the_bottom, at_the_top
end

# Without a top only the bottom cell is partial
@inline function partial_fractions(i, j, k, ib::PartialCellBottom)
    sᵇ = @inbounds ib.fractional_bottom_index[i, j, 1]
    at_the_bottom = k == unsafe_trunc(Int, sᵇ)
    return k + 1 - sᵇ, one(sᵇ), at_the_bottom, false
end

@inline function Δrᶜᶜᶜ(i, j, k, ibg::PCBIBG)
    φᵇ, φᵗ, at_the_bottom, at_the_top = partial_fractions(i, j, k, ibg.immersed_boundary)

    # φᵇ + φᵗ - 1 only where both are partial, so that cells partial on one side only are exact
    φ = ifelse(at_the_top, ifelse(at_the_bottom, φᵇ + φᵗ - 1, φᵗ), φᵇ)

    full_Δr = Δrᶜᶜᶜ(i, j, k, ibg.underlying_grid)

    return ifelse(at_the_bottom | at_the_top, φ * full_Δr, full_Δr)
end

@inline function Δrᶜᶜᶠ(i, j, k, ibg::PCBIBG)
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

# The wet interval of cell k, in units of its spacing and measured from the face below it
@inline function wet_interval(i, j, k, ib)
    φᵇ, φᵗ, at_the_bottom, at_the_top = partial_fractions(i, j, k, ib)
    lower = ifelse(at_the_bottom, 1 - φᵇ, zero(φᵇ))
    upper = ifelse(at_the_top, φᵗ, one(φᵗ))
    return lower, upper
end

# Where a top cell faces a bottom cell their wet intervals are staggered, and overlap by less than either is wet
@inline function staggered_overlap(i⁻, j⁻, i, j, k, ib)
    l⁻, u⁻ = wet_interval(i⁻, j⁻, k, ib)
    l,  u  = wet_interval(i,  j,  k, ib)
    staggered = ((l⁻ < l) & (u⁻ < u)) | ((l < l⁻) & (u < u⁻))
    overlap = min(u⁻, u) - max(l⁻, l)
    open = (overlap > 0) & (overlap ≥ ib.minimum_fractional_cell_height)
    return staggered, open, overlap
end

@inline function closed_face(i⁻, j⁻, i, j, k, ib::PartialCellWithTop)
    staggered, open, _ = staggered_overlap(i⁻, j⁻, i, j, k, ib)
    return staggered & !open
end

# Closed faces keep the smaller spacing, so that their (masked) velocities never divide by zero
@inline function face_spacing(i⁻, j⁻, i, j, k, ibg, ib::PartialCellWithTop)
    staggered, open, overlap = staggered_overlap(i⁻, j⁻, i, j, k, ib)
    Δr = min(Δrᶜᶜᶜ(i⁻, j⁻, k, ibg), Δrᶜᶜᶜ(i, j, k, ibg))
    return ifelse(staggered & open, overlap * Δrᶜᶜᶜ(i, j, k, ibg.underlying_grid), Δr)
end

@inline face_reciprocal_spacing(i⁻, j⁻, i, j, k, ibg, ib) = max(Δr⁻¹ᶜᶜᶜ(i⁻, j⁻, k, ibg), Δr⁻¹ᶜᶜᶜ(i, j, k, ibg))

@inline function face_reciprocal_spacing(i⁻, j⁻, i, j, k, ibg, ib::PartialCellWithTop)
    staggered, open, overlap = staggered_overlap(i⁻, j⁻, i, j, k, ib)
    Δr⁻¹ = max(Δr⁻¹ᶜᶜᶜ(i⁻, j⁻, k, ibg), Δr⁻¹ᶜᶜᶜ(i, j, k, ibg))
    return ifelse(staggered & open, Δr⁻¹ᶜᶜᶜ(i, j, k, ibg.underlying_grid) / overlap, Δr⁻¹)
end

@inline Δrᶠᶜᶜ(i, j, k, ibg::PCBIBG) = face_spacing(i-1, j, i, j, k, ibg, ibg.immersed_boundary)
@inline Δrᶜᶠᶜ(i, j, k, ibg::PCBIBG) = face_spacing(i, j-1, i, j, k, ibg, ibg.immersed_boundary)
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

# Faces between active cells whose wet intervals do not overlap are closed like immersed faces
@inline closed_xface(i, j, k, ibg) = closed_face(i-1, j, i, j, k, ibg.immersed_boundary)
@inline closed_yface(i, j, k, ibg) = closed_face(i, j-1, i, j, k, ibg.immersed_boundary)
@inline closed_xface(i, j, k, ibg::XFlatPCBIBG) = false
@inline closed_yface(i, j, k, ibg::YFlatPCBIBG) = false

@inline peripheral_node(i, j, k, ibg::PCBTIBG, ::Face, ::Center, ::Center) =
    inactive_cell(i, j, k, ibg) | inactive_cell(i-1, j, k, ibg) | closed_xface(i, j, k, ibg)

@inline peripheral_node(i, j, k, ibg::PCBTIBG, ::Center, ::Face, ::Center) =
    inactive_cell(i, j, k, ibg) | inactive_cell(i, j-1, k, ibg) | closed_yface(i, j, k, ibg)

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

# The stored reciprocal at k = ⌊s⌋, or Δr⁻¹ elsewhere and without a stored reciprocal
@inline function partial_cell_reciprocal_spacing(i, j, k, s, partial_Δr⁻¹, Δr⁻¹)
    at_the_partial_cell = k == unsafe_trunc(Int, @inbounds(s[i, j, 1]))
    return ifelse(at_the_partial_cell, @inbounds(partial_Δr⁻¹[i, j, 1]), Δr⁻¹)
end

@inline partial_cell_reciprocal_spacing(i, j, k, s, ::Nothing, Δr⁻¹) = Δr⁻¹

# Where a cell is both the bottom and the top cell, both stored reciprocals hold the same value
@inline function Δr⁻¹ᶜᶜᶜ(i, j, k, ibg::PCBIBG)
    ib = ibg.immersed_boundary
    full_Δr⁻¹   = Δr⁻¹ᶜᶜᶜ(i, j, k, ibg.underlying_grid)
    bottom_Δr⁻¹ = partial_cell_reciprocal_spacing(i, j, k, ib.fractional_bottom_index, ib.Δr⁻¹ᶜᶜᶜᵇ, full_Δr⁻¹)
    return partial_cell_reciprocal_spacing(i, j, k, ib.fractional_top_index, ib.Δr⁻¹ᶜᶜᶜᵗ, bottom_Δr⁻¹)
end

# The bottom reciprocal is stored at the face above the bottom cell, kᵇ + 1
@inline function Δr⁻¹ᶜᶜᶠ(i, j, k, ibg::PCBIBG)
    ib = ibg.immersed_boundary
    full_Δr⁻¹   = Δr⁻¹ᶜᶜᶠ(i, j, k, ibg.underlying_grid)
    bottom_Δr⁻¹ = partial_cell_reciprocal_spacing(i, j, k-1, ib.fractional_bottom_index, ib.Δr⁻¹ᶜᶜᶠᵇ, full_Δr⁻¹)
    return partial_cell_reciprocal_spacing(i, j, k, ib.fractional_top_index, ib.Δr⁻¹ᶜᶜᶠᵗ, bottom_Δr⁻¹)
end

# The reciprocal of the smallest spacing is the largest reciprocal
@inline Δr⁻¹ᶠᶜᶜ(i, j, k, ibg::PCBIBG) = face_reciprocal_spacing(i-1, j, i, j, k, ibg, ibg.immersed_boundary)
@inline Δr⁻¹ᶜᶠᶜ(i, j, k, ibg::PCBIBG) = face_reciprocal_spacing(i, j-1, i, j, k, ibg, ibg.immersed_boundary)
@inline Δr⁻¹ᶠᶠᶜ(i, j, k, ibg::PCBIBG) = max(Δr⁻¹ᶠᶜᶜ(i, j-1, k, ibg), Δr⁻¹ᶠᶜᶜ(i, j, k, ibg))

@inline Δr⁻¹ᶠᶜᶠ(i, j, k, ibg::PCBIBG) = max(Δr⁻¹ᶜᶜᶠ(i-1, j, k, ibg), Δr⁻¹ᶜᶜᶠ(i, j, k, ibg))
@inline Δr⁻¹ᶜᶠᶠ(i, j, k, ibg::PCBIBG) = max(Δr⁻¹ᶜᶜᶠ(i, j-1, k, ibg), Δr⁻¹ᶜᶜᶠ(i, j, k, ibg))
@inline Δr⁻¹ᶠᶠᶠ(i, j, k, ibg::PCBIBG) = max(Δr⁻¹ᶠᶜᶠ(i, j-1, k, ibg), Δr⁻¹ᶠᶜᶠ(i, j, k, ibg))

@inline Δr⁻¹ᶠᶜᶜ(i, j, k, ibg::XFlatPCBIBG) = Δr⁻¹ᶜᶜᶜ(i, j, k, ibg)
@inline Δr⁻¹ᶠᶜᶠ(i, j, k, ibg::XFlatPCBIBG) = Δr⁻¹ᶜᶜᶠ(i, j, k, ibg)
@inline Δr⁻¹ᶜᶠᶜ(i, j, k, ibg::YFlatPCBIBG) = Δr⁻¹ᶜᶜᶜ(i, j, k, ibg)

@inline Δr⁻¹ᶜᶠᶠ(i, j, k, ibg::YFlatPCBIBG) = Δr⁻¹ᶜᶜᶠ(i, j, k, ibg)
@inline Δr⁻¹ᶠᶠᶜ(i, j, k, ibg::XFlatPCBIBG) = Δr⁻¹ᶜᶠᶜ(i, j, k, ibg)
@inline Δr⁻¹ᶠᶠᶜ(i, j, k, ibg::YFlatPCBIBG) = Δr⁻¹ᶠᶜᶜ(i, j, k, ibg)
@inline Δr⁻¹ᶠᶠᶜ(i, j, k, ibg::XYFlatPCBIBG) = Δr⁻¹ᶜᶜᶜ(i, j, k, ibg)

@inline Δz⁻¹ᶜᶜᶜ(i, j, k, ibg::VSPCBIBG) = Δr⁻¹ᶜᶜᶜ(i, j, k, ibg)
@inline Δz⁻¹ᶠᶜᶜ(i, j, k, ibg::VSPCBIBG) = Δr⁻¹ᶠᶜᶜ(i, j, k, ibg)
@inline Δz⁻¹ᶜᶠᶜ(i, j, k, ibg::VSPCBIBG) = Δr⁻¹ᶜᶠᶜ(i, j, k, ibg)
@inline Δz⁻¹ᶜᶜᶠ(i, j, k, ibg::VSPCBIBG) = Δr⁻¹ᶜᶜᶠ(i, j, k, ibg)
@inline Δz⁻¹ᶠᶠᶜ(i, j, k, ibg::VSPCBIBG) = Δr⁻¹ᶠᶠᶜ(i, j, k, ibg)
@inline Δz⁻¹ᶜᶠᶠ(i, j, k, ibg::VSPCBIBG) = Δr⁻¹ᶜᶠᶠ(i, j, k, ibg)
@inline Δz⁻¹ᶠᶜᶠ(i, j, k, ibg::VSPCBIBG) = Δr⁻¹ᶠᶜᶠ(i, j, k, ibg)
@inline Δz⁻¹ᶠᶠᶠ(i, j, k, ibg::VSPCBIBG) = Δr⁻¹ᶠᶠᶠ(i, j, k, ibg)

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
