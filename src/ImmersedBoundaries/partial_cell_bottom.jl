using Oceananigans.Fields: Field, fill_halo_regions!, set!
using Oceananigans.Grids: Grids, bottommost_active_node, topmost_active_node, AbstractStaticGrid, constructor_arguments
using Oceananigans.Utils: prettysummary

import Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶜᶜᶠ, Δrᶜᶠᶜ, Δrᶜᶠᶠ, Δrᶠᶜᶜ, Δrᶠᶜᶠ, Δrᶠᶠᶜ, Δrᶠᶠᶠ,
                               Δzᶜᶜᶜ, Δzᶜᶜᶠ, Δzᶜᶠᶜ, Δzᶜᶠᶠ, Δzᶠᶜᶜ, Δzᶠᶜᶠ, Δzᶠᶠᶜ, Δzᶠᶠᶠ

#####
##### PartialCellBottom
#####

struct PartialCellBottom{H, T, E, L} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    top_height :: T
    minimum_fractional_cell_height :: E
    top_load :: L
end

PartialCellBottom(bottom_height, minimum_fractional_cell_height) = PartialCellBottom(bottom_height, nothing, minimum_fractional_cell_height)
PartialCellBottom(bottom_height, top_height, minimum_fractional_cell_height) = PartialCellBottom(bottom_height, top_height, minimum_fractional_cell_height, nothing)

const PCBIBG{FT, TX, TY, TZ} = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:Any, <:PartialCellBottom} where {FT, TX, TY, TZ}

# PartialCellBottom with an immersed top
const PCBTIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:PartialCellBottom{<:Any, <:AbstractArray}}

function Base.summary(ib::PartialCellBottom)
    bottom_interior = bottom_height_interior(ib.bottom_height)
    zmax = maximum(bottom_interior)
    zmin = minimum(bottom_interior)
    zmean = sum(bottom_interior) / length(bottom_interior)

    summary1 = "PartialCellBottom("

    summary2 = string("mean(zb)=", prettysummary(zmean),
                      ", min(zb)=", prettysummary(zmin),
                      ", max(zb)=", prettysummary(zmax),
                      top_height_summary(ib.top_height),
                      ", ϵ=", prettysummary(ib.minimum_fractional_cell_height))

    summary3 = ")"

    return summary1 * summary2 * summary3
end

Base.summary(ib::PartialCellBottom{<:Function}) = @sprintf("PartialCellBottom(%s%s, ϵ=%.1f)",
                                                           prettysummary(ib.bottom_height, false),
                                                           top_height_summary(ib.top_height),
                                                           ib.minimum_fractional_cell_height)

Base.summary(ib::PartialCellBottom{Nothing}) = @sprintf("PartialCellBottom(nothing%s, ϵ=%.1f)",
                                                        top_height_summary(ib.top_height),
                                                        ib.minimum_fractional_cell_height)

function Base.show(io::IO, ib::PartialCellBottom{<:Any, Nothing})
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "└── minimum_fractional_cell_height: ", prettysummary(ib.minimum_fractional_cell_height))
end

function Base.show(io::IO, ib::PartialCellBottom)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "├── top_height: ", prettysummary(ib.top_height), '\n')
    print(io, "├── top_load: ", prettysummary(ib.top_load), '\n')
    print(io, "└── minimum_fractional_cell_height: ", prettysummary(ib.minimum_fractional_cell_height))
end

"""
    PartialCellBottom(bottom_height=nothing; top_height=nothing, minimum_fractional_cell_height=0.2, top_load=nothing)

Return `PartialCellBottom` representing an immersed boundary with "partial"
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

julia> ImmersedBoundaryGrid(grid, PartialCellBottom(-95; top_height=(x, y) -> -20 - 0.2y))
2×8×10 ImmersedBoundaryGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo:
├── immersed_boundary: PartialCellBottom(mean(zb)=-95.0, min(zb)=-95.0, max(zb)=-95.0, mean(zt)=-29.8125, min(zt)=-38.0, max(zt)=-21.25, ϵ=0.2)
├── underlying_grid: 2×8×10 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── Periodic x ∈ [0.0, 100.0)  regularly spaced with Δx=50.0
├── Periodic y ∈ [0.0, 100.0)  regularly spaced with Δy=12.5
└── Bounded  z ∈ [-100.0, 0.0] regularly spaced with Δz=10.0
```
"""
function PartialCellBottom(bottom_height=nothing; top_height=nothing, minimum_fractional_cell_height=0.2, top_load=nothing)
    validate_top_load(top_height, top_load)
    return PartialCellBottom(bottom_height, top_height, minimum_fractional_cell_height, top_load)
end

function materialize_immersed_boundary(grid, ib::PartialCellBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(bottom_field, ib.bottom_height)
    top_field = materialize_top_height(grid, ib.top_height)

    minimum_fractional_cell_height = convert(eltype(grid), ib.minimum_fractional_cell_height)
    compute_ib = PartialCellBottom(bottom_field, top_field, minimum_fractional_cell_height)

    @apply_regionally compute_numerical_bottom_height!(bottom_field, grid, compute_ib)
    @apply_regionally compute_numerical_top_height!(bottom_field, top_field, grid, compute_ib)
    fill_halo_regions!(bottom_field)
    fill_top_height_halo_regions!(top_field)

    unloaded_ib = PartialCellBottom(bottom_field.data, top_height_data(top_field), minimum_fractional_cell_height)
    top_load = materialize_top_load(grid, unloaded_ib, ib.top_load)

    return PartialCellBottom(bottom_field.data, top_height_data(top_field), minimum_fractional_cell_height, top_load)
end

@kernel function _compute_numerical_bottom_height!(bottom_field, grid, ib::PartialCellBottom)
    i, j = @index(Global, NTuple)

    # Save analytical bottom height
    zb = @inbounds bottom_field[i, j, 1]

    # Cap bottom height at Lz and at rnode(i, j, grid.Nz+1, grid, c, c, f)

    domain_bottom = rnode(i, j, 1, grid, c, c, f)
    domain_top    = rnode(i, j, grid.Nz+1, grid, c, c, f)
    @inbounds bottom_field[i, j, 1] = clamp(zb, domain_bottom, domain_top)
    adjusted_zb = bottom_field[i, j, 1]

    ϵ  = ib.minimum_fractional_cell_height

    for k in 1:grid.Nz
        z⁻ = rnode(i, j, k,   grid, c, c, f)
        z⁺ = rnode(i, j, k+1, grid, c, c, f)
        Δz = Δrᶜᶜᶜ(i, j, k, grid)
        bottom_cell = z⁻ ≤ adjusted_zb < z⁺
        capped_zb   = min(z⁺ - ϵ * Δz, adjusted_zb)

        # If the size of the bottom cell is less than ϵ Δz,
        # we enforce a minimum size of ϵ Δz.
        adjusted_zb = ifelse(bottom_cell, capped_zb, adjusted_zb)
    end
    @inbounds bottom_field[i, j, 1] = adjusted_zb
end

@kernel function _compute_numerical_top_height!(bottom_field, top_field, grid, ib::PartialCellBottom)
    i, j = @index(Global, NTuple)

    domain_bottom = rnode(i, j, 1, grid, c, c, f)
    domain_top    = rnode(i, j, grid.Nz+1, grid, c, c, f)
    ẑᵗ = clamp(@inbounds(top_field[i, j, 1]), domain_bottom, domain_top)
    ẑᵇ = @inbounds bottom_field[i, j, 1]

    ϵ  = ib.minimum_fractional_cell_height
    Δzᵇ = zero(grid)

    for k in 1:grid.Nz
        z⁻ = rnode(i, j, k,   grid, c, c, f)
        z⁺ = rnode(i, j, k+1, grid, c, c, f)
        Δz = Δrᶜᶜᶜ(i, j, k, grid)
        top_cell = z⁻ < ẑᵗ ≤ z⁺

        # If the size of the top cell is less than ϵ Δz, we enforce a minimum size of ϵ Δz
        ẑᵗ = ifelse(top_cell, max(z⁻ + ϵ * Δz, ẑᵗ), ẑᵗ)
        Δzᵇ = ifelse(z⁻ ≤ ẑᵇ < z⁺, Δz, Δzᵇ)
    end

    # Close columns in which the top and bottom leave less than ϵ Δz of fluid
    closed = (ẑᵗ ≤ ẑᵇ) | (ẑᵗ - ẑᵇ < ϵ * Δzᵇ)
    @inbounds bottom_field[i, j, 1] = ifelse(closed, domain_top, ẑᵇ)
    @inbounds top_field[i, j, 1] = ifelse(closed, domain_top, ẑᵗ)
end

Adapt.adapt_structure(to, ib::PartialCellBottom) = PartialCellBottom(adapt(to, ib.bottom_height),
                                                                     adapt(to, ib.top_height),
                                                                     ib.minimum_fractional_cell_height,
                                                                     adapt(to, ib.top_load))

Architectures.on_architecture(to, ib::PartialCellBottom) = PartialCellBottom(on_architecture(to, ib.bottom_height),
                                                                             on_architecture(to, ib.top_height),
                                                                             on_architecture(to, ib.minimum_fractional_cell_height),
                                                                             on_architecture(to, ib.top_load))

"""
    immersed     underlying

      --x--        --x--


        ∘   ↑        ∘   k+1
            |
            |
  k+1 --x-- |  k+1 --x--    ↑      <- node z
        ∘   ↓               |
   zb ⋅⋅x⋅⋅                 |
                            |
                     ∘   k  | Δz
                            |
                            |
                 k --x--    ↓

Criterion is zb ≥ z - ϵ Δz

"""
@inline function _immersed_cell(i, j, k, underlying_grid, ib::PartialCellBottom)
    r⁺ = rnode(i, j, k + 1, underlying_grid, c, c, f)
    ϵ  = ib.minimum_fractional_cell_height
    Δr = Δrᶜᶜᶜ(i, j, k, underlying_grid)
    r★ = r⁺ - Δr * ϵ
    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    return (r★ < rᵇ) | immersed_by_top(i, j, k, underlying_grid, ib)
end

@inline immersed_by_top(i, j, k, underlying_grid, ::PartialCellBottom{<:Any, Nothing}) = false

@inline function immersed_by_top(i, j, k, underlying_grid, ib::PartialCellBottom)
    r⁻ = rnode(i, j, k, underlying_grid, c, c, f)
    ϵ  = ib.minimum_fractional_cell_height
    Δr = Δrᶜᶜᶜ(i, j, k, underlying_grid)
    r☆ = r⁻ + Δr * ϵ
    rᵗ = @inbounds ib.top_height[i, j, 1]
    return r☆ > rᵗ
end

@inline function Δrᶜᶜᶜ(i, j, k, ibg::PCBIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    # Get node at face above and defining nodes on c,c,f
    r⁺ = rnode(i, j, k + 1, underlying_grid, c, c, f)

    # Get bottom r-coordinate and fractional Δr parameter
    rᵇ = @inbounds ib.bottom_height[i, j, 1]

    # Are we in a bottom cell?
    at_the_bottom = bottommost_active_node(i, j, k, ibg, c, c, c)

    full_Δr    = Δrᶜᶜᶜ(i, j, k, ibg.underlying_grid)
    partial_Δr = r⁺ - rᵇ

    return ifelse(at_the_bottom, partial_Δr, full_Δr)
end

@inline function Δrᶜᶜᶜ(i, j, k, ibg::PCBTIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    r⁻ = rnode(i, j, k,     underlying_grid, c, c, f)
    r⁺ = rnode(i, j, k + 1, underlying_grid, c, c, f)
    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    rᵗ = @inbounds ib.top_height[i, j, 1]

    at_the_bottom = bottommost_active_node(i, j, k, ibg, c, c, c)
    at_the_top    = partial_top_cell(i, j, k, ibg)

    lower = ifelse(at_the_bottom, rᵇ, r⁻)
    upper = ifelse(at_the_top,    rᵗ, r⁺)

    full_Δr    = Δrᶜᶜᶜ(i, j, k, underlying_grid)
    partial_Δr = upper - lower

    return ifelse(at_the_bottom | at_the_top, partial_Δr, full_Δr)
end

# The topmost active cell of an open column keeps its full height
@inline function partial_top_cell(i, j, k, ibg)
    r⁺ = rnode(i, j, k + 1, ibg.underlying_grid, c, c, f)
    rᵗ = @inbounds ibg.immersed_boundary.top_height[i, j, 1]
    return topmost_active_node(i, j, k, ibg, c, c, c) & (rᵗ < r⁺)
end

@inline function Δrᶜᶜᶠ(i, j, k, ibg::PCBTIBG)
    just_above_bottom = bottommost_active_node(i, j, k-1, ibg, c, c, c)
    just_below_top    = partial_top_cell(i, j, k, ibg)
    rᶜ⁻ = rnode(i, j, k-1, ibg.underlying_grid, c, c, c)
    rᶜ  = rnode(i, j, k,   ibg.underlying_grid, c, c, c)
    rᶠ  = rnode(i, j, k,   ibg.underlying_grid, c, c, f)

    lower = ifelse(just_above_bottom, Δrᶜᶜᶜ(i, j, k-1, ibg) / 2, rᶠ - rᶜ⁻)
    upper = ifelse(just_below_top,    Δrᶜᶜᶜ(i, j, k, ibg) / 2, rᶜ - rᶠ)

    full_Δr    = Δrᶜᶜᶠ(i, j, k, ibg.underlying_grid)
    partial_Δr = lower + upper

    return ifelse(just_above_bottom | just_below_top, partial_Δr, full_Δr)
end

@inline function Δrᶜᶜᶠ(i, j, k, ibg::PCBIBG)
    # The face at k is just above the bottom when the cell below it (k-1) is the partial cell
    just_above_bottom = bottommost_active_node(i, j, k-1, ibg, c, c, c)
    rᶜ = rnode(i, j, k, ibg.underlying_grid, c, c, c)
    rᶠ = rnode(i, j, k, ibg.underlying_grid, c, c, f)

    full_Δr    = Δrᶜᶜᶠ(i, j, k, ibg.underlying_grid)
    partial_Δr = rᶜ - rᶠ + Δrᶜᶜᶜ(i, j, k-1, ibg) / 2

    return ifelse(just_above_bottom, partial_Δr, full_Δr)
end

@inline Δrᶠᶜᶜ(i, j, k, ibg::PCBIBG) = min(Δrᶜᶜᶜ(i-1, j, k, ibg), Δrᶜᶜᶜ(i, j, k, ibg))
@inline Δrᶜᶠᶜ(i, j, k, ibg::PCBIBG) = min(Δrᶜᶜᶜ(i, j-1, k, ibg), Δrᶜᶜᶜ(i, j, k, ibg))
@inline Δrᶠᶠᶜ(i, j, k, ibg::PCBIBG) = min(Δrᶠᶜᶜ(i, j-1, k, ibg), Δrᶠᶜᶜ(i, j, k, ibg))

@inline Δrᶠᶜᶠ(i, j, k, ibg::PCBIBG) = min(Δrᶜᶜᶠ(i-1, j, k, ibg), Δrᶜᶜᶠ(i, j, k, ibg))
@inline Δrᶜᶠᶠ(i, j, k, ibg::PCBIBG) = min(Δrᶜᶜᶠ(i, j-1, k, ibg), Δrᶜᶜᶠ(i, j, k, ibg))
@inline Δrᶠᶠᶠ(i, j, k, ibg::PCBIBG) = min(Δrᶠᶜᶠ(i, j-1, k, ibg), Δrᶠᶜᶠ(i, j, k, ibg))

# Make sure Δz works for horizontally-Flat topologies.
# (There's no point in using z-Flat with PartialCellBottom).
const XFlatPCBIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:PartialCellBottom}
const YFlatPCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:PartialCellBottom}
const XYFlatPCBIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Flat, <:Any, <:Any, <:PartialCellBottom}

@inline Δrᶠᶜᶜ(i, j, k, ibg::XFlatPCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶜᶠ(i, j, k, ibg::XFlatPCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶜᶠᶜ(i, j, k, ibg::YFlatPCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)

@inline Δrᶜᶠᶠ(i, j, k, ibg::YFlatPCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XFlatPCBIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::YFlatPCBIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XYFlatPCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)

# Vertically-static, partial cell bottom, immersed boundary grid
VSPCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractStaticGrid, <:PartialCellBottom}
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

function Base.:(==)(pcb1::PartialCellBottom, pcb2::PartialCellBottom)
    return bottom_heights_equal(pcb1.bottom_height, pcb2.bottom_height) &&
           bottom_heights_equal(pcb1.top_height, pcb2.top_height) &&
           bottom_heights_equal(pcb1.top_load, pcb2.top_load) &&
           pcb1.minimum_fractional_cell_height == pcb2.minimum_fractional_cell_height
end
