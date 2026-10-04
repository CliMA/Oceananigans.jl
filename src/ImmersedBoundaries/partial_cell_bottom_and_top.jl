using Oceananigans.Grids: Grids, bottommost_active_node, topmost_active_node, AbstractStaticGrid, constructor_arguments
using Oceananigans.Utils: prettysummary

struct PartialCellBottomAndTop{H, T, E, R, L} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    top_height :: T
    minimum_fractional_cell_height :: E
    minimum_cell_height :: R
    top_load :: L
end

const PCBTIBG{FT, TX, TY, TZ} = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:Any, <:PartialCellBottomAndTop} where {FT, TX, TY, TZ}

function Base.summary(ib::PartialCellBottomAndTop)
    bottom_interior = bottom_height_interior(ib.bottom_height)
    top_interior    = bottom_height_interior(ib.top_height)
    zbmean = sum(bottom_interior) / length(bottom_interior)
    ztmean = sum(top_interior) / length(top_interior)

    return string("PartialCellBottomAndTop(mean(zb)=", prettysummary(zbmean),
                  ", mean(zt)=", prettysummary(ztmean),
                  ", ϵ=", prettysummary(ib.minimum_fractional_cell_height),
                  ", minimum_cell_height=", prettysummary(ib.minimum_cell_height), ")")
end

Base.summary(ib::PartialCellBottomAndTop{<:Function}) = @sprintf("PartialCellBottomAndTop(ϵ=%.1f)", ib.minimum_fractional_cell_height)

function Base.show(io::IO, ib::PartialCellBottomAndTop)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "├── top_height: ", prettysummary(ib.top_height), '\n')
    print(io, "├── minimum_fractional_cell_height: ", prettysummary(ib.minimum_fractional_cell_height), '\n')
    print(io, "├── minimum_cell_height: ", prettysummary(ib.minimum_cell_height), '\n')
    print(io, "└── top_load: ", prettysummary(ib.top_load))
    return nothing
end

"""
$(TYPEDSIGNATURES)

Return `PartialCellBottomAndTop`, an immersed boundary in which the fluid lies between
`bottom_height` and `top_height` (e.g. beneath an ice shelf whose draft is `top_height`),
in which the bottommost and topmost active cells of every column have fractional heights rather
than being snapped to the underlying grid faces as in [`GridFittedBottomAndTop`](@ref).

`bottom_height` and `top_height` may each be an `Array` or a function of `(x, y)`.
Columns where `top_height` is at or above the top of the grid are open.

The height of a partial cell is floored at `max(minimum_fractional_cell_height * Δz, minimum_cell_height)`,
where `Δz` is the underlying cell height.
A cell thinner than half this floor is removed and a thicker one is raised to the floor.

`top_load` is the potential of the weight of the solid top, added to the hydrostatic pressure of
every column: either `nothing` (default, no load), a [`TopLoad`](@ref) computed from a reference
state, or an array, function of `(x, y)` or two-dimensional field such as the one returned by
[`top_load_potential`](@ref).

Example
=======

```jldoctest
julia> using Oceananigans

julia> bottom_height(x, y) = -100
bottom_height (generic function with 1 method)

julia> top_height(x, y) = -20 + 0.01y
top_height (generic function with 1 method)

julia> grid = RectilinearGrid(size=(2, 8, 10), x=(0, 100), y=(0, 100), z=(-100, 0), topology=(Periodic, Periodic, Bounded));

julia> ibg = ImmersedBoundaryGrid(grid, PartialCellBottomAndTop(bottom_height, top_height))
2×8×10 ImmersedBoundaryGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo:
├── immersed_boundary: PartialCellBottomAndTop(mean(zb)=-100.0, mean(zt)=-20.0, ϵ=0.2, minimum_cell_height=0.0)
├── underlying_grid: 2×8×10 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── Periodic x ∈ [0.0, 100.0)  regularly spaced with Δx=50.0
├── Periodic y ∈ [0.0, 100.0)  regularly spaced with Δy=12.5
└── Bounded  z ∈ [-100.0, 0.0] regularly spaced with Δz=10.0
```
"""
function PartialCellBottomAndTop(bottom_height, top_height; minimum_fractional_cell_height=0.2, minimum_cell_height=0, top_load=nothing)
    return PartialCellBottomAndTop(bottom_height, top_height, minimum_fractional_cell_height, minimum_cell_height, top_load)
end

function materialize_immersed_boundary(grid, ib::PartialCellBottomAndTop)
    bottom_field = Field{Center, Center, Nothing}(grid)
    top_field    = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(bottom_field, ib.bottom_height)
    set_bottom_height!(top_field, ib.top_height)

    minimum_fractional_cell_height = convert(eltype(grid), ib.minimum_fractional_cell_height)
    minimum_cell_height = convert(eltype(grid), ib.minimum_cell_height)
    new_ib = PartialCellBottomAndTop(bottom_field, top_field, minimum_fractional_cell_height, minimum_cell_height, nothing)

    @apply_regionally compute_numerical_bottom_and_top_heights!(bottom_field, top_field, grid, new_ib)
    fill_halo_regions!(bottom_field)
    fill_halo_regions!(top_field)

    unloaded_ib = PartialCellBottomAndTop(bottom_field.data, top_field.data, minimum_fractional_cell_height, minimum_cell_height, nothing)
    top_load = materialize_top_load(grid, unloaded_ib, ib.top_load)
    return PartialCellBottomAndTop(bottom_field.data, top_field.data, minimum_fractional_cell_height, minimum_cell_height, top_load)
end

compute_numerical_bottom_and_top_heights!(bottom_field, top_field, grid, ib::PartialCellBottomAndTop) =
    launch!(architecture(grid), grid, :xy, _compute_numerical_bottom_and_top_heights!, bottom_field, top_field, grid, ib)

@kernel function _compute_numerical_bottom_and_top_heights!(bottom_field, top_field, grid, ib::PartialCellBottomAndTop)
    i, j = @index(Global, NTuple)

    domain_bottom = rnode(i, j, 1, grid, c, c, f)
    domain_top    = rnode(i, j, grid.Nz+1, grid, c, c, f)

    zᵇ = clamp(@inbounds(bottom_field[i, j, 1]), domain_bottom, domain_top)
    zᵗ = clamp(@inbounds(top_field[i, j, 1]),    domain_bottom, domain_top)

    ϵ  = ib.minimum_fractional_cell_height
    Δrᵐⁱⁿ = ib.minimum_cell_height
    has_top = zᵗ < domain_top

    # Raise to ϵ★ = max(ϵ, Δrᵐⁱⁿ / Δz), remove below ϵ★ / 2
    ẑᵇ = zᵇ
    Δzᵇ = zero(grid)
    for k in 1:grid.Nz
        z⁻ = rnode(i, j, k,   grid, c, c, f)
        z⁺ = rnode(i, j, k+1, grid, c, c, f)
        Δz = Δrᶜᶜᶜ(i, j, k, grid)
        ϵ★ = max(ϵ, Δrᵐⁱⁿ / Δz)
        bottom_cell = z⁻ ≤ zᵇ < z⁺
        fᵇ = (z⁺ - zᵇ) / Δz
        z̃ᵇ = ifelse(fᵇ < ϵ★ / 2, z⁺, z⁺ - max(fᵇ, ϵ★) * Δz)
        ẑᵇ = ifelse(bottom_cell, z̃ᵇ, ẑᵇ)
        Δzᵇ = ifelse(bottom_cell, Δz, Δzᵇ)
    end

    ẑᵗ = zᵗ
    for k in grid.Nz : -1 : 1
        z⁻ = rnode(i, j, k,   grid, c, c, f)
        z⁺ = rnode(i, j, k+1, grid, c, c, f)
        Δz = Δrᶜᶜᶜ(i, j, k, grid)
        ϵ★ = max(ϵ, Δrᵐⁱⁿ / Δz)
        top_cell = has_top & (z⁻ < zᵗ) & (zᵗ ≤ z⁺)
        fᵗ = (zᵗ - z⁻) / Δz
        z̃ᵗ = ifelse(fᵗ < ϵ★ / 2, z⁻, z⁻ + max(fᵗ, ϵ★) * Δz)
        ẑᵗ = ifelse(top_cell, z̃ᵗ, ẑᵗ)
    end

    # Close columns thinner than one floored cell, tested against the unfloored heights
    ϵᵇ = ifelse(Δzᵇ > 0, max(ϵ, Δrᵐⁱⁿ / Δzᵇ), ϵ)
    too_thin = (zᵗ - zᵇ < ϵᵇ * Δzᵇ) | (ẑᵗ ≤ ẑᵇ)
    ẑᵇ = ifelse(too_thin, domain_top, ẑᵇ)
    ẑᵗ = ifelse(too_thin, domain_top, ẑᵗ)

    @inbounds bottom_field[i, j, 1] = ẑᵇ
    @inbounds top_field[i, j, 1]    = ẑᵗ
end

function Architectures.on_architecture(arch, ib::PartialCellBottomAndTop)
    return PartialCellBottomAndTop(on_architecture(arch, ib.bottom_height),
                                   on_architecture(arch, ib.top_height),
                                   on_architecture(arch, ib.minimum_fractional_cell_height),
                                   on_architecture(arch, ib.minimum_cell_height),
                                   on_architecture(arch, ib.top_load))
end

Adapt.adapt_structure(to, ib::PartialCellBottomAndTop) = PartialCellBottomAndTop(adapt(to, ib.bottom_height),
                                                                                 adapt(to, ib.top_height),
                                                                                 ib.minimum_fractional_cell_height,
                                                                                 ib.minimum_cell_height,
                                                                                 adapt(to, ib.top_load))

@inline function _immersed_cell(i, j, k, underlying_grid, ib::PartialCellBottomAndTop)
    Δr = Δrᶜᶜᶜ(i, j, k, underlying_grid)
    # Materialized fractions are 0 or ≥ ϵ, so testing at ϵ / 2 avoids rounding ties with the floored heights
    ϵ  = max(ib.minimum_fractional_cell_height, ib.minimum_cell_height / Δr) / 2

    r⁺ = rnode(i, j, k + 1, underlying_grid, c, c, f)
    r★ = r⁺ - Δr * ϵ
    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    bottom_immersed = r★ < rᵇ

    r⁻ = rnode(i, j, k, underlying_grid, c, c, f)
    r☆ = r⁻ + Δr * ϵ
    rᵗ = @inbounds ib.top_height[i, j, 1]
    top_immersed = r☆ > rᵗ

    return bottom_immersed | top_immersed
end

@inline function Δrᶜᶜᶜ(i, j, k, ibg::PCBTIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    r⁻ = rnode(i, j, k,   underlying_grid, c, c, f)
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)

    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    rᵗ = @inbounds ib.top_height[i, j, 1]

    at_the_bottom = bottommost_active_node(i, j, k, ibg, c, c, c)
    at_the_top    = topmost_active_node(i, j, k, ibg, c, c, c)

    full_Δr = Δrᶜᶜᶜ(i, j, k, underlying_grid)

    lower = ifelse(at_the_bottom, rᵇ, r⁻)
    upper = ifelse(at_the_top,    rᵗ, r⁺)
    partial_Δr = upper - lower

    return ifelse(at_the_bottom | at_the_top, partial_Δr, full_Δr)
end

# Only used to size the boundary faces in Δrᶜᶜᶠ; tracers stay at the nominal cell center
@inline function wet_center(i, j, k, ibg::PCBTIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    at_the_bottom = bottommost_active_node(i, j, k, ibg, c, c, c)
    at_the_top    = topmost_active_node(i, j, k, ibg, c, c, c)

    r⁻ = rnode(i, j, k,   underlying_grid, c, c, f)
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)

    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    rᵗ = @inbounds ib.top_height[i, j, 1]

    lower = ifelse(at_the_bottom, rᵇ, r⁻)
    upper = ifelse(at_the_top,    rᵗ, r⁺)
    partial_center = (lower + upper) / 2

    full_center = rnode(i, j, k, underlying_grid, c, c, c)

    return ifelse(at_the_bottom | at_the_top, partial_center, full_center)
end

@inline function Δrᶜᶜᶠ(i, j, k, ibg::PCBTIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    rᵗ = @inbounds ib.top_height[i, j, 1]

    # Beyond the domain top/bottom, compare against the top/bottom height directly
    upper_immersed = ifelse(k <= underlying_grid.Nz,
                             _immersed_cell(i, j, k, underlying_grid, ib),
                             rᵗ < rnode(i, j, k, underlying_grid, c, c, f))
    lower_immersed = ifelse(k - 1 >= 1,
                             _immersed_cell(i, j, k - 1, underlying_grid, ib),
                             rᵇ > rnode(i, j, k, underlying_grid, c, c, f))

    full_Δr = Δrᶜᶜᶠ(i, j, k, underlying_grid)

    upper = ifelse(upper_immersed, rᵗ, wet_center(i, j, k,     ibg))
    lower = ifelse(lower_immersed, rᵇ, wet_center(i, j, k - 1, ibg))
    partial_Δr = upper - lower

    return ifelse(upper_immersed ⊻ lower_immersed, partial_Δr, full_Δr)
end

# Lop the face with max(neighbor bottoms) and min(neighbor tops)
@inline function partial_face_thickness(i, j, k, underlying_grid, rᵇ, rᵗ, at_the_bottom, at_the_top, full_Δr)
    r⁻ = rnode(i, j, k,   underlying_grid, c, c, f)
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)
    lower = ifelse(at_the_bottom, rᵇ, r⁻)
    upper = ifelse(at_the_top,    rᵗ, r⁺)
    partial_Δr = upper - lower
    return ifelse(at_the_bottom | at_the_top, partial_Δr, full_Δr)
end

@inline function Δrᶠᶜᶜ(i, j, k, ibg::PCBTIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    rᵇ = max(@inbounds(ib.bottom_height[i-1, j, 1]),  @inbounds(ib.bottom_height[i, j, 1]))
    rᵗ = min(@inbounds(ib.top_height[i-1, j, 1]), @inbounds(ib.top_height[i, j, 1]))

    at_the_bottom = !peripheral_node(i, j, k, ibg, f, c, c) & peripheral_node(i, j, k-1, ibg, f, c, c)
    at_the_top    = !peripheral_node(i, j, k, ibg, f, c, c) & peripheral_node(i, j, k+1, ibg, f, c, c)

    full_Δr = Δrᶠᶜᶜ(i, j, k, underlying_grid)
    return partial_face_thickness(i, j, k, underlying_grid, rᵇ, rᵗ, at_the_bottom, at_the_top, full_Δr)
end

@inline function Δrᶜᶠᶜ(i, j, k, ibg::PCBTIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    rᵇ = max(@inbounds(ib.bottom_height[i, j-1, 1]),  @inbounds(ib.bottom_height[i, j, 1]))
    rᵗ = min(@inbounds(ib.top_height[i, j-1, 1]), @inbounds(ib.top_height[i, j, 1]))

    at_the_bottom = !peripheral_node(i, j, k, ibg, c, f, c) & peripheral_node(i, j, k-1, ibg, c, f, c)
    at_the_top    = !peripheral_node(i, j, k, ibg, c, f, c) & peripheral_node(i, j, k+1, ibg, c, f, c)

    full_Δr = Δrᶜᶠᶜ(i, j, k, underlying_grid)
    return partial_face_thickness(i, j, k, underlying_grid, rᵇ, rᵗ, at_the_bottom, at_the_top, full_Δr)
end

@inline Δrᶠᶠᶜ(i, j, k, ibg::PCBTIBG) = min(Δrᶠᶜᶜ(i, j-1, k, ibg), Δrᶠᶜᶜ(i, j, k, ibg))

@inline Δrᶠᶜᶠ(i, j, k, ibg::PCBTIBG) = min(Δrᶜᶜᶠ(i-1, j, k, ibg), Δrᶜᶜᶠ(i, j, k, ibg))
@inline Δrᶜᶠᶠ(i, j, k, ibg::PCBTIBG) = min(Δrᶜᶜᶠ(i, j-1, k, ibg), Δrᶜᶜᶠ(i, j, k, ibg))
@inline Δrᶠᶠᶠ(i, j, k, ibg::PCBTIBG) = min(Δrᶠᶜᶠ(i, j-1, k, ibg), Δrᶠᶜᶠ(i, j, k, ibg))

# Make sure Δz works for horizontally-Flat topologies
const XFlatPCBTIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:PartialCellBottomAndTop}
const YFlatPCBTIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:PartialCellBottomAndTop}
const XYFlatPCBTIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Flat, <:Any, <:Any, <:PartialCellBottomAndTop}

@inline Δrᶠᶜᶜ(i, j, k, ibg::XFlatPCBTIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶜᶠ(i, j, k, ibg::XFlatPCBTIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶜᶠᶜ(i, j, k, ibg::YFlatPCBTIBG) = Δrᶜᶜᶜ(i, j, k, ibg)

@inline Δrᶜᶠᶠ(i, j, k, ibg::YFlatPCBTIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XFlatPCBTIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::YFlatPCBTIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XYFlatPCBTIBG) = Δrᶜᶜᶜ(i, j, k, ibg)

# Vertically-static, partial cell bottom and top, immersed boundary grid
const VSPCBTIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractStaticGrid, <:PartialCellBottomAndTop}
@inline Δzᶜᶜᶜ(i, j, k, ibg::VSPCBTIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δzᶠᶜᶜ(i, j, k, ibg::VSPCBTIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶜ(i, j, k, ibg::VSPCBTIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶜᶠ(i, j, k, ibg::VSPCBTIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶜ(i, j, k, ibg::VSPCBTIBG) = Δrᶠᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶠ(i, j, k, ibg::VSPCBTIBG) = Δrᶜᶠᶠ(i, j, k, ibg)
@inline Δzᶠᶜᶠ(i, j, k, ibg::VSPCBTIBG) = Δrᶠᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶠ(i, j, k, ibg::VSPCBTIBG) = Δrᶠᶠᶠ(i, j, k, ibg)

@inline static_column_depthᶜᶜᵃ(i, j, ibg::PCBTIBG) =
    @inbounds ibg.immersed_boundary.top_height[i, j, 1] - ibg.immersed_boundary.bottom_height[i, j, 1]

@inline static_column_depthᶠᶜᵃ(i, j, ibg::PCBTIBG) = active_column_depthᶠᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::PCBTIBG) = active_column_depthᶜᶠᵃ(i, j, ibg)

# Disambiguate against the generic XFlatAGFIBG/YFlatAGFIBG methods in grid_fitted_bottom.jl.
@inline static_column_depthᶠᶜᵃ(i, j, ibg::XFlatPCBTIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::YFlatPCBTIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)

function Grids.constructor_arguments(grid::PCBTIBG)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid.underlying_grid)
    partial_cell_bottom_and_top_args = Dict(:bottom_height => grid.immersed_boundary.bottom_height,
                                            :top_height => grid.immersed_boundary.top_height,
                                            :minimum_fractional_cell_height => grid.immersed_boundary.minimum_fractional_cell_height,
                                            :minimum_cell_height => grid.immersed_boundary.minimum_cell_height,
                                            :top_load => grid.immersed_boundary.top_load)
    return underlying_grid_args, underlying_grid_kwargs, partial_cell_bottom_and_top_args
end

function Base.:(==)(pcc1::PartialCellBottomAndTop, pcc2::PartialCellBottomAndTop)
    return bottom_heights_equal(pcc1.bottom_height, pcc2.bottom_height) &&
           bottom_heights_equal(pcc1.top_height, pcc2.top_height) &&
           pcc1.minimum_fractional_cell_height == pcc2.minimum_fractional_cell_height &&
           pcc1.minimum_cell_height == pcc2.minimum_cell_height &&
           bottom_heights_equal(pcc1.top_load, pcc2.top_load)
end
