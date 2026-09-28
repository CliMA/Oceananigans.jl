using Oceananigans.Grids: Grids, bottommost_active_node, topmost_active_node, AbstractStaticGrid, constructor_arguments
using Oceananigans.Utils: prettysummary

struct PartialCellCavity{H, C, E, R, L} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    ceiling_height :: C
    minimum_fractional_cell_height :: E
    minimum_cell_height :: R
    ice_load :: L
end

const PCCIBG{FT, TX, TY, TZ} = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:Any, <:PartialCellCavity} where {FT, TX, TY, TZ}

function Base.summary(ib::PartialCellCavity)
    bottom_interior  = bottom_height_interior(ib.bottom_height)
    ceiling_interior = bottom_height_interior(ib.ceiling_height)
    zbmean = sum(bottom_interior) / length(bottom_interior)
    zdmean = sum(ceiling_interior) / length(ceiling_interior)

    return string("PartialCellCavity(mean(zb)=", prettysummary(zbmean),
                  ", mean(zd)=", prettysummary(zdmean),
                  ", ϵ=", prettysummary(ib.minimum_fractional_cell_height),
                  ", minimum_cell_height=", prettysummary(ib.minimum_cell_height), ")")
end

Base.summary(ib::PartialCellCavity{<:Function}) = @sprintf("PartialCellCavity(ϵ=%.1f)", ib.minimum_fractional_cell_height)

function Base.show(io::IO, ib::PartialCellCavity)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "├── ceiling_height: ", prettysummary(ib.ceiling_height), '\n')
    print(io, "├── minimum_fractional_cell_height: ", prettysummary(ib.minimum_fractional_cell_height), '\n')
    print(io, "├── minimum_cell_height: ", prettysummary(ib.minimum_cell_height), '\n')
    print(io, "└── ice_load: ", prettysummary(ib.ice_load))
    return nothing
end

"""
$(TYPEDSIGNATURES)

Return `PartialCellCavity`, an immersed boundary representing the ocean cavity between
`bottom_height` and `ceiling_height` (e.g. beneath an ice shelf whose draft is `ceiling_height`),
in which the bottommost and topmost active cells of every column have fractional heights rather
than being snapped to the underlying grid faces as in [`GridFittedCavity`](@ref).

`bottom_height` and `ceiling_height` may each be an `Array` or a function of `(x, y)`.
Columns where `ceiling_height` is at or above the top of the grid are open.

The height of a partial cell is floored at `max(minimum_fractional_cell_height * Δz, minimum_cell_height)`,
where `Δz` is the underlying cell height.
A cell thinner than half this floor is removed and a thicker one is raised to the floor.

`ice_load` is the potential of the weight of the ice, added to the hydrostatic pressure of every
column: either `nothing` (default, no load), a [`CavityLoad`](@ref)
computed from a reference state, or an array, function of `(x, y)` or two-dimensional field such as
the one returned by [`cavity_load_potential`](@ref).

Example
=======

```jldoctest
julia> using Oceananigans

julia> bottom_height(x, y) = -100
bottom_height (generic function with 1 method)

julia> ceiling_height(x, y) = -20 + 0.01y
ceiling_height (generic function with 1 method)

julia> grid = RectilinearGrid(size=(2, 8, 10), x=(0, 100), y=(0, 100), z=(-100, 0), topology=(Periodic, Periodic, Bounded));

julia> ibg = ImmersedBoundaryGrid(grid, PartialCellCavity(bottom_height, ceiling_height))
2×8×10 ImmersedBoundaryGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo:
├── immersed_boundary: PartialCellCavity(mean(zb)=-100.0, mean(zd)=-20.0, ϵ=0.2, minimum_cell_height=0.0)
├── underlying_grid: 2×8×10 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── Periodic x ∈ [0.0, 100.0)  regularly spaced with Δx=50.0
├── Periodic y ∈ [0.0, 100.0)  regularly spaced with Δy=12.5
└── Bounded  z ∈ [-100.0, 0.0] regularly spaced with Δz=10.0
```
"""
function PartialCellCavity(bottom_height, ceiling_height; minimum_fractional_cell_height=0.2, minimum_cell_height=0, ice_load=nothing)
    return PartialCellCavity(bottom_height, ceiling_height, minimum_fractional_cell_height, minimum_cell_height, ice_load)
end

function materialize_immersed_boundary(grid, ib::PartialCellCavity)
    bottom_field  = Field{Center, Center, Nothing}(grid)
    ceiling_field = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(bottom_field, ib.bottom_height)
    set_bottom_height!(ceiling_field, ib.ceiling_height)

    minimum_fractional_cell_height = convert(eltype(grid), ib.minimum_fractional_cell_height)
    minimum_cell_height = convert(eltype(grid), ib.minimum_cell_height)
    new_ib = PartialCellCavity(bottom_field, ceiling_field, minimum_fractional_cell_height, minimum_cell_height, nothing)

    @apply_regionally compute_numerical_cavity_heights!(bottom_field, ceiling_field, grid, new_ib)
    fill_halo_regions!(bottom_field)
    fill_halo_regions!(ceiling_field)

    unloaded_ib = PartialCellCavity(bottom_field.data, ceiling_field.data, minimum_fractional_cell_height, minimum_cell_height, nothing)
    ice_load = materialize_ice_load(grid, unloaded_ib, ib.ice_load)
    return PartialCellCavity(bottom_field.data, ceiling_field.data, minimum_fractional_cell_height, minimum_cell_height, ice_load)
end

compute_numerical_cavity_heights!(bottom_field, ceiling_field, grid, ib::PartialCellCavity) =
    launch!(architecture(grid), grid, :xy, _compute_numerical_cavity_heights!, bottom_field, ceiling_field, grid, ib)

@kernel function _compute_numerical_cavity_heights!(bottom_field, ceiling_field, grid, ib::PartialCellCavity)
    i, j = @index(Global, NTuple)

    domain_bottom = rnode(i, j, 1, grid, c, c, f)
    domain_top    = rnode(i, j, grid.Nz+1, grid, c, c, f)

    zᵇ = clamp(@inbounds(bottom_field[i, j, 1]),  domain_bottom, domain_top)
    zᵈ = clamp(@inbounds(ceiling_field[i, j, 1]), domain_bottom, domain_top)

    ϵ  = ib.minimum_fractional_cell_height
    Δrᵐⁱⁿ = ib.minimum_cell_height
    has_ceiling = zᵈ < domain_top

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

    ẑᵈ = zᵈ
    for k in grid.Nz : -1 : 1
        z⁻ = rnode(i, j, k,   grid, c, c, f)
        z⁺ = rnode(i, j, k+1, grid, c, c, f)
        Δz = Δrᶜᶜᶜ(i, j, k, grid)
        ϵ★ = max(ϵ, Δrᵐⁱⁿ / Δz)
        ceiling_cell = has_ceiling & (z⁻ < zᵈ) & (zᵈ ≤ z⁺)
        fᵈ = (zᵈ - z⁻) / Δz
        z̃ᵈ = ifelse(fᵈ < ϵ★ / 2, z⁻, z⁻ + max(fᵈ, ϵ★) * Δz)
        ẑᵈ = ifelse(ceiling_cell, z̃ᵈ, ẑᵈ)
    end

    # Close cavities thinner than one floored cell, tested against the unfloored heights
    ϵᵇ = ifelse(Δzᵇ > 0, max(ϵ, Δrᵐⁱⁿ / Δzᵇ), ϵ)
    too_thin = (zᵈ - zᵇ < ϵᵇ * Δzᵇ) | (ẑᵈ ≤ ẑᵇ)
    ẑᵇ = ifelse(too_thin, domain_top, ẑᵇ)
    ẑᵈ = ifelse(too_thin, domain_top, ẑᵈ)

    @inbounds bottom_field[i, j, 1]  = ẑᵇ
    @inbounds ceiling_field[i, j, 1] = ẑᵈ
end

function Architectures.on_architecture(arch, ib::PartialCellCavity)
    return PartialCellCavity(on_architecture(arch, ib.bottom_height),
                             on_architecture(arch, ib.ceiling_height),
                             on_architecture(arch, ib.minimum_fractional_cell_height),
                             on_architecture(arch, ib.minimum_cell_height),
                             on_architecture(arch, ib.ice_load))
end

Adapt.adapt_structure(to, ib::PartialCellCavity) = PartialCellCavity(adapt(to, ib.bottom_height),
                                                                     adapt(to, ib.ceiling_height),
                                                                     ib.minimum_fractional_cell_height,
                                                                     ib.minimum_cell_height,
                                                                     adapt(to, ib.ice_load))

@inline function _immersed_cell(i, j, k, underlying_grid, ib::PartialCellCavity)
    Δr = Δrᶜᶜᶜ(i, j, k, underlying_grid)
    # Materialized fractions are 0 or ≥ ϵ, so testing at ϵ / 2 avoids rounding ties with the floored heights
    ϵ  = max(ib.minimum_fractional_cell_height, ib.minimum_cell_height / Δr) / 2

    r⁺ = rnode(i, j, k + 1, underlying_grid, c, c, f)
    r★ = r⁺ - Δr * ϵ
    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    bottom_immersed = r★ < rᵇ

    r⁻ = rnode(i, j, k, underlying_grid, c, c, f)
    r☆ = r⁻ + Δr * ϵ
    rᵈ = @inbounds ib.ceiling_height[i, j, 1]
    ceiling_immersed = r☆ > rᵈ

    return bottom_immersed | ceiling_immersed
end

@inline function Δrᶜᶜᶜ(i, j, k, ibg::PCCIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    r⁻ = rnode(i, j, k,   underlying_grid, c, c, f)
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)

    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    rᵈ = @inbounds ib.ceiling_height[i, j, 1]

    at_the_bottom = bottommost_active_node(i, j, k, ibg, c, c, c)
    at_the_top    = topmost_active_node(i, j, k, ibg, c, c, c)

    full_Δr = Δrᶜᶜᶜ(i, j, k, underlying_grid)

    lower = ifelse(at_the_bottom, rᵇ, r⁻)
    upper = ifelse(at_the_top,    rᵈ, r⁺)
    partial_Δr = upper - lower

    return ifelse(at_the_bottom | at_the_top, partial_Δr, full_Δr)
end

# Only used to size the boundary faces in Δrᶜᶜᶠ; tracers stay at the nominal cell center
@inline function wet_center(i, j, k, ibg::PCCIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    at_the_bottom = bottommost_active_node(i, j, k, ibg, c, c, c)
    at_the_top    = topmost_active_node(i, j, k, ibg, c, c, c)

    r⁻ = rnode(i, j, k,   underlying_grid, c, c, f)
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)

    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    rᵈ = @inbounds ib.ceiling_height[i, j, 1]

    lower = ifelse(at_the_bottom, rᵇ, r⁻)
    upper = ifelse(at_the_top,    rᵈ, r⁺)
    partial_center = (lower + upper) / 2

    full_center = rnode(i, j, k, underlying_grid, c, c, c)

    return ifelse(at_the_bottom | at_the_top, partial_center, full_center)
end

@inline function Δrᶜᶜᶠ(i, j, k, ibg::PCCIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    rᵈ = @inbounds ib.ceiling_height[i, j, 1]

    # Beyond the domain top/bottom, compare against the ceiling/bottom height directly
    upper_immersed = ifelse(k <= underlying_grid.Nz,
                             _immersed_cell(i, j, k, underlying_grid, ib),
                             rᵈ < rnode(i, j, k, underlying_grid, c, c, f))
    lower_immersed = ifelse(k - 1 >= 1,
                             _immersed_cell(i, j, k - 1, underlying_grid, ib),
                             rᵇ > rnode(i, j, k, underlying_grid, c, c, f))

    full_Δr = Δrᶜᶜᶠ(i, j, k, underlying_grid)

    upper = ifelse(upper_immersed, rᵈ, wet_center(i, j, k,     ibg))
    lower = ifelse(lower_immersed, rᵇ, wet_center(i, j, k - 1, ibg))
    partial_Δr = upper - lower

    return ifelse(upper_immersed ⊻ lower_immersed, partial_Δr, full_Δr)
end

# Lop the face with max(neighbor bottoms) and min(neighbor ceilings)
@inline function partial_face_thickness(i, j, k, underlying_grid, rᵇ, rᵈ, at_the_bottom, at_the_top, full_Δr)
    r⁻ = rnode(i, j, k,   underlying_grid, c, c, f)
    r⁺ = rnode(i, j, k+1, underlying_grid, c, c, f)
    lower = ifelse(at_the_bottom, rᵇ, r⁻)
    upper = ifelse(at_the_top,    rᵈ, r⁺)
    partial_Δr = upper - lower
    return ifelse(at_the_bottom | at_the_top, partial_Δr, full_Δr)
end

@inline function Δrᶠᶜᶜ(i, j, k, ibg::PCCIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    rᵇ = max(@inbounds(ib.bottom_height[i-1, j, 1]),  @inbounds(ib.bottom_height[i, j, 1]))
    rᵈ = min(@inbounds(ib.ceiling_height[i-1, j, 1]), @inbounds(ib.ceiling_height[i, j, 1]))

    at_the_bottom = !peripheral_node(i, j, k, ibg, f, c, c) & peripheral_node(i, j, k-1, ibg, f, c, c)
    at_the_top    = !peripheral_node(i, j, k, ibg, f, c, c) & peripheral_node(i, j, k+1, ibg, f, c, c)

    full_Δr = Δrᶠᶜᶜ(i, j, k, underlying_grid)
    return partial_face_thickness(i, j, k, underlying_grid, rᵇ, rᵈ, at_the_bottom, at_the_top, full_Δr)
end

@inline function Δrᶜᶠᶜ(i, j, k, ibg::PCCIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    rᵇ = max(@inbounds(ib.bottom_height[i, j-1, 1]),  @inbounds(ib.bottom_height[i, j, 1]))
    rᵈ = min(@inbounds(ib.ceiling_height[i, j-1, 1]), @inbounds(ib.ceiling_height[i, j, 1]))

    at_the_bottom = !peripheral_node(i, j, k, ibg, c, f, c) & peripheral_node(i, j, k-1, ibg, c, f, c)
    at_the_top    = !peripheral_node(i, j, k, ibg, c, f, c) & peripheral_node(i, j, k+1, ibg, c, f, c)

    full_Δr = Δrᶜᶠᶜ(i, j, k, underlying_grid)
    return partial_face_thickness(i, j, k, underlying_grid, rᵇ, rᵈ, at_the_bottom, at_the_top, full_Δr)
end

@inline Δrᶠᶠᶜ(i, j, k, ibg::PCCIBG) = min(Δrᶠᶜᶜ(i, j-1, k, ibg), Δrᶠᶜᶜ(i, j, k, ibg))

@inline Δrᶠᶜᶠ(i, j, k, ibg::PCCIBG) = min(Δrᶜᶜᶠ(i-1, j, k, ibg), Δrᶜᶜᶠ(i, j, k, ibg))
@inline Δrᶜᶠᶠ(i, j, k, ibg::PCCIBG) = min(Δrᶜᶜᶠ(i, j-1, k, ibg), Δrᶜᶜᶠ(i, j, k, ibg))
@inline Δrᶠᶠᶠ(i, j, k, ibg::PCCIBG) = min(Δrᶠᶜᶠ(i, j-1, k, ibg), Δrᶠᶜᶠ(i, j, k, ibg))

# Make sure Δz works for horizontally-Flat topologies
const XFlatPCCIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:PartialCellCavity}
const YFlatPCCIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:PartialCellCavity}
const XYFlatPCCIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Flat, <:Any, <:Any, <:PartialCellCavity}

@inline Δrᶠᶜᶜ(i, j, k, ibg::XFlatPCCIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶜᶠ(i, j, k, ibg::XFlatPCCIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶜᶠᶜ(i, j, k, ibg::YFlatPCCIBG) = Δrᶜᶜᶜ(i, j, k, ibg)

@inline Δrᶜᶠᶠ(i, j, k, ibg::YFlatPCCIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XFlatPCCIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::YFlatPCCIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XYFlatPCCIBG) = Δrᶜᶜᶜ(i, j, k, ibg)

# Vertically-static, partial cell cavity, immersed boundary grid
const VSPCCIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractStaticGrid, <:PartialCellCavity}
@inline Δzᶜᶜᶜ(i, j, k, ibg::VSPCCIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δzᶠᶜᶜ(i, j, k, ibg::VSPCCIBG) = Δrᶠᶜᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶜ(i, j, k, ibg::VSPCCIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶜᶠ(i, j, k, ibg::VSPCCIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶜ(i, j, k, ibg::VSPCCIBG) = Δrᶠᶠᶜ(i, j, k, ibg)
@inline Δzᶜᶠᶠ(i, j, k, ibg::VSPCCIBG) = Δrᶜᶠᶠ(i, j, k, ibg)
@inline Δzᶠᶜᶠ(i, j, k, ibg::VSPCCIBG) = Δrᶠᶜᶠ(i, j, k, ibg)
@inline Δzᶠᶠᶠ(i, j, k, ibg::VSPCCIBG) = Δrᶠᶠᶠ(i, j, k, ibg)

@inline static_column_depthᶜᶜᵃ(i, j, ibg::PCCIBG) =
    @inbounds ibg.immersed_boundary.ceiling_height[i, j, 1] - ibg.immersed_boundary.bottom_height[i, j, 1]

@inline static_column_depthᶠᶜᵃ(i, j, ibg::PCCIBG) = active_column_depthᶠᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::PCCIBG) = active_column_depthᶜᶠᵃ(i, j, ibg)

# Disambiguate against the generic XFlatAGFIBG/YFlatAGFIBG methods in grid_fitted_bottom.jl.
@inline static_column_depthᶠᶜᵃ(i, j, ibg::XFlatPCCIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::YFlatPCCIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)

function Grids.constructor_arguments(grid::PCCIBG)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid.underlying_grid)
    partial_cell_cavity_args = Dict(:bottom_height => grid.immersed_boundary.bottom_height,
                                    :ceiling_height => grid.immersed_boundary.ceiling_height,
                                    :minimum_fractional_cell_height => grid.immersed_boundary.minimum_fractional_cell_height,
                                    :minimum_cell_height => grid.immersed_boundary.minimum_cell_height,
                                    :ice_load => grid.immersed_boundary.ice_load)
    return underlying_grid_args, underlying_grid_kwargs, partial_cell_cavity_args
end

function Base.:(==)(pcc1::PartialCellCavity, pcc2::PartialCellCavity)
    return bottom_heights_equal(pcc1.bottom_height, pcc2.bottom_height) &&
           bottom_heights_equal(pcc1.ceiling_height, pcc2.ceiling_height) &&
           pcc1.minimum_fractional_cell_height == pcc2.minimum_fractional_cell_height &&
           pcc1.minimum_cell_height == pcc2.minimum_cell_height &&
           bottom_heights_equal(pcc1.ice_load, pcc2.ice_load)
end
