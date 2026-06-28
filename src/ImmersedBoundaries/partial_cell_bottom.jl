using Oceananigans.Fields: Field, fill_halo_regions!, set!
using Oceananigans.Grids: Grids, bottommost_active_node, AbstractStaticGrid, constructor_arguments
using Oceananigans.Utils: prettysummary

import Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶜᶜᶠ, Δrᶜᶠᶜ, Δrᶜᶠᶠ, Δrᶠᶜᶜ, Δrᶠᶜᶠ, Δrᶠᶠᶜ, Δrᶠᶠᶠ,
                               Δzᶜᶜᶜ, Δzᶜᶜᶠ, Δzᶜᶠᶜ, Δzᶜᶠᶠ, Δzᶠᶜᶜ, Δzᶠᶜᶠ, Δzᶠᶠᶜ, Δzᶠᶠᶠ

#####
##### PartialCellBottom
#####

struct PartialCellBottom{H, E} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    minimum_fractional_cell_height :: E
end

const PCBIBG{FT, TX, TY, TZ} = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:Any, <:PartialCellBottom} where {FT, TX, TY, TZ}

function Base.summary(ib::PartialCellBottom)
    bottom_interior = bottom_height_interior(ib.bottom_height)
    zmax = maximum(bottom_interior)
    zmin = minimum(bottom_interior)
    zmean = mean(bottom_interior)

    summary1 = "PartialCellBottom("

    summary2 = string("mean(zb)=", prettysummary(zmean),
                      ", min(zb)=", prettysummary(zmin),
                      ", max(zb)=", prettysummary(zmax),
                      ", ϵ=", prettysummary(ib.minimum_fractional_cell_height))

    summary3 = ")"

    return summary1 * summary2 * summary3
end

Base.summary(ib::PartialCellBottom{<:Function}) = @sprintf("PartialCellBottom(%s, ϵ=%.1f)",
                                                           prettysummary(ib.bottom_height, false),
                                                           ib.minimum_fractional_cell_height)

function Base.show(io::IO, ib::PartialCellBottom)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "└── minimum_fractional_cell_height: ", prettysummary(ib.minimum_fractional_cell_height))
end

"""
    PartialCellBottom(bottom_height; minimum_fractional_cell_height=0.2)

Return `PartialCellBottom` representing an immersed boundary with "partial"
bottom cells. That is, the height of the bottommost cell in each column is reduced
to fit the provided `bottom_height`, which may be a `Field`, `Array`, or function
of `(x, y)`.

The height of partial bottom cells is greater than

```
minimum_fractional_cell_height * Δz,
```

where `Δz` is the original height of the bottom cell underlying grid.
"""
function PartialCellBottom(bottom_height; minimum_fractional_cell_height=0.2)
    return PartialCellBottom(bottom_height, minimum_fractional_cell_height)
end

function materialize_immersed_boundary(grid, ib::PartialCellBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(bottom_field, ib.bottom_height)

    minimum_fractional_cell_height = convert(eltype(grid), ib.minimum_fractional_cell_height)
    compute_ib = PartialCellBottom(bottom_field, minimum_fractional_cell_height)

    @apply_regionally compute_numerical_bottom_height!(bottom_field, grid, compute_ib)
    fill_halo_regions!(bottom_field)

    return PartialCellBottom(bottom_field.data, minimum_fractional_cell_height)
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

Adapt.adapt_structure(to, ib::PartialCellBottom) = PartialCellBottom(adapt(to, ib.bottom_height),
                                                                     ib.minimum_fractional_cell_height)

Architectures.on_architecture(to, ib::PartialCellBottom) = PartialCellBottom(on_architecture(to, ib.bottom_height),
                                                                             on_architecture(to, ib.minimum_fractional_cell_height))

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
    return r★ < rᵇ
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

@inline function Δrᶜᶜᶠ(i, j, k, ibg::PCBIBG)
    just_above_bottom = bottommost_active_node(i, j, k, ibg, c, c, f)
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
XFlatPCBIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:PartialCellBottom}
YFlatPCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:PartialCellBottom}

@inline Δrᶠᶜᶜ(i, j, k, ibg::XFlatPCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)
@inline Δrᶠᶜᶠ(i, j, k, ibg::XFlatPCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶜᶠᶜ(i, j, k, ibg::YFlatPCBIBG) = Δrᶜᶜᶜ(i, j, k, ibg)

@inline Δrᶜᶠᶠ(i, j, k, ibg::YFlatPCBIBG) = Δrᶜᶜᶠ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::XFlatPCBIBG) = Δrᶜᶠᶜ(i, j, k, ibg)
@inline Δrᶠᶠᶜ(i, j, k, ibg::YFlatPCBIBG) = Δrᶠᶜᶜ(i, j, k, ibg)

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
    return pcb1.bottom_height == pcb2.bottom_height && pcb1.minimum_fractional_cell_height == pcb2.minimum_fractional_cell_height
end
