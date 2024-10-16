using Oceananigans.Utils: prettysummary
using Oceananigans.Fields: fill_halo_regions!
using Printf

import Oceananigans.Architectures: on_architecture

#####
##### PartialCellBottom
#####

struct PartialCellBottom{H, E} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    minimum_fractional_cell_height :: E
end

const PCBIBG{FT, TX, TY, TZ} = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:Any, <:PartialCellBottom} where {FT, TX, TY, TZ}

function Base.summary(ib::PartialCellBottom)
    zmax = maximum(parent(ib.bottom_height))
    zmin = minimum(parent(ib.bottom_height))
    zmean = mean(parent(ib.bottom_height))

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
                                                           prettysummary(ib.minimum_fractional_cell_height))

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

function ImmersedBoundaryGrid(grid, ib::PartialCellBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.bottom_height)
    @apply_regionally correct_bottom_height!(bottom_field, grid, ib)
    fill_halo_regions!(bottom_field)
    new_ib = PartialCellBottom(bottom_field, ib.minimum_fractional_cell_height)
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, new_ib)
end

@kernel function _correct_bottom_height!(bottom_field, grid, ib::PartialCellBottom)
    i, j = @index(Global, NTuple)
    zb = @inbounds bottom_field[i, j, 1]
    ϵ  = ib.minimum_fractional_cell_height
    for k in 1:grid.Nz
        # We use `rnode` for the `immersed_cell` because we do not want to have
        # wetting or drying that could happen for a moving grid if we use znode
        z⁻ = rnode(i, j, k, grid, c, c, c)
        # For the same reason, here we use `Δrᶜᶜᶜ` instead of `Δzᶜᶜᶜ`
        Δz = Δrᶜᶜᶜ(i, j, k, grid)
        bottom_cell =  z⁻ + Δz * (1 - ϵ) ≤ zb
        @inbounds bottom_field[i, j, 1] = ifelse(bottom_cell, z⁻ + Δz * (1 - ϵ), zb)
    end
end

function on_architecture(arch, ib::PartialCellBottom{<:Field})
    architecture(ib.bottom_height) == arch && return ib
    arch_grid = on_architecture(arch, ib.bottom_height.grid)
    new_bottom_height = Field{Center, Center, Nothing}(arch_grid)
    copyto!(parent(new_bottom_height), parent(ib.bottom_height))
    return PartialCellBottom(new_bottom_height, ib.minimum_fractional_cell_height)
end

Adapt.adapt_structure(to, ib::PartialCellBottom) = PartialCellBottom(adapt(to, ib.bottom_height),
                                                                     ib.minimum_fractional_cell_height)     

on_architecture(to, ib::PartialCellBottom) = PartialCellBottom(on_architecture(to, ib.bottom_height),
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
    z  = rnode(i, j, k+1, underlying_grid, c, c, f)
    zb = @inbounds ib.bottom_height[i, j, 1]
    return z ≤ zb
end

@inline function bottom_cell(i, j, k, ibg::PCBIBG)
    grid = ibg.underlying_grid
    ib = ibg.immersed_boundary
    # This one's not immersed, but the next one down is
    return !immersed_cell(i, j, k, grid, ib) & immersed_cell(i, j, k-1, grid, ib)
end

@inline function Δrᶜᶜᶜ(i, j, k, ibg::PCBIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary

    # Get node at face above and defining nodes on c,c,f
    z = znode(i, j, k+1, underlying_grid, c, c, f)

    # Get bottom z-coordinate and fractional Δz parameter
    zb = @inbounds ib.bottom_height[i, j, 1]

    # Are we in a bottom cell?
    at_the_bottom = bottom_cell(i, j, k, ibg)

    full_Δz    = Δrᶜᶜᶜ(i, j, k, ibg.underlying_grid)
    partial_Δz = z - zb

    return ifelse(at_the_bottom, partial_Δz, full_Δz)
end

@inline function Δrᶜᶜᶠ(i, j, k, ibg::PCBIBG)
    just_above_bottom = bottom_cell(i, j, k-1, ibg)
    zc = znode(i, j, k, ibg.underlying_grid, c, c, c)
    zf = znode(i, j, k, ibg.underlying_grid, c, c, f)

    full_Δz = Δrᶜᶜᶠ(i, j, k, ibg.underlying_grid)
    partial_Δz = zc - zf + Δrᶜᶜᶜ(i, j, k-1, ibg) / 2

    Δz = ifelse(just_above_bottom, partial_Δz, full_Δz)

    return Δz
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

@inline bottom_height(i, j, ibg::PCBIBG) = @inbounds ibg.immersed_boundary.bottom_height[i, j, 1]

