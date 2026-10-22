using Oceananigans.Fields: Field, fill_halo_regions!, set!
using Oceananigans.Grids: Grids, AbstractStaticGrid, constructor_arguments
using Oceananigans.Utils: prettysummary, KernelParameters

import Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶜᶜᶠ, Δrᶜᶠᶜ, Δrᶜᶠᶠ, Δrᶠᶜᶜ, Δrᶠᶜᶠ, Δrᶠᶠᶜ, Δrᶠᶠᶠ,
                               Δzᶜᶜᶜ, Δzᶜᶜᶠ, Δzᶜᶠᶜ, Δzᶜᶠᶠ, Δzᶠᶜᶜ, Δzᶠᶜᶠ, Δzᶠᶠᶜ, Δzᶠᶠᶠ,
                               Δr⁻¹ᶜᶜᶜ, Δr⁻¹ᶜᶜᶠ, Δr⁻¹ᶜᶠᶜ, Δr⁻¹ᶜᶠᶠ, Δr⁻¹ᶠᶜᶜ, Δr⁻¹ᶠᶜᶠ, Δr⁻¹ᶠᶠᶜ, Δr⁻¹ᶠᶠᶠ,
                               Δz⁻¹ᶜᶜᶜ, Δz⁻¹ᶜᶜᶠ, Δz⁻¹ᶜᶠᶜ, Δz⁻¹ᶜᶠᶠ, Δz⁻¹ᶠᶜᶜ, Δz⁻¹ᶠᶜᶠ, Δz⁻¹ᶠᶠᶜ, Δz⁻¹ᶠᶠᶠ

#####
##### PartialCellBottom
#####

struct PartialCellBottom{H, E, S, R} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    minimum_fractional_cell_height :: E
    fractional_bottom_index :: S
    # reciprocal vertical spacings of the bottom cell and of its top face, one per column
    Δr⁻¹ᶜᶜᶜ :: R
    Δr⁻¹ᶜᶜᶠ :: R
end

PartialCellBottom(bottom_height, minimum_fractional_cell_height, fractional_bottom_index) =
    PartialCellBottom(bottom_height, minimum_fractional_cell_height, fractional_bottom_index, nothing, nothing)

PartialCellBottom(bottom_height, minimum_fractional_cell_height) =
    PartialCellBottom(bottom_height, minimum_fractional_cell_height, nothing)

const PCBIBG{FT, TX, TY, TZ} = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:Any, <:PartialCellBottom} where {FT, TX, TY, TZ}

function Base.summary(ib::PartialCellBottom)
    bottom_interior = bottom_height_interior(ib.bottom_height)
    zmax = maximum(bottom_interior)
    zmin = minimum(bottom_interior)
    zmean = sum(bottom_interior) / length(bottom_interior)

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
    fill_halo_regions!(bottom_field)

    fractional_bottom_index = Field{Center, Center, Nothing}(grid)

    minimum_fractional_cell_height = convert(eltype(grid), ib.minimum_fractional_cell_height)
    compute_ib = PartialCellBottom(bottom_field, minimum_fractional_cell_height, fractional_bottom_index)

    @apply_regionally compute_numerical_bottom_height!(bottom_field, grid, compute_ib)
    @apply_regionally compute_bottom_cell!(grid, compute_ib)

    Δr⁻¹ᶜᶜᶜ = Field{Center, Center, Nothing}(grid)
    Δr⁻¹ᶜᶜᶠ = Field{Center, Center, Nothing}(grid)

    @apply_regionally compute_bottom_reciprocal_spacings!(Δr⁻¹ᶜᶜᶜ, Δr⁻¹ᶜᶜᶠ, grid, compute_ib)

    return PartialCellBottom(bottom_field.data, minimum_fractional_cell_height, fractional_bottom_index.data,
                             Δr⁻¹ᶜᶜᶜ.data, Δr⁻¹ᶜᶜᶠ.data)
end

function compute_bottom_reciprocal_spacings!(Δr⁻¹ᶜᶜᶜ, Δr⁻¹ᶜᶜᶠ, grid, ib)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    parameters = KernelParameters(1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
    return launch!(architecture(grid), grid, parameters, _compute_bottom_reciprocal_spacings!, Δr⁻¹ᶜᶜᶜ, Δr⁻¹ᶜᶜᶠ, grid, ib)
end

@kernel function _compute_bottom_reciprocal_spacings!(Δr⁻¹ᶜᶜᶜ, Δr⁻¹ᶜᶜᶠ, grid, ib)
    i, j = @index(Global, NTuple)

    sᵇ = @inbounds ib.fractional_bottom_index[i, j, 1]
    kᵇ = unsafe_trunc(Int, sᵇ)

    # Same spacings as `Δrᶜᶜᶜ` at kᵇ and `Δrᶜᶜᶠ` at kᵇ + 1 for a `PartialCellBottom`
    Δrᶜᶜᶜᵇ = (kᵇ + 1 - sᵇ) * Δrᶜᶜᶜ(i, j, kᵇ, grid)

    k  = kᵇ + 1
    rᶜ = rnode(i, j, k, grid, c, c, c)
    rᶠ = rnode(i, j, k, grid, c, c, f)
    Δrᶜᶜᶠᵇ = rᶜ - rᶠ + (k - sᵇ) * Δrᶜᶜᶜ(i, j, kᵇ, grid) / 2

    @inbounds Δr⁻¹ᶜᶜᶜ[i, j, 1] = 1 / Δrᶜᶜᶜᵇ
    @inbounds Δr⁻¹ᶜᶜᶠ[i, j, 1] = 1 / Δrᶜᶜᶠᵇ
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

# `fractional_bottom_index` is the position of the bottom in index space, sᵇ = kᵇ + 1 - φ, where kᵇ is the first active cell
# of the column and φ ∈ [ϵ, 1] the open fraction of its height; sᵇ = Nz + 1 in columns without active cells
@kernel function _compute_bottom_cell!(grid, ib::PartialCellBottom, bottom_field)
    i, j = @index(Global, NTuple)
    rᵇ = @inbounds bottom_field[i, j, 1]
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

    @inbounds ib.fractional_bottom_index[i, j, 1] = kᵇ + 1 - φ
end

# Kernels read the bottom through `fractional_bottom_index` only, see `bottom_height(i, j, ibg)`
Adapt.adapt_structure(to, ib::PartialCellBottom) = PartialCellBottom(nothing,
                                                                     ib.minimum_fractional_cell_height,
                                                                     adapt(to, ib.fractional_bottom_index),
                                                                     adapt(to, ib.Δr⁻¹ᶜᶜᶜ),
                                                                     adapt(to, ib.Δr⁻¹ᶜᶜᶠ))

Architectures.on_architecture(to, ib::PartialCellBottom) = PartialCellBottom(on_architecture(to, ib.bottom_height),
                                                                             on_architecture(to, ib.minimum_fractional_cell_height),
                                                                             on_architecture(to, ib.fractional_bottom_index),
                                                                             on_architecture(to, ib.Δr⁻¹ᶜᶜᶜ),
                                                                             on_architecture(to, ib.Δr⁻¹ᶜᶜᶠ))

# kᵇ = ⌊sᵇ⌋: `unsafe_trunc` is a single conversion, while mixed `Int`-`Float` comparisons of `sᵇ` are exact but several times longer
@inline bottom_cell(i, j, ib::PartialCellBottom) = @inbounds unsafe_trunc(Int, ib.fractional_bottom_index[i, j, 1])

# Measured from the face below kᵇ, so that full bottom cells and columns without active cells are exact
@inline function bottom_height(i, j, ibg::PCBIBG)
    sᵇ = @inbounds ibg.immersed_boundary.fractional_bottom_index[i, j, 1]
    kᵇ = unsafe_trunc(Int, sᵇ)
    return rnode(i, j, kᵇ, ibg.underlying_grid, c, c, f) + (sᵇ - kᵇ) * Δrᶜᶜᶜ(i, j, kᵇ, ibg.underlying_grid)
end

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

@inline function Δr⁻¹ᶜᶜᶜ(i, j, k, ibg::PCBIBG)
    sᵇ = @inbounds ibg.immersed_boundary.fractional_bottom_index[i, j, 1]
    at_the_bottom = k == unsafe_trunc(Int, sᵇ)
    partial_Δr⁻¹  = @inbounds ibg.immersed_boundary.Δr⁻¹ᶜᶜᶜ[i, j, 1]
    full_Δr⁻¹     = Δr⁻¹ᶜᶜᶜ(i, j, k, ibg.underlying_grid)
    return ifelse(at_the_bottom, partial_Δr⁻¹, full_Δr⁻¹)
end

@inline function Δr⁻¹ᶜᶜᶠ(i, j, k, ibg::PCBIBG)
    sᵇ = @inbounds ibg.immersed_boundary.fractional_bottom_index[i, j, 1]
    just_above_bottom = k - 1 == unsafe_trunc(Int, sᵇ)
    partial_Δr⁻¹      = @inbounds ibg.immersed_boundary.Δr⁻¹ᶜᶜᶠ[i, j, 1]
    full_Δr⁻¹         = Δr⁻¹ᶜᶜᶠ(i, j, k, ibg.underlying_grid)
    return ifelse(just_above_bottom, partial_Δr⁻¹, full_Δr⁻¹)
end

# The reciprocal of the smallest spacing is the largest reciprocal
@inline Δr⁻¹ᶠᶜᶜ(i, j, k, ibg::PCBIBG) = max(Δr⁻¹ᶜᶜᶜ(i-1, j, k, ibg), Δr⁻¹ᶜᶜᶜ(i, j, k, ibg))
@inline Δr⁻¹ᶜᶠᶜ(i, j, k, ibg::PCBIBG) = max(Δr⁻¹ᶜᶜᶜ(i, j-1, k, ibg), Δr⁻¹ᶜᶜᶜ(i, j, k, ibg))
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

function Base.:(==)(pcb1::PartialCellBottom, pcb2::PartialCellBottom)
    return bottom_heights_equal(pcb1.bottom_height, pcb2.bottom_height) && pcb1.minimum_fractional_cell_height == pcb2.minimum_fractional_cell_height
end
