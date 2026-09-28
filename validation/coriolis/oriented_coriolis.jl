# Oriented Coriolis scheme: the energy-conserving scheme plus a Coriolis pressure built from the grid's own operators,
#
#   ∂t (u, v) = EnergyConserving + grad A + curl* B,    A_c = Σ_z α_cz f_z Γ_z,    B_z = - Σ_c α_cz f_z D_c,
#
# where Γ = δx(Ax v) - δy(Ay u) is the area-weighted circulation at the corners z, D = δx(Ax u) + δy(Ay v) the volume flux
# divergence at the centres c, and α_cz couples each centre to the corners of a 4 × 4 stencil. Since grad = -divᵀ and
# curl* = -curlᵀ in the ΔxΔyΔz inner product, the increment does no work for any α, f, metric or land mask, and
# curl(grad A) = 0 makes the curl of the increment a function of D only. Pairs whose corner touches land or a wall, or whose
# centre lies outside the ocean, are dropped, because there the immersed-boundary differences omit the land faces from Γ
# (or D comes from halo velocities) and the pair would enter one potential but not the other; so are the pairs that cross
# a tripolar fold, which glues the lattice to its 180° rotation and would give a pair different weights from its two
# sides. On a uniform grid with constant f the potential-vorticity interpolation is
#
#   m = cos(kΔ/2) cos(lΔ/2) + 4η S² sin(kΔ/2) sin(lΔ/2) + 2iε S² sin((k-l)Δ/2),   S² = 4 sin²(kΔ/2) + 4 sin²(lΔ/2),
#
# with the weights w_SE = ε - η, w_NW = -ε - η, w_NE = w_SW = η on the corners of each cell. In general the corner-to-centre
# map has the symbol ν = Σ w(p, q) exp(i(pk + ql)Δ) over the offsets p, q ∈ {±1/2, ±3/2}, and m = cos(kΔ/2) cos(lΔ/2) - S² ν*.

using Oceananigans.Coriolis: AbstractRotation, fᶠᶠᵃ, f_ℑx_Ay_vᶠᶠᶜ, f_ℑy_Ax_uᶠᶠᶜ
using Oceananigans.Grids: peripheral_node, inactive_node, topology, RightCenterFolded, RightFaceFolded
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, Ax⁻¹ᶜᶠᶜ, Ay⁻¹ᶠᶜᶜ, Ax_qᶜᶠᶜ, Ay_qᶠᶜᶜ, δxᶠᶠᶜ, δyᶠᶠᶜ, flux_div_xyᶜᶜᶜ,
                              ∂xᶠᶜᶜ, ∂yᶠᶜᶜ, ∂xᶜᶠᶜ, ∂yᶜᶠᶜ, Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ, Δzᶠᶠᶜ

import Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U

# Corner (i + a, j + b) of the centre (i, j) sits at the offset (a - 1/2, b - 1/2)
const corner_offsets = Tuple((a, b) for a in -1:2 for b in -1:2)

struct OrientedCoriolis{FT}
    weights :: NTuple{16, FT}
end

inner_corner_weights(ε, η) = Dict((1/2, -1/2) => ε - η, (-1/2, 1/2) => -ε - η, (1/2, 1/2) => η, (-1/2, -1/2) => η)

"""
    OrientedCoriolis(FT=Float64; ε=1/8, η=1/32, weights=inner_corner_weights(ε, η))

`weights` maps the corner offsets (p, q), p, q ∈ {±1/2, ±3/2}, to the weights of the corner-to-centre map.
"""
function OrientedCoriolis(FT=Float64; ε=1/8, η=1/32, weights=inner_corner_weights(ε, η))
    return OrientedCoriolis(ntuple(n -> FT(get(weights, corner_offsets[n] .- 1/2, 0)), 16))
end

Base.summary(::OrientedCoriolis) = "OrientedCoriolis"

const OrientedRotation = AbstractRotation{<:OrientedCoriolis}

@inline folded_row(j, grid) = (topology(grid, 2) <: Union{RightCenterFolded, RightFaceFolded}) & (j ≥ size(grid, 2))

# Corners on the domain walls are peripheral nodes, so this also drops the pairs that reach beyond a wall
@inline coastal_corner(i, j, k, grid) = peripheral_node(i, j, k, grid, Face(), Face(), Center()) | inactive_node(i, j, k, grid, Face(), Face(), Center())

@inline circulationᶠᶠᶜ(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, Ax_qᶜᶠᶜ, v) - δyᶠᶠᶜ(i, j, k, grid, Ay_qᶠᶜᶜ, u)

# α f for the pair (centre (ic, jc), corner (iz, jz)). Modes varying along the short side of a cell respond at
# (aspect ratio) × f; the min/max factor caps this at ≈ f.
@inline function pair_coupling(ic, jc, iz, jz, k, grid, coriolis, w)
    excluded = folded_row(jc, grid) | folded_row(jz, grid) | coastal_corner(iz, jz, k, grid) |
               inactive_node(ic, jc, k, grid, Center(), Center(), Center())
    f = fᶠᶠᵃ(iz, jz, k, grid, coriolis)
    Δx = Δxᶜᶜᶜ(ic, jc, k, grid)
    Δy = Δyᶜᶜᶜ(ic, jc, k, grid)
    α = w * min(Δx, Δy) / max(Δx, Δy) / sqrt(Δzᶜᶜᶜ(ic, jc, k, grid) * Δzᶠᶠᶜ(iz, jz, k, grid))
    return ifelse(excluded, zero(grid), α * f)
end

@inline function coriolis_pressureᶜᶜᶜ(i, j, k, grid, coriolis, u, v)
    w = coriolis.scheme.weights
    A = zero(grid)
    for n in 1:16
        iszero(w[n]) && continue
        a, b = corner_offsets[n]
        A += pair_coupling(i, j, i+a, j+b, k, grid, coriolis, w[n]) * circulationᶠᶠᶜ(i+a, j+b, k, grid, u, v)
    end
    return A
end

# Corner (i, j) is the corner (a, b) of the centre (i - a, j - b)
@inline function coriolis_streamfunctionᶠᶠᶜ(i, j, k, grid, coriolis, u, v)
    w = coriolis.scheme.weights
    B = zero(grid)
    for n in 1:16
        iszero(w[n]) && continue
        a, b = corner_offsets[n]
        B -= pair_coupling(i-a, j-b, i, j, k, grid, coriolis, w[n]) * flux_div_xyᶜᶜᶜ(i-a, j-b, k, grid, u, v)
    end
    return B
end

@inline oriented_increment_u(i, j, k, grid, coriolis, U) = ∂xᶠᶜᶜ(i, j, k, grid, coriolis_pressureᶜᶜᶜ,       coriolis, U[1], U[2]) -
                                                           ∂yᶠᶜᶜ(i, j, k, grid, coriolis_streamfunctionᶠᶠᶜ, coriolis, U[1], U[2])

@inline oriented_increment_v(i, j, k, grid, coriolis, U) = ∂yᶜᶠᶜ(i, j, k, grid, coriolis_pressureᶜᶜᶜ,       coriolis, U[1], U[2]) +
                                                           ∂xᶜᶠᶜ(i, j, k, grid, coriolis_streamfunctionᶠᶠᶜ, coriolis, U[1], U[2])

@inline x_f_cross_U(i, j, k, grid, coriolis::OrientedRotation, U) = - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_Ay_vᶠᶠᶜ, coriolis, U[2]) * Ay⁻¹ᶠᶜᶜ(i, j, k, grid) - oriented_increment_u(i, j, k, grid, coriolis, U)
@inline y_f_cross_U(i, j, k, grid, coriolis::OrientedRotation, U) = + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_Ax_uᶠᶠᶜ, coriolis, U[1]) * Ax⁻¹ᶜᶠᶜ(i, j, k, grid) - oriented_increment_v(i, j, k, grid, coriolis, U)
