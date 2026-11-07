using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.Grids: peripheral_node
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, IBG

# Typically zero!
@inline z_f_cross_U(i, j, k, grid, ::AbstractRotation, U) = zero(grid)

"""
    ActiveWeightedEnstrophyConserving

Enstrophy-conserving Coriolis scheme with the wet-point correction
of [Jamart and Ozer (1986)](@cite JamartOzer1986).
Near immersed boundaries, the interpolation weights are divided by the number
of active (non-masked) nodes to compensate for missing neighbors.

References
==========
* Jamart, B. M., & Ozer, J. (1986). "Numerical boundary layers and spurious residual flows."
  Journal of Geophysical Research: Oceans, 91(C9), 10621-10631.
"""
struct ActiveWeightedEnstrophyConserving end

"""
    ActiveWeightedEnergyConserving

Energy-conserving Coriolis scheme with the wet-point correction
of [Jamart and Ozer (1986)](@cite JamartOzer1986).
Near immersed boundaries, the interpolation weights are divided by the number
of active (non-masked) nodes to compensate for missing neighbors.

References
==========
* Jamart, B. M., & Ozer, J. (1986). "Numerical boundary layers and spurious residual flows."
  Journal of Geophysical Research: Oceans, 91(C9), 10621-10631.
"""
struct ActiveWeightedEnergyConserving end

"""
    TriadScheme

Energy- and enstrophy-conserving Coriolis scheme based on the triad formulation
of [Arakawa and Lamb (1981)](@cite ArakawaLamb1981).
Each triad at a tracer point sums three of the four surrounding vorticity
values, paired with transports at diagonally adjacent velocity points.

References
==========
* Arakawa, A., & Lamb, V. R. (1981). "A potential enstrophy and energy conserving scheme for the
  shallow water equations." Monthly Weather Review, 109(1), 18-36.
"""
struct TriadScheme end

Base.summary(::EnstrophyConserving) = "EnstrophyConserving"
Base.summary(::EnergyConserving) = "EnergyConserving"
Base.summary(::ActiveWeightedEnstrophyConserving) = "ActiveWeightedEnstrophyConserving"
Base.summary(::ActiveWeightedEnergyConserving) = "ActiveWeightedEnergyConserving"
Base.summary(::TriadScheme) = "TriadScheme"

# Helpers for counting active velocity nodes in the 4-point stencil
@inline not_peripheral_nodeᶜᶠᶜ(i, j, k, grid) = !peripheral_node(i, j, k, grid, Center(), Face(), Center())
@inline not_peripheral_nodeᶠᶜᶜ(i, j, k, grid) = !peripheral_node(i, j, k, grid, Face(), Center(), Center())

@inline masked_Ax_qᶠᶜᶜ(i, j, k, grid, q) = Ax_qᶠᶜᶜ(i, j, k, grid, q)
@inline masked_Ay_qᶜᶠᶜ(i, j, k, grid, q) = Ay_qᶜᶠᶜ(i, j, k, grid, q)

@inline function masked_Ax_qᶠᶜᶜ(i, j, k, grid::IBG, q)
    active = !immersed_peripheral_node(i, j, k, grid, Face(), Center(), Center())
    return ifelse(active, Ax_qᶠᶜᶜ(i, j, k, grid, q), zero(grid))
end

@inline function masked_Ay_qᶜᶠᶜ(i, j, k, grid::IBG, q)
    active = !immersed_peripheral_node(i, j, k, grid, Center(), Face(), Center())
    return ifelse(active, Ay_qᶜᶠᶜ(i, j, k, grid, q), zero(grid))
end

#####
##### Active Point Enstrophy-conserving scheme
#####

const ESC = AbstractRotation{<:EnstrophyConserving}

@inline x_f_cross_U(i, j, k, grid, coriolis::ESC, U) = @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑxyᶠᶜᵃ(i, j, k, grid, masked_Ay_qᶜᶠᶜ, U[2]) * Ay⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline y_f_cross_U(i, j, k, grid, coriolis::ESC, U) = @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑxyᶜᶠᵃ(i, j, k, grid, masked_Ax_qᶠᶜᶜ, U[1]) * Ax⁻¹ᶜᶠᶜ(i, j, k, grid)

#####
##### Energy-conserving scheme
#####

const ENC = AbstractRotation{<:EnergyConserving}

@inline f_ℑx_Ay_vᶠᶠᶜ(i, j, k, grid, coriolis::AbstractRotation, v) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, masked_Ay_qᶜᶠᶜ, v)
@inline f_ℑy_Ax_uᶠᶠᶜ(i, j, k, grid, coriolis::AbstractRotation, u) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, masked_Ax_qᶠᶜᶜ, u)

@inline x_f_cross_U(i, j, k, grid, coriolis::ENC, U) = @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_Ay_vᶠᶠᶜ, coriolis, U[2]) * Ay⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline y_f_cross_U(i, j, k, grid, coriolis::ENC, U) = @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_Ax_uᶠᶠᶜ, coriolis, U[1]) * Ax⁻¹ᶜᶠᶜ(i, j, k, grid)

#####
##### Active-weighted schemes (Jamart & Ozer, 1986)
#####

const AESC = AbstractRotation{<:ActiveWeightedEnstrophyConserving}

@inline function x_f_cross_U(i, j, k, grid, coriolis::AESC, U)
    @inbounds begin
        active_nodes = ℑxyᶠᶜᵃ(i, j, k, grid, not_peripheral_nodeᶜᶠᶜ)
        result = - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑxyᶠᶜᵃ(i, j, k, grid, masked_Ay_qᶜᶠᶜ, U[2])
        return ifelse(active_nodes == 0, zero(grid), result / active_nodes) * Ay⁻¹ᶠᶜᶜ(i, j, k, grid)
    end
end

@inline function y_f_cross_U(i, j, k, grid, coriolis::AESC, U)
    @inbounds begin
        active_nodes = ℑxyᶜᶠᵃ(i, j, k, grid, not_peripheral_nodeᶠᶜᶜ)
        result = ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑxyᶜᶠᵃ(i, j, k, grid, masked_Ax_qᶠᶜᶜ, U[1])
        return ifelse(active_nodes == 0, zero(grid), result / active_nodes) * Ax⁻¹ᶜᶠᶜ(i, j, k, grid)
    end
end

const AENC = AbstractRotation{<:ActiveWeightedEnergyConserving}

@inline function x_f_cross_U(i, j, k, grid, coriolis::AENC, U)
    @inbounds begin
        active_nodes = ℑxyᶠᶜᵃ(i, j, k, grid, not_peripheral_nodeᶜᶠᶜ)
        result = - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_Ay_vᶠᶠᶜ, coriolis, U[2]) * Ay⁻¹ᶠᶜᶜ(i, j, k, grid)
        return ifelse(active_nodes == 0, zero(grid), result / active_nodes)
    end
end

@inline function y_f_cross_U(i, j, k, grid, coriolis::AENC, U)
    @inbounds begin
        active_nodes = ℑxyᶜᶠᵃ(i, j, k, grid, not_peripheral_nodeᶠᶜᶜ)
        result = ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_Ax_uᶠᶠᶜ, coriolis, U[1]) * Ax⁻¹ᶜᶠᶜ(i, j, k, grid)
        return ifelse(active_nodes == 0, zero(grid), result / active_nodes)
    end
end

#####
##### Thickness-weighted energy-conserving scheme
#####

"""
    ThicknessWeightedEnergyConserving()

Energy-conserving Coriolis scheme for grids whose cell thickness ``Δz`` varies horizontally,
such as an `ImmersedBoundaryGrid` with `PartialCellBottom`. It is the [Sadourny (1975)](@cite Sadourny1975)
energy-conserving form written for volume transports 𝒰 = Aˣ u and 𝒱 = Aʸ v, with potential
vorticity q = f / h at cell corners:

    (f × u)ˣ = - ℑʸ(q ℑˣ𝒱) / Δx,    (f × u)ʸ = + ℑˣ(q ℑʸ𝒰) / Δy,

where ℑˣ and ℑʸ are the two-point interpolation operators, which take unweighted averages of
neighbouring values. The corner thickness h is the mean thickness of the active
(non-peripheral) velocity faces that meet at the corner. With these weights the discrete Coriolis work

    Σ Δx Aˣ u (f × u)ˣ + Σ Δy Aʸ v (f × u)ʸ = Σ q (ℑˣ𝒱 ℑʸ𝒰 - ℑʸ𝒰 ℑˣ𝒱)

vanishes identically, corner by corner, for any distribution of Δz. The conserved kinetic energy is
Σ Δx Aˣ u² / 2 + Σ Δy Aʸ v² / 2, which is also the energy that the hydrostatic pressure gradient and
the continuity equation exchange consistently. It equals Σ V u² / 2 + Σ V v² / 2 when Az = Δx Δy.

The `EnergyConserving` and `EnstrophyConserving` (default) schemes divide by the thickness-weighted
face area instead, so their work vanishes only when Δz is uniform. When Δz is uniform,
`ThicknessWeightedEnergyConserving` is identical to `EnergyConserving`.

Using one corner q for both components changes the local Coriolis strength where Δz varies:
a thin face next to thicker ones feels its neighbors' Coriolis acceleration amplified by up to
a factor of 4, and a thick face next to thinner ones feels it reduced. Like `EnergyConserving`, the
scheme does not conserve potential enstrophy. On `PartialCellBottom` grids, the vorticity term of
`VectorInvariant(vorticity_scheme=EnergyConserving())` uses the same flux with q = ζ / h, so the
total (f + ζ) term is energy-neutral.

Example
=======

```jldoctest
julia> using Oceananigans

julia> using Oceananigans.Coriolis: ThicknessWeightedEnergyConserving

julia> HydrostaticSphericalCoriolis(scheme = ThicknessWeightedEnergyConserving())
SphericalCoriolis
├─ rotation rate: 7.29e-05 s⁻¹ = 1.00 Ω_Earth
├─ formulation: HydrostaticFormulation
└─ scheme: ThicknessWeightedEnergyConserving
```
"""
struct ThicknessWeightedEnergyConserving end

Base.summary(::ThicknessWeightedEnergyConserving) = "ThicknessWeightedEnergyConserving"

const TWENC = AbstractRotation{<:ThicknessWeightedEnergyConserving}

@inline x_f_cross_U(i, j, k, grid, coriolis::TWENC, U) = - potential_vorticity_flux_xᶠᶜᶜ(i, j, k, grid, U[2], fᶠᶠᵃ, coriolis)
@inline y_f_cross_U(i, j, k, grid, coriolis::TWENC, U) = + potential_vorticity_flux_yᶜᶠᶜ(i, j, k, grid, U[1], fᶠᶠᵃ, coriolis)

#####
##### EEN (Energy and Enstrophy conserving, Arakawa & Lamb, 1981) scheme
#####

# Uses triads at the two Center points flanking u and v (respectively).
# Each triad multiplies the transport (Δx * v and Δy * u) at the diagonally-paired points.

# Triads at (Center, Center) each sums 3 of the 4 surrounding f-points.
@inline 𝒯⁺⁺(i, j, k, grid, coriolis) = fᶠᶠᵃ(i,   j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j,   k, grid, coriolis)
@inline 𝒯⁻⁺(i, j, k, grid, coriolis) = fᶠᶠᵃ(i,   j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j+1, k, grid, coriolis)
@inline 𝒯⁺⁻(i, j, k, grid, coriolis) = fᶠᶠᵃ(i+1, j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j,   k, grid, coriolis)
@inline 𝒯⁻⁻(i, j, k, grid, coriolis) = fᶠᶠᵃ(i+1, j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j+1, k, grid, coriolis)

const TS = AbstractRotation{<:TriadScheme}

@inline function x_f_cross_U(i, j, k, grid, coriolis::TS, U)
    @inbounds begin
        return - Ay⁻¹ᶠᶜᶜ(i, j, k, grid) / 12 * (
            𝒯⁺⁺(i-1, j, k, grid, coriolis) * masked_Ay_qᶜᶠᶜ(i-1, j+1, k, grid, U[2]) +
            𝒯⁻⁻(i,   j, k, grid, coriolis) * masked_Ay_qᶜᶠᶜ(i,   j,   k, grid, U[2]) +
            𝒯⁺⁻(i-1, j, k, grid, coriolis) * masked_Ay_qᶜᶠᶜ(i-1, j,   k, grid, U[2]) +
            𝒯⁻⁺(i,   j, k, grid, coriolis) * masked_Ay_qᶜᶠᶜ(i,   j+1, k, grid, U[2]))
    end
end

# Uses triads at (i,j-1) and (i,j), paired with u-transports (Δy * u).
@inline function y_f_cross_U(i, j, k, grid, coriolis::TS, U)
    @inbounds begin
        return + Ax⁻¹ᶜᶠᶜ(i, j, k, grid) / 12 * (
            𝒯⁻⁻(i, j,   k, grid, coriolis) * masked_Ax_qᶠᶜᶜ(i,   j,   k, grid, U[1]) +
            𝒯⁺⁺(i, j-1, k, grid, coriolis) * masked_Ax_qᶠᶜᶜ(i+1, j-1, k, grid, U[1]) +
            𝒯⁻⁺(i, j-1, k, grid, coriolis) * masked_Ax_qᶠᶜᶜ(i,   j-1, k, grid, U[1]) +
            𝒯⁺⁻(i, j,   k, grid, coriolis) * masked_Ax_qᶠᶜᶜ(i+1, j,   k, grid, U[1]))
    end
end
