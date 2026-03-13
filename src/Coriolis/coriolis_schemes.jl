using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.Grids: XFlatGrid, YFlatGrid, XYFlatGrid, inactive_node, peripheral_node
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, IBG

# Typically zero!
@inline z_f_cross_U(i, j, k, grid, ::AbstractRotation, U) = zero(grid)

"""
    ActiveWeightedEnstrophyConserving

Enstrophy-conserving Coriolis scheme with the wet-point correction
of [Jamart and Ozer (1986)](@cite JamartOzer1986).
Near immersed boundaries, the interpolation weights are divided by the number
of active (non-masked) nodes to compensate for missing neighbors.
"""
struct ActiveWeightedEnstrophyConserving end

"""
    ActiveWeightedEnergyConserving

Energy-conserving Coriolis scheme with the wet-point correction
of [Jamart and Ozer (1986)](@cite JamartOzer1986).
Near immersed boundaries, the interpolation weights are divided by the number
of active (non-masked) nodes to compensate for missing neighbors.
"""
struct ActiveWeightedEnergyConserving end

"""
    EENConserving

Energy- and enstrophy-conserving Coriolis scheme based on the triad formulation
of [Arakawa and Lamb (1981)](@cite ArakawaLamb1981).
Each triad at a tracer point sums three of the four surrounding vorticity
values, paired with transports at diagonally adjacent velocity points.
"""
struct EENConserving end

Base.summary(::ActiveWeightedEnstrophyConserving) = "ActiveWeightedEnstrophyConserving"
Base.summary(::ActiveWeightedEnergyConserving) = "ActiveWeightedEnergyConserving"
Base.summary(::EENConserving) = "EENConserving"

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
##### EEN (Energy and Enstrophy conserving, Arakawa & Lamb, 1981) scheme
#####

# Uses triads at the two Center points flanking u and v (respectively).
# Each triad multiplies the transport (Δx * v and Δy * u) at the diagonally-paired points.

# Triads at (Center, Center) each sums 3 of the 4 surrounding f-points.
@inline 𝒯⁺⁺(i, j, k, grid, coriolis) = fᶠᶠᵃ(i,   j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j,   k, grid, coriolis)
@inline 𝒯⁻⁺(i, j, k, grid, coriolis) = fᶠᶠᵃ(i,   j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j+1, k, grid, coriolis)
@inline 𝒯⁺⁻(i, j, k, grid, coriolis) = fᶠᶠᵃ(i+1, j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j,   k, grid, coriolis)
@inline 𝒯⁻⁻(i, j, k, grid, coriolis) = fᶠᶠᵃ(i+1, j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j+1, k, grid, coriolis)

const EENC = AbstractRotation{<:EENConserving}

@inline function x_f_cross_U(i, j, k, grid, coriolis::EENC, U)
    @inbounds begin
        return - Ay⁻¹ᶠᶜᶜ(i, j, k, grid) / 12 * (
            𝒯⁺⁺(i-1, j, k, grid, coriolis) * masked_Ay_qᶜᶠᶜ(i-1, j+1, k, grid, U[2]) +
            𝒯⁻⁺(i,   j, k, grid, coriolis) * masked_Ay_qᶜᶠᶜ(i,   j,   k, grid, U[2]) +
            𝒯⁺⁻(i-1, j, k, grid, coriolis) * masked_Ay_qᶜᶠᶜ(i-1, j,   k, grid, U[2]) +
            𝒯⁻⁻(i,   j, k, grid, coriolis) * masked_Ay_qᶜᶠᶜ(i,   j+1, k, grid, U[2]))
    end
end

# Uses triads at (i,j-1) and (i,j), paired with u-transports (Δy * u).
@inline function y_f_cross_U(i, j, k, grid, coriolis::EENC, U)
    @inbounds begin
        return + Ax⁻¹ᶜᶠᶜ(i, j, k, grid) / 12 * (
            𝒯⁻⁻(i, j,   k, grid, coriolis) * masked_Ax_qᶠᶜᶜ(i,   j,   k, grid, U[1]) +
            𝒯⁺⁺(i, j-1, k, grid, coriolis) * masked_Ax_qᶠᶜᶜ(i+1, j-1, k, grid, U[1]) +
            𝒯⁻⁺(i, j-1, k, grid, coriolis) * masked_Ax_qᶠᶜᶜ(i,   j-1, k, grid, U[1]) +
            𝒯⁺⁻(i, j,   k, grid, coriolis) * masked_Ax_qᶠᶜᶜ(i+1, j,   k, grid, U[1]))
    end
end
