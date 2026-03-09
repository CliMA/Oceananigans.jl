using Oceananigans.Advection: EnergyConserving, EnstrophyConserving

# Typically zero!
@inline z_f_cross_U(i, j, k, grid, ::AbstractRotation, U) = zero(grid)

# Some usefull types to dispatch on
struct ActiveWeightedEnstrophyConserving end
struct ActiveWeightedEnergyConserving end
struct EENConserving end

#####
##### Active Point Enstrophy-conserving scheme
#####

const ESC = AbstractRotation{<:EnstrophyConserving}

@inline x_f_cross_U(i, j, k, grid, coriolis::ESC, U) = @inbounds - ℑxᶠᵃᵃ(i, j, k, grid, fᶜᶜᵃ, coriolis) * ℑxyᶠᶜᵃ(i, j, k, grid, U[2])
@inline y_f_cross_U(i, j, k, grid, coriolis::ESC, U) = @inbounds + ℑyᵃᶠᵃ(i, j, k, grid, fᶜᶜᵃ, coriolis) * ℑxyᶜᶠᵃ(i, j, k, grid, U[1])

#####
##### Energy-conserving scheme
#####

const ENC = AbstractRotation{<:EnergyConserving}

@inline f_ℑy_uᶠᶠᶜ(i, j, k, grid, coriolis::AbstractRotation, u) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)
@inline f_ℑx_vᶠᶠᶜ(i, j, k, grid, coriolis::AbstractRotation, v) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)

@inline x_f_cross_U(i, j, k, grid, coriolis::ENC, U) = @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_vᶠᶠᶜ, coriolis, U[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline y_f_cross_U(i, j, k, grid, coriolis::ENC, U) = @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_uᶠᶠᶜ, coriolis, U[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)

#####
##### Active-weighted schemes
#####

# Helpers for counting active velocity nodes in the 4-point stencil
@inline not_peripheral_nodeᶜᶠᶜ(i, j, k, grid) = !peripheral_node(i, j, k, grid, Center(), Face(), Center())
@inline not_peripheral_nodeᶠᶜᶜ(i, j, k, grid) = !peripheral_node(i, j, k, grid, Face(), Center(), Center())

const AESC = AbstractRotation{<:ActiveWeightedEnstrophyConserving}

@inline function x_f_cross_U(i, j, k, grid, coriolis::AESC, U)
    @inbounds begin
        active_nodes = ℑxyᶠᶜᵃ(i, j, k, grid, not_peripheral_nodeᶜᶠᶜ)
        result = - ℑxᶠᵃᵃ(i, j, k, grid, fᶜᶜᵃ, coriolis) * ℑxyᶠᶜᵃ(i, j, k, grid, U[2])
        return ifelse(active_nodes == 0, zero(grid), result / active_nodes)
    end
end

@inline function y_f_cross_U(i, j, k, grid, coriolis::AESC, U)
    @inbounds begin
        active_nodes = ℑxyᶜᶠᵃ(i, j, k, grid, not_peripheral_nodeᶠᶜᶜ)
        result = ℑyᵃᶠᵃ(i, j, k, grid, fᶜᶜᵃ, coriolis) * ℑxyᶜᶠᵃ(i, j, k, grid, U[1])
        return ifelse(active_nodes == 0, zero(grid), result / active_nodes)
    end
end

const AENC = AbstractRotation{<:ActiveWeightedEnergyConserving}

@inline function x_f_cross_U(i, j, k, grid, coriolis::AENC, U)
    @inbounds begin
        active_nodes = ℑxyᶠᶜᵃ(i, j, k, grid, not_peripheral_nodeᶜᶠᶜ)
        result = - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_vᶠᶠᶜ, coriolis, U[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
        return ifelse(active_nodes == 0, zero(grid), result / active_nodes)
    end
end

@inline function y_f_cross_U(i, j, k, grid, coriolis::AENC, U)
    @inbounds begin
        active_nodes = ℑxyᶜᶠᵃ(i, j, k, grid, not_peripheral_nodeᶠᶜᶜ)
        result = ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_uᶠᶠᶜ, coriolis, U[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
        return ifelse(active_nodes == 0, zero(grid), result / active_nodes)
    end
end

#####
##### EEN (Energy and Enstrophy conserving, Arakawa & Lamb 1981) scheme
#####

# Uses triads at the two Center points flanking u and v (respectively).
# Each triad multiplies the transport (Δx * v and Δy * u) at the diagonally-paired points.

# Triads at (Center, Center) each sums 3 of the 4 surrounding f-points
@inline 𝒯⁺⁺(i, j, k, grid, coriolis) = fᶠᶠᵃ(i,   j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j,   k, grid, coriolis)
@inline 𝒯⁻⁺(i, j, k, grid, coriolis) = fᶠᶠᵃ(i,   j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j+1, k, grid, coriolis)
@inline 𝒯⁺⁻(i, j, k, grid, coriolis) = fᶠᶠᵃ(i+1, j+1, k, grid, coriolis) + fᶠᶠᵃ(i+1, j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j,   k, grid, coriolis)
@inline 𝒯⁻⁻(i, j, k, grid, coriolis) = fᶠᶠᵃ(i+1, j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j,   k, grid, coriolis) + fᶠᶠᵃ(i,   j+1, k, grid, coriolis)

const EENC = AbstractRotation{<:EENConserving}

@inline function x_f_cross_U(i, j, k, grid, coriolis::EENC, U)
    @inbounds begin
        return - Δx⁻¹ᶠᶜᶜ(i, j, k, grid) / 12 * (
            𝒯⁺⁺(i-1, j, k, grid, coriolis) * Δx_qᶜᶠᶜ(i-1, j+1, k, grid, U[2]) +
            𝒯⁻⁺(i,   j, k, grid, coriolis) * Δx_qᶜᶠᶜ(i,   j,   k, grid, U[2]) +
            𝒯⁺⁻(i-1, j, k, grid, coriolis) * Δx_qᶜᶠᶜ(i-1, j,   k, grid, U[2]) +
            𝒯⁻⁻(i,   j, k, grid, coriolis) * Δx_qᶜᶠᶜ(i,   j+1, k, grid, U[2]))
    end
end

# Uses triads at (i,j-1) and (i,j), paired with u-transports (Δy * u).
@inline function y_f_cross_U(i, j, k, grid, coriolis::EENC, U)
    @inbounds begin
        return + Δy⁻¹ᶜᶠᶜ(i, j, k, grid) / 12 * (
            𝒯⁻⁻(i, j,   k, grid, coriolis) * Δy_qᶠᶜᶜ(i,   j,   k, grid, U[1]) +
            𝒯⁺⁺(i, j-1, k, grid, coriolis) * Δy_qᶠᶜᶜ(i+1, j-1, k, grid, U[1]) +
            𝒯⁻⁺(i, j-1, k, grid, coriolis) * Δy_qᶠᶜᶜ(i,   j-1, k, grid, U[1]) +
            𝒯⁺⁻(i, j,   k, grid, coriolis) * Δy_qᶠᶜᶜ(i+1, j,   k, grid, U[1]))
    end
end
