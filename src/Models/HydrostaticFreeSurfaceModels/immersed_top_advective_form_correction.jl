using Oceananigans.Advection: conditional_flux_fcc, conditional_flux_cfc, conditional_flux_ccf
using Oceananigans.ImmersedBoundaries: ImmersedTopIBG, inactive_cell
using Oceananigans.Operators: δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ, Azᶜᶜᶜ, Δrᶜᶜᶜ, V⁻¹ᶜᶜᶜ, ∂t_σ

@inline immersed_top_advective_form_correctionᶜᶜᶜ(i, j, k, grid, advection, velocities, c) = zero(grid)
@inline immersed_top_advective_form_correctionᶜᶜᶜ(i, j, k, grid::ImmersedTopIBG, ::Nothing, velocities, c) = zero(grid)

@inline masked_Ax_uᶠᶜᶜ(i, j, k, grid, u) = conditional_flux_fcc(i, j, k, grid, zero(grid), Ax_qᶠᶜᶜ(i, j, k, grid, u))
@inline masked_Ay_vᶜᶠᶜ(i, j, k, grid, v) = conditional_flux_cfc(i, j, k, grid, zero(grid), Ay_qᶜᶠᶜ(i, j, k, grid, v))
@inline masked_Az_wᶜᶜᶠ(i, j, k, grid, w) = conditional_flux_ccf(i, j, k, grid, zero(grid), Az_qᶜᶜᶠ(i, j, k, grid, w))

@inline function masked_transport_divergenceᶜᶜᶜ(i, j, k, grid, velocities)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, masked_Ax_uᶠᶜᶜ, velocities.u) +
                                    δyᵃᶜᵃ(i, j, k, grid, masked_Ay_vᶜᶠᶜ, velocities.v) +
                                    δzᵃᵃᶜ(i, j, k, grid, masked_Az_wᶜᶜᶠ, velocities.w) +
                                    Azᶜᶜᶜ(i, j, k, grid) * Δrᶜᶜᶜ(i, j, k, grid) * ∂t_σ(i, j, k, grid))
end

# Add c ∇⋅𝐔: the masked transports do not close in the topmost wet cell
@inline function immersed_top_advective_form_correctionᶜᶜᶜ(i, j, k, grid::ImmersedTopIBG, advection, velocities, c)
    correction = @inbounds c[i, j, k] * masked_transport_divergenceᶜᶜᶜ(i, j, k, grid, velocities)
    return ifelse(inactive_cell(i, j, k, grid), zero(grid), correction)
end
