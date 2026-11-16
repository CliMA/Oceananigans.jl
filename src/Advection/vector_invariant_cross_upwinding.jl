using Oceananigans.Operators: Δrᶜᶜᶜ

#####
##### Cross upwinding results in the largest kinetic energy content,
##### but because of presence of mixed upwinding the truncation error of
##### the numerical discretization is not always negative definite at
##### leading (diffusive) order. This scheme might be unstable at larger orders.
#####

#####
##### Due to the presence of cross derivative terms that generate excessive noise and result in
##### numerical instabilities, it is undesirable to perform a complete upwinding of the Kinetic
##### Energy gradient. Consequently, the `OnlySelfUpwinding` scheme is implemented for the Kinetic
##### Energy gradient in the case of `CrossAndSelfUpwinding`.
##### For details on the implementation refer to the file `vector_invariant_self_upwinding.jl`
#####

#####
##### Cross and Self Upwinding of the Divergence flux
#####

# If the grid is moving, the discrete continuity equation is calculated as:
#
#  ωᵏ⁺¹ - ωᵏ      δx(Ax u) + δy(Ay v)     Δrᶜᶜᶜ ∂t_σ
# ---------- = - --------------------- - -------------
#    Δzᶜᶜᶜ                Vᶜᶜᶜ              Δzᶜᶜᶜ
#
# Where ω is the vertical velocity with respect to a moving grid.
# We upwind the discrete divergence `δx(Ax u) + δy(Ay v)` and then divide by the volume,
# therefore, the correct term to be added to the divergence transport due to the moving grid is:
#
#  Azᶜᶜᶜ Δrᶜᶜᶜ ∂t_σ
#
# which represents the static volume times the time derivative of the vertical grid scaling.
# If the grid is stationary, ∂t_σ evaluates to zero, so this term disappears from the divergence flux.
@inline Az_Δr_∂t_σ(i, j, k, grid) = Azᶜᶜᶜ(i, j, k, grid) * Δrᶜᶜᶜ(i, j, k, grid) * ∂t_σ(i, j, k, grid)

# As for self upwinding, only the velocity differences of δx(Ax u) + δy(Ay v) are upwinded; the change of the face areas
# is interpolated symmetrically with the moving-grid term (see `vector_invariant_self_upwinding.jl`)
@inline δᴿᵁⱽ(i, j, k, grid, u, v) = δᴿᵁ(i, j, k, grid, u, v) + δᴿⱽ(i, j, k, grid, u, v)

@inline δˢᵁⱽ(i, j, k, grid, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, u) * δxᶜᵃᵃ(i, j, k, grid, Axᶠᶜᶜ) +
                                    ℑyᵃᶜᵃ(i, j, k, grid, v) * δyᵃᶜᵃ(i, j, k, grid, Ayᶜᶠᶜ) + Az_Δr_∂t_σ(i, j, k, grid)

@inline function upwinded_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantCrossVerticalUpwinding, u, v)
    @inbounds û = u[i, j, k]
    δ_stencil = scheme.upwinding.divergence_stencil
    cross_scheme = scheme.upwinding.cross_scheme

    δᴿ   =    _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, bias(û), δᴿᵁⱽ, δ_stencil, u, v)
    δˢ   = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, cross_scheme, δˢᵁⱽ, u, v)

    return û * (δᴿ + δˢ)
end

@inline function upwinded_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantCrossVerticalUpwinding, u, v)
    @inbounds v̂ = v[i, j, k]
    δ_stencil = scheme.upwinding.divergence_stencil
    cross_scheme = scheme.upwinding.cross_scheme

    δᴿ   =    _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, bias(v̂), δᴿᵁⱽ, δ_stencil, u, v)
    δˢ   = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, cross_scheme, δˢᵁⱽ, u, v)

    return v̂ * (δᴿ + δˢ)
end
