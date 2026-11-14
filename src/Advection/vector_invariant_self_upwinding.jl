using Oceananigans.Operators: δxᶜᶜᶜ, δxᶠᶠᶜ, δyᶜᶜᶜ, δyᶠᶠᶜ

#####
##### Self Upwinding of Divergence Flux, the best option!
#####

@inline δx_U(i, j, k, grid, u, v) = δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline δy_V(i, j, k, grid, u, v) = δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# For moving grids, we include the time-derivative of the grid scaling in the divergence flux.
# If the grid is stationary, `Az_Δr_∂t_σ` evaluates to zero.
@inline δx_U_plus_∂t_σ(i, j, k, grid, u, v) = δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, u) + Az_Δr_∂t_σ(i, j, k, grid)
@inline δy_V_plus_∂t_σ(i, j, k, grid, u, v) = δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, v) + Az_Δr_∂t_σ(i, j, k, grid)

# Velocity smoothness for divergence upwinding
@inline U_smoothness(i, j, k, grid, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline V_smoothness(i, j, k, grid, u, v) = ℑyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# Divergence smoothness for divergence upwinding
@inline divergence_smoothness(i, j, k, grid, u, v) = δx_U(i, j, k, grid, u, v) + δy_V(i, j, k, grid, u, v)

@inline function upwinded_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, u, v)

    δU_stencil   = scheme.upwinding.δU_stencil
    cross_scheme = scheme.upwinding.cross_scheme

    @inbounds û = u[i, j, k]
    δvˢ = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, cross_scheme, δy_V_plus_∂t_σ, u, v)
    δuᴿ =    _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, bias(û), δx_U, δU_stencil, u, v)

    return û * (δvˢ + δuᴿ)
end

@inline function upwinded_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, u, v)

    δV_stencil   = scheme.upwinding.δV_stencil
    cross_scheme = scheme.upwinding.cross_scheme

    @inbounds v̂ = v[i, j, k]
    δuˢ = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, cross_scheme, δx_U_plus_∂t_σ, u, v)
    δvᴿ =    _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, bias(v̂), δy_V, δV_stencil, u, v)

    return v̂ * (δuˢ + δvᴿ)
end

#####
##### Vertical flux consistent with the cross term of the divergence flux
#####

# With continuity, the cross term u ∂y v of the divergence flux cancels the term u ∂z w of the vertical flux ∂z(w u).
# Discretely, this cancellation is exact only if the advecting velocity W is interpolated with the same scheme
# as the cross term. Otherwise the difference couples with the Lamb term and produces a spurious growing mode.

@inline vector_invariant_flux_Wu(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, W, u) =
    self_upwinding_flux_Wu(i, j, k, grid, scheme, scheme.vertical_advection_scheme, TimeSteppers.time_discretization(scheme), W, u)

@inline vector_invariant_flux_Wv(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, W, v) =
    self_upwinding_flux_Wv(i, j, k, grid, scheme, scheme.vertical_advection_scheme, TimeSteppers.time_discretization(scheme), W, v)

@inline self_upwinding_flux_Wu(i, j, k, grid, scheme, vertical_scheme, td, W, u) = vertical_flux_Wu(i, j, k, grid, vertical_scheme, W, u)
@inline self_upwinding_flux_Wv(i, j, k, grid, scheme, vertical_scheme, td, W, v) = vertical_flux_Wv(i, j, k, grid, vertical_scheme, W, v)

@inline function self_upwinding_flux_Wu(i, j, k, grid, scheme, vertical_scheme::UpwindScheme, ::ETD, W, u)
    w̃  = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.upwinding.cross_scheme, Az_qᶜᶜᶠ, W)
    uᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, vertical_scheme, bias(w̃), u)
    return conditional_flux(i, j, k, grid, Face(), Center(), Face(), zero(grid), w̃ * uᴿ)
end

@inline function self_upwinding_flux_Wv(i, j, k, grid, scheme, vertical_scheme::UpwindScheme, ::ETD, W, v)
    w̃  = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.upwinding.cross_scheme, Az_qᶜᶜᶠ, W)
    vᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, vertical_scheme, bias(w̃), v)
    return conditional_flux(i, j, k, grid, Center(), Face(), Face(), zero(grid), w̃ * vᴿ)
end

#####
##### Self Upwinding of Kinetic Energy Gradient
#####

@inline half_ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2 / 2

@inline δx_u²(i, j, k, grid, u, v) = δxᶜᵃᵃ(i, j, k, grid, half_ϕ², u)
@inline δy_u²(i, j, k, grid, u, v) = δyᶠᶠᶜ(i, j, k, grid, half_ϕ², u)

@inline δx_v²(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, half_ϕ², v)
@inline δy_v²(i, j, k, grid, u, v) = δyᵃᶜᵃ(i, j, k, grid, half_ϕ², v)

@inline u_smoothness(i, j, k, grid, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline v_smoothness(i, j, k, grid, u, v) = ℑyᵃᶜᵃ(i, j, k, grid, v)

@inline function bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantKineticEnergyUpwinding, u, v)

    @inbounds û = u[i, j, k]

    δu²_stencil  = scheme.upwinding.δu²_stencil
    cross_scheme = scheme.upwinding.cross_scheme

    δKvˢ = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, cross_scheme, δx_v², u, v)
    δKuᴿ =    _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.kinetic_energy_gradient_scheme, bias(û), δx_u², δu²_stencil, u, v)

    return (δKuᴿ + δKvˢ) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
end

@inline function bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantKineticEnergyUpwinding, u, v)

    @inbounds v̂ = v[i, j, k]

    δv²_stencil  = scheme.upwinding.δv²_stencil
    cross_scheme = scheme.upwinding.cross_scheme

    δKuˢ = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, cross_scheme, δy_u², u, v)
    δKvᴿ =    _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.kinetic_energy_gradient_scheme, bias(v̂), δy_v², δv²_stencil, u, v)

    return (δKvᴿ + δKuˢ) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
end
