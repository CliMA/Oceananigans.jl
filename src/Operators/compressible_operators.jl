####
#### Moist density
####

@inline ρᵐ(i, j, k, grid, ρᵈ) = @inbounds ρᵈ[i, j, k] # * (1 + ...)

####
#### Coriolis terms
####

@inline x_f_cross_U(i, j, k, grid, f, Ũ) = - f * ℑxyᶠᶜᵃ(i, j, k, grid, Ũ.V)
@inline y_f_cross_U(i, j, k, grid, f, Ũ) =   f * ℑxyᶜᶠᵃ(i, j, k, grid, Ũ.U)
@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, f, Ũ) where FT = zero(FT)

####
#### Pressure gradient terms
####

@inline ∂x_pressure(i, j, k, grid, gas, Θᵐ) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, Θᵐ) * ∂xᶠᵃᵃ(i, j, k, grid, Θᵐ)
@inline ∂y_pressure(i, j, k, grid, gas, Θᵐ) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, Θᵐ) * ∂yᵃᶠᵃ(i, j, k, grid, Θᵐ)
@inline ∂z_pressure(i, j, k, grid, gas, Θᵐ) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, Θᵐ) * ∂xᵃᵃᶠ(i, j, k, grid, Θᵐ)

####
#### Buoyancy term
####

@inline buoyancy_perturbation(i, j, k, grid, ρᵈ, g) = g * ρᵐ(i, j, k, grid, ρᵈ)

####
#### Tracer advection
####

@inline advective_tracer_flux_x(i, j, k, grid, U, C, ρᵈ) = Ax_ψᶠᵃᵃ(i, j, k, grid, U) * ℑxᶠᵃᵃ(i, j, k, grid, C) / ℑxᶠᵃᵃ(i, j, k, grid, ρᵈ)
@inline advective_tracer_flux_y(i, j, k, grid, V, C, ρᵈ) = Ay_ψᵃᶠᵃ(i, j, k, grid, V) * ℑyᵃᶠᵃ(i, j, k, grid, C) / ℑyᵃᶠᵃ(i, j, k, grid, ρᵈ)
@inline advective_tracer_flux_z(i, j, k, grid, W, C, ρᵈ) = Az_ψᵃᵃᵃ(i, j, k, grid, W) * ℑzᵃᵃᶠ(i, j, k, grid, C) / ℑzᵃᵃᶠ(i, j, k, grid, ρᵈ)

@inline function div_flux(i, j, k, grid, U, V, W, C, ρᵈ)
    1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, advective_tracer_flux_x, U, C, ρᵈ) +
                             δyᵃᶜᵃ(i, j, k, grid, advective_tracer_flux_y, V, C, ρᵈ) +
                             δzᵃᵃᶜ(i, j, k, grid, advective_tracer_flux_z, W, C, ρᵈ))
end

####
#### Momentum advection
####

@inline momentum_flux_ρuu(i, j, k, grid, ρᵈ, U)    = @inbounds ℑxᶜᵃᵃ(i, j, k, grid, Ax_Ψᶠᵃᵃ, U) * ℑxᶜᵃᵃ(i, j, k, grid, U) / ρᵈ[i, j, k]
@inline momentum_flux_ρuv(i, j, k, grid, ρᵈ, U, V) =           ℑxᶠᵃᵃ(i, j, k, grid, Ay_ψᵃᶠᵃ, V) * ℑyᵃᶠᵃ(i, j, k, grid, U) / ℑxyᶠᶠᵃ(i, j, k, grid, ρᵈ)
@inline momentum_flux_ρuw(i, j, k, grid, ρᵈ, U, W) =           ℑxᶠᵃᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, W) * ℑzᵃᵃᶠ(i, j, k, grid, U) / ℑxzᶠᵃᶠ(i, j, k, grid, ρᵈ)

@inline momentum_flux_ρvu(i, j, k, grid, ρᵈ, U, V) =           ℑyᵃᶠᵃ(i, j, k, grid, Ax_ψᶠᵃᵃ, U) * ℑxᶠᵃᵃ(i, j, k, grid, V) / ℑxyᶠᶠᵃ(i, j, k, grid, ρᵈ)
@inline momentum_flux_ρvv(i, j, k, grid, ρᵈ, V)    = @inbounds ℑyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᶠᵃ, V) * ℑyᵃᶜᵃ(i, j, k, grid, V) / ρᵈ[i, j, k]
@inline momentum_flux_ρvw(i, j, k, grid, ρᵈ, V, W) =           ℑyᵃᶠᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, W) * ℑzᵃᵃᶠ(i, j, k, grid, V) / ℑyzᵃᶠᶠ(i, j, k, grid, ρᵈ)

@inline momentum_flux_ρwu(i, j, k, grid, ρᵈ, U, W) =           ℑzᵃᵃᶠ(i, j, k, grid, Ax_ψᶠᵃᵃ, U) * ℑxᶠᵃᵃ(i, j, k, grid, W) / ℑxzᶠᵃᶠ(i, j, k, grid, ρᵈ)
@inline momentum_flux_ρwv(i, j, k, grid, ρᵈ, V, W) =           ℑzᵃᵃᶠ(i, j, k, grid, Ay_ψᵃᶠᵃ, V) * ℑyᵃᶠᵃ(i, j, k, grid, W) / ℑyzᵃᶠᶠ(i, j, k, grid, ρᵈ)
@inline momentum_flux_ρww(i, j, k, grid, ρᵈ, W)    = @inbounds ℑzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, W) * ℑzᵃᵃᶜ(i, j, k, grid, W) / ρᵈ[i, j, k]


@inline function div_ρuũ(i, j, k, grid, ρᵈ, Ũ)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_ρuu, ρᵈ, Ũ.U)    +
                                    δyᵃᶜᵃ(i, j, k, grid, momentum_flux_ρuv, ρᵈ, Ũ.U, Ũ.V) +
                                    δzᵃᵃᶜ(i, j, k, grid, momentum_flux_ρuw, ρᵈ, Ũ.U, Ũ.W))
end

@inline function div_ρvũ(i, j, k, grid, ρᵈ, Ũ)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_ρvu, ρᵈ, Ũ.U, Ũ.V) +
                                    δyᵃᶠᵃ(i, j, k, grid, momentum_flux_ρvv, ρᵈ, Ũ.V)    +
                                    δzᵃᵃᶜ(i, j, k, grid, momentum_flux_ρvw, ρᵈ, Ũ.V, Ũ.W))
end

@inline function div_ρwũ(i, j, k, grid, ρᵈ, Ũ)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_ρwu, ρᵈ, Ũ.U, Ũ.V) +
                                    δyᵃᶜᵃ(i, j, k, grid, momentum_flux_ρwv, ρᵈ, Ũ.V, Ũ.W) +
                                    δzᵃᵃᶠ(i, j, k, grid, momentum_flux_ρww, ρᵈ, Ũ.W))
end
