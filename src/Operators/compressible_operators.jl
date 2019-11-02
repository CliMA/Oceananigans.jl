####
#### Moist density
####

@inline ρᵐ(i, j, k, grid, ρᵈ, C) = @inbounds ρᵈ[i, j, k] # * (1 + ...)

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


