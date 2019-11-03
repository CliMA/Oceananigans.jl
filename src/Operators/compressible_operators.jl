####
#### Moist density
####

@inline ρᵐ(i, j, k, grid, ρᵈ, C) = @inbounds ρᵈ[i, j, k] # * (1 + ...)
@inline ρᵈ_over_ρᵐ(i, j, k, grid, ρᵈ, C) = @inbounds ρᵈ[i, j, k] / ρᵐ(i, j, k, grid, ρᵈ, C)

####
#### Coriolis terms
####


@inline x_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, Ũ) where FT = zero(FT)
@inline y_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, Ũ) where FT = zero(FT)
@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, Ũ) where FT = zero(FT)

####
#### Tracer advection
####

@inline advective_tracer_flux_x(i, j, k, grid, U, C, ρᵈ) = Ax_ψᵃᵃᶠ(i, j, k, grid, U) * ℑxᶠᵃᵃ(i, j, k, grid, C) / ℑxᶠᵃᵃ(i, j, k, grid, ρᵈ)
@inline advective_tracer_flux_y(i, j, k, grid, V, C, ρᵈ) = Ay_ψᵃᵃᶠ(i, j, k, grid, V) * ℑyᵃᶠᵃ(i, j, k, grid, C) / ℑyᵃᶠᵃ(i, j, k, grid, ρᵈ)
@inline advective_tracer_flux_z(i, j, k, grid, W, C, ρᵈ) = Az_ψᵃᵃᵃ(i, j, k, grid, W) * ℑzᵃᵃᶠ(i, j, k, grid, C) / ℑzᵃᵃᶠ(i, j, k, grid, ρᵈ)

@inline function div_flux(i, j, k, grid, ρᵈ, U, V, W, C)
    1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, advective_tracer_flux_x, U, C, ρᵈ) +
                             δyᵃᶜᵃ(i, j, k, grid, advective_tracer_flux_y, V, C, ρᵈ) +
                             δzᵃᵃᶜ(i, j, k, grid, advective_tracer_flux_z, W, C, ρᵈ))
end

####
#### Diffusion
####

@inline C_over_ρ(i, j, k, grid, C, ρ) = @inbounds C[i, j, k] / ρ[i, j, k]

@inline diffusive_flux_x(i, j, k, grid, κᶠᶜᶜ, ρᵈ, C) = κᶠᶜᶜ * Axᵃᵃᶠ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, C_over_ρ, C, ρᵈ)
@inline diffusive_flux_y(i, j, k, grid, κᶜᶠᶜ, ρᵈ, C) = κᶜᶠᶜ * Ayᵃᵃᶠ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, C_over_ρ, C, ρᵈ)
@inline diffusive_flux_z(i, j, k, grid, κᶜᶜᶠ, ρᵈ, C) = κᶜᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, C_over_ρ, C, ρᵈ)

@inline function div_κ∇c(i, j, k, grid, κ, ρᵈ, C)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, diffusive_flux_x, κ, ρᵈ, C) +
                                    δyᵃᶜᵃ(i, j, k, grid, diffusive_flux_y, κ, ρᵈ, C) +
                                    δzᵃᵃᶜ(i, j, k, grid, diffusive_flux_z, κ, ρᵈ, C))
end

####
#### Momentum advection
####

@inline momentum_flux_ρuu(i, j, k, grid, ρᵈ, U)    = @inbounds ℑxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, U) * ℑxᶜᵃᵃ(i, j, k, grid, U) / ρᵈ[i, j, k]
@inline momentum_flux_ρuv(i, j, k, grid, ρᵈ, U, V) =           ℑxᶠᵃᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, V) * ℑyᵃᶠᵃ(i, j, k, grid, U) / ℑxyᶠᶠᵃ(i, j, k, grid, ρᵈ)
@inline momentum_flux_ρuw(i, j, k, grid, ρᵈ, U, W) =           ℑxᶠᵃᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, W) * ℑzᵃᵃᶠ(i, j, k, grid, U) / ℑxzᶠᵃᶠ(i, j, k, grid, ρᵈ)

@inline momentum_flux_ρvu(i, j, k, grid, ρᵈ, U, V) =           ℑyᵃᶠᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, U) * ℑxᶠᵃᵃ(i, j, k, grid, V) / ℑxyᶠᶠᵃ(i, j, k, grid, ρᵈ)
@inline momentum_flux_ρvv(i, j, k, grid, ρᵈ, V)    = @inbounds ℑyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, V) * ℑyᵃᶜᵃ(i, j, k, grid, V) / ρᵈ[i, j, k]
@inline momentum_flux_ρvw(i, j, k, grid, ρᵈ, V, W) =           ℑyᵃᶠᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, W) * ℑzᵃᵃᶠ(i, j, k, grid, V) / ℑyzᵃᶠᶠ(i, j, k, grid, ρᵈ)

@inline momentum_flux_ρwu(i, j, k, grid, ρᵈ, U, W) =           ℑzᵃᵃᶠ(i, j, k, grid, Ax_ψᵃᵃᶠ, U) * ℑxᶠᵃᵃ(i, j, k, grid, W) / ℑxzᶠᵃᶠ(i, j, k, grid, ρᵈ)
@inline momentum_flux_ρwv(i, j, k, grid, ρᵈ, V, W) =           ℑzᵃᵃᶠ(i, j, k, grid, Ay_ψᵃᵃᶠ, V) * ℑyᵃᶠᵃ(i, j, k, grid, W) / ℑyzᵃᶠᶠ(i, j, k, grid, ρᵈ)
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

####
#### Viscous dissipation
####

@inline U_over_ρ(i, j, k, grid, U, ρᵈ) = @inbounds U[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, ρᵈ)
@inline V_over_ρ(i, j, k, grid, V, ρᵈ) = @inbounds V[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, ρᵈ)
@inline W_over_ρ(i, j, k, grid, W, ρᵈ) = @inbounds W[i, j, k] / ℑzᵃᵃᶠ(i, j, k, grid, ρᵈ)

@inline viscous_flux_ux(i, j, k, grid, μ_ccc, ρᵈ, U) = μ_ccc * ℑxᶜᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * δxᶜᵃᵃ(i, j, k, grid, U_over_ρ, U, ρᵈ)
@inline viscous_flux_uy(i, j, k, grid, μ_ffc, ρᵈ, U) = μ_ffc * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * δyᵃᶠᵃ(i, j, k, grid, U_over_ρ, U, ρᵈ)
@inline viscous_flux_uz(i, j, k, grid, μ_fcf, ρᵈ, U) = μ_fcf * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * δzᵃᵃᶠ(i, j, k, grid, U_over_ρ, U, ρᵈ)

@inline viscous_flux_vx(i, j, k, grid, μ_ffc, ρᵈ, V) = μ_ffc * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * δxᶠᵃᵃ(i, j, k, grid, V_over_ρ, V, ρᵈ)
@inline viscous_flux_vy(i, j, k, grid, μ_ccc, ρᵈ, V) = μ_ccc * ℑyᵃᶜᵃ(i, j, k, grid, Ayᵃᵃᶜ) * δyᵃᶜᵃ(i, j, k, grid, V_over_ρ, V, ρᵈ)
@inline viscous_flux_vz(i, j, k, grid, μ_cff, ρᵈ, V) = μ_cff * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * δzᵃᵃᶠ(i, j, k, grid, V_over_ρ, V, ρᵈ)

@inline viscous_flux_wx(i, j, k, grid, μᶠᶜᶠ, ρᵈ, W) = μᶠᶜᶠ * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * δxᶠᵃᵃ(i, j, k, grid, W_over_ρ, W, ρᵈ)
@inline viscous_flux_wy(i, j, k, grid, μᶜᶠᶠ, ρᵈ, W) = μᶜᶠᶠ * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * δyᵃᶠᵃ(i, j, k, grid, W_over_ρ, W, ρᵈ)
@inline viscous_flux_wz(i, j, k, grid, μᶜᶜᶜ, ρᵈ, W) = μᶜᶜᶜ * ℑzᵃᵃᶜ(i, j, k, grid, Azᵃᵃᵃ) * δzᵃᵃᶜ(i, j, k, grid, W_over_ρ, W, ρᵈ)

@inline function div_μ∇u(i, j, k, grid, μ, ρᵈ, U)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, viscous_flux_ux, μ, ρᵈ, U) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_uy, μ, ρᵈ, U) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_uz, μ, ρᵈ, U))
end

@inline function div_μ∇v(i, j, k, grid, μ, ρᵈ, V)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_vx, μ, ρᵈ, V) +
                                    δyᵃᶠᵃ(i, j, k, grid, viscous_flux_vy, μ, ρᵈ, V) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_vz, μ, ρᵈ, V))
end

@inline function div_μ∇w(i, j, k, grid, μ, ρᵈ, W)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_wx, μ, ρᵈ, W) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_wy, μ, ρᵈ, W) +
                                    δzᵃᵃᶠ(i, j, k, grid, viscous_flux_wz, μ, ρᵈ, W))
end

