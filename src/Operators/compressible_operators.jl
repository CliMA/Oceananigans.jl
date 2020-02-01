using Oceananigans.Operators

using Oceananigans: AbstractGrid

#####
##### Moist density
#####

@inline ρᵐ(i, j, k, grid, ρᵈ, C) = @inbounds ρᵈ[i, j, k] # TODO: * (1 + ...)
@inline ρᵈ_over_ρᵐ(i, j, k, grid, ρᵈ, C) = @inbounds ρᵈ[i, j, k] / ρᵐ(i, j, k, grid, ρᵈ, C)

@inline U_over_ρ(i, j, k, grid, U, ρᵈ) = @inbounds U[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, ρᵈ)
@inline V_over_ρ(i, j, k, grid, V, ρᵈ) = @inbounds V[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, ρᵈ)
@inline W_over_ρ(i, j, k, grid, W, ρᵈ) = @inbounds W[i, j, k] / ℑzᵃᵃᶠ(i, j, k, grid, ρᵈ)
@inline C_over_ρ(i, j, k, grid, C, ρᵈ) = @inbounds C[i, j, k] / ρᵈ[i, j, k]

#####
##### Coriolis terms
#####

@inline x_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, Ũ) where FT = zero(FT)
@inline y_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, Ũ) where FT = zero(FT)
@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, Ũ) where FT = zero(FT)

#####
##### Tracer advection
#####

@inline advective_tracer_flux_x(i, j, k, grid, U, C, ρᵈ) = Ax_ψᵃᵃᶠ(i, j, k, grid, U) * ℑxᶠᵃᵃ(i, j, k, grid, C) / ℑxᶠᵃᵃ(i, j, k, grid, ρᵈ)
@inline advective_tracer_flux_y(i, j, k, grid, V, C, ρᵈ) = Ay_ψᵃᵃᶠ(i, j, k, grid, V) * ℑyᵃᶠᵃ(i, j, k, grid, C) / ℑyᵃᶠᵃ(i, j, k, grid, ρᵈ)
@inline advective_tracer_flux_z(i, j, k, grid, W, C, ρᵈ) = Az_ψᵃᵃᵃ(i, j, k, grid, W) * ℑzᵃᵃᶠ(i, j, k, grid, C) / ℑzᵃᵃᶠ(i, j, k, grid, ρᵈ)

@inline function div_flux(i, j, k, grid, ρᵈ, U, V, W, C)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, advective_tracer_flux_x, U, C, ρᵈ) +
                                    δyᵃᶜᵃ(i, j, k, grid, advective_tracer_flux_y, V, C, ρᵈ) +
                                    δzᵃᵃᶜ(i, j, k, grid, advective_tracer_flux_z, W, C, ρᵈ))
end

#####
##### Diffusion
#####

@inline diffusive_flux_x(i, j, k, grid, κᶠᶜᶜ, ρᵈ, C) = κᶠᶜᶜ * Axᵃᵃᶠ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, C_over_ρ, C, ρᵈ)
@inline diffusive_flux_y(i, j, k, grid, κᶜᶠᶜ, ρᵈ, C) = κᶜᶠᶜ * Ayᵃᵃᶠ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, C_over_ρ, C, ρᵈ)
@inline diffusive_flux_z(i, j, k, grid, κᶜᶜᶠ, ρᵈ, C) = κᶜᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, C_over_ρ, C, ρᵈ)

@inline function ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, κˣ, κʸ, κᶻ, ρᵈ, C)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, diffusive_flux_x, κˣ, ρᵈ, C) +
                                    δyᵃᶜᵃ(i, j, k, grid, diffusive_flux_y, κʸ, ρᵈ, C) +
                                    δzᵃᵃᶜ(i, j, k, grid, diffusive_flux_z, κᶻ, ρᵈ, C))
end

@inline ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, κ, ρᵈ, C) = ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, κ, κ, κ, ρᵈ, C)

@inline function ∇_κ_∇c(i, j, k, grid, closure::ConstantIsotropicDiffusivity,
                        ρᵈ, C, ::Val{tracer_index}, args...) where tracer_index

    @inbounds κ = closure.κ[tracer_index]
    return ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, κ, ρᵈ, C)
end

#####
##### Momentum advection
#####

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
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (  δxᶠᵃᵃ(i, j, k, grid, momentum_flux_ρuu, ρᵈ, Ũ.ρu)
                                    + δyᵃᶜᵃ(i, j, k, grid, momentum_flux_ρuv, ρᵈ, Ũ.ρu, Ũ.ρv)
                                    + δzᵃᵃᶜ(i, j, k, grid, momentum_flux_ρuw, ρᵈ, Ũ.ρu, Ũ.ρw))
end

@inline function div_ρvũ(i, j, k, grid, ρᵈ, Ũ)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (  δxᶜᵃᵃ(i, j, k, grid, momentum_flux_ρvu, ρᵈ, Ũ.ρu, Ũ.ρv)
                                    + δyᵃᶠᵃ(i, j, k, grid, momentum_flux_ρvv, ρᵈ, Ũ.ρv)
                                    + δzᵃᵃᶜ(i, j, k, grid, momentum_flux_ρvw, ρᵈ, Ũ.ρv, Ũ.ρw))
end

@inline function div_ρwũ(i, j, k, grid, ρᵈ, Ũ)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (  δxᶜᵃᵃ(i, j, k, grid, momentum_flux_ρwu, ρᵈ, Ũ.ρu, Ũ.ρw)
                                    + δyᵃᶜᵃ(i, j, k, grid, momentum_flux_ρwv, ρᵈ, Ũ.ρv, Ũ.ρw)
                                    + δzᵃᵃᶠ(i, j, k, grid, momentum_flux_ρww, ρᵈ, Ũ.ρw))
end

#####
##### Viscous dissipation
#####

@inline viscous_flux_ux(i, j, k, grid, νᶜᶜᶜ, ρᵈ, U) = νᶜᶜᶜ * ℑxᶜᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * δxᶜᵃᵃ(i, j, k, grid, U_over_ρ, U, ρᵈ)
@inline viscous_flux_uy(i, j, k, grid, νᶠᶠᶜ, ρᵈ, U) = νᶠᶠᶜ * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * δyᵃᶠᵃ(i, j, k, grid, U_over_ρ, U, ρᵈ)
@inline viscous_flux_uz(i, j, k, grid, νᶠᶜᶠ, ρᵈ, U) = νᶠᶜᶠ * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * δzᵃᵃᶠ(i, j, k, grid, U_over_ρ, U, ρᵈ)

@inline viscous_flux_vx(i, j, k, grid, νᶠᶠᶜ, ρᵈ, V) = νᶠᶠᶜ * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * δxᶠᵃᵃ(i, j, k, grid, V_over_ρ, V, ρᵈ)
@inline viscous_flux_vy(i, j, k, grid, νᶜᶜᶜ, ρᵈ, V) = νᶜᶜᶜ * ℑyᵃᶜᵃ(i, j, k, grid, Ayᵃᵃᶜ) * δyᵃᶜᵃ(i, j, k, grid, V_over_ρ, V, ρᵈ)
@inline viscous_flux_vz(i, j, k, grid, νᶜᶠᶠ, ρᵈ, V) = νᶜᶠᶠ * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * δzᵃᵃᶠ(i, j, k, grid, V_over_ρ, V, ρᵈ)

@inline viscous_flux_wx(i, j, k, grid, νᶠᶜᶠ, ρᵈ, W) = νᶠᶜᶠ * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * δxᶠᵃᵃ(i, j, k, grid, W_over_ρ, W, ρᵈ)
@inline viscous_flux_wy(i, j, k, grid, νᶜᶠᶠ, ρᵈ, W) = νᶜᶠᶠ * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * δyᵃᶠᵃ(i, j, k, grid, W_over_ρ, W, ρᵈ)
@inline viscous_flux_wz(i, j, k, grid, νᶜᶜᶜ, ρᵈ, W) = νᶜᶜᶜ * ℑzᵃᵃᶜ(i, j, k, grid, Azᵃᵃᵃ) * δzᵃᵃᶜ(i, j, k, grid, W_over_ρ, W, ρᵈ)

@inline function ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, νˣ, νʸ, νᶻ, ρᵈ, U)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, viscous_flux_ux, νˣ, ρᵈ, U) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_uy, νʸ, ρᵈ, U) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_uz, νᶻ, ρᵈ, U))
end

@inline function ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, νˣ, νʸ, νᶻ, ρᵈ, V)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_vx, νˣ, ρᵈ, V) +
                                    δyᵃᶠᵃ(i, j, k, grid, viscous_flux_vy, νʸ, ρᵈ, V) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_vz, νᶻ, ρᵈ, V))
end

@inline function ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, νˣ, νʸ, νᶻ, ρᵈ, W)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_wx, νˣ, ρᵈ, W) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_wy, νʸ, ρᵈ, W) +
                                    δzᵃᵃᶠ(i, j, k, grid, viscous_flux_wz, νᶻ, ρᵈ, W))
end

@inline ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, ν, ρᵈ, U) = ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, ν, ν, ν, ρᵈ, U)
@inline ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, ν, ρᵈ, V) = ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, ν, ν, ν, ρᵈ, V)
@inline ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, ν, ρᵈ, W) = ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, ν, ν, ν, ρᵈ, W)

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, ρᵈ, Ũ, args...) = ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, closure.ν, ρᵈ, Ũ.ρu)
@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, ρᵈ, Ũ, args...) = ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, closure.ν, ρᵈ, Ũ.ρv)
@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, ρᵈ, Ũ, args...) = ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, closure.ν, ρᵈ, Ũ.ρw)
