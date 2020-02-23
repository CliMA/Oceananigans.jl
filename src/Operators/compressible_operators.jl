#####
##### Legend:
##### ρ  -> density fields
##### ρŨ -> momentum fields
##### ρC̃ -> conservative tracer fields
##### K̃  -> diffusivity fields
##### Ũ  -> velocity fields
##### C̃  -> tracer fields
#####

#####
##### Convert conservative variables to primitive variables
#####

@inline U_over_ρ(i, j, k, grid, ρ, ρu) = @inbounds ρu[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, ρ)
@inline V_over_ρ(i, j, k, grid, ρ, ρv) = @inbounds ρv[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, ρ)
@inline W_over_ρ(i, j, k, grid, ρ, ρw) = @inbounds ρw[i, j, k] / ℑzᵃᵃᶠ(i, j, k, grid, ρ)
@inline C_over_ρ(i, j, k, grid, ρ, ρc) = @inbounds ρc[i, j, k] / ρ[i, j, k]

#####
##### Kinetic energy
#####

@inline kinetic_energy(i, j, k, grid::AbstractGrid{FT}, ρ, ρũ) where FT =
    @inbounds FT(0.5) * ((ℑxᶜᵃᵃ(i, j, k, grid, ρũ.ρu) / ρ[i, j, k])^2 +
                         (ℑyᵃᶜᵃ(i, j, k, grid, ρũ.ρv) / ρ[i, j, k])^2 +
                         (ℑzᵃᵃᶜ(i, j, k, grid, ρũ.ρw) / ρ[i, j, k])^2)

#####
##### Pressure work
#####

@inline p∂u∂x(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) = ℑxᶠᵃᵃ(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) * Axᵃᵃᶠ(i, j, k, grid) * U_over_ρ(i, j, k, grid, ρ, ρũ.ρu)
@inline p∂v∂y(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) = ℑyᵃᶠᵃ(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) * Ayᵃᵃᶠ(i, j, k, grid) * V_over_ρ(i, j, k, grid, ρ, ρũ.ρv)
@inline p∂w∂z(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) = ℑzᵃᵃᶠ(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) * Azᵃᵃᵃ(i, j, k, grid) * W_over_ρ(i, j, k, grid, ρ, ρũ.ρw)

@inline function ∂ⱼpuⱼ(i, j, k, grid, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, p∂u∂x, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) +
                                    δyᵃᶜᵃ(i, j, k, grid, p∂v∂y, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃) +
                                    δzᵃᵃᶜ(i, j, k, grid, p∂w∂z, diagnose_p, tvar, gases, gravity, ρ, ρũ, ρc̃))
end


#####
##### Tracer advection
#####

@inline advective_tracer_flux_x(i, j, k, grid, U, C, ρ) = Ax_ψᵃᵃᶠ(i, j, k, grid, U) * ℑxᶠᵃᵃ(i, j, k, grid, C) / ℑxᶠᵃᵃ(i, j, k, grid, ρ)
@inline advective_tracer_flux_y(i, j, k, grid, V, C, ρ) = Ay_ψᵃᵃᶠ(i, j, k, grid, V) * ℑyᵃᶠᵃ(i, j, k, grid, C) / ℑyᵃᶠᵃ(i, j, k, grid, ρ)
@inline advective_tracer_flux_z(i, j, k, grid, W, C, ρ) = Az_ψᵃᵃᵃ(i, j, k, grid, W) * ℑzᵃᵃᶠ(i, j, k, grid, C) / ℑzᵃᵃᶠ(i, j, k, grid, ρ)

@inline function div_flux(i, j, k, grid, ρ, U, V, W, C)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, advective_tracer_flux_x, U, C, ρ) +
                                    δyᵃᶜᵃ(i, j, k, grid, advective_tracer_flux_y, V, C, ρ) +
                                    δzᵃᵃᶜ(i, j, k, grid, advective_tracer_flux_z, W, C, ρ))
end



#####
##### Diffusion
##### FIXME: need a better way to enforce boundary conditions on diffusive fluxes at rigid boundaries
#####

@inline diffusive_flux_x(i, j, k, grid, κᶠᶜᶜ, ρ, C) = ℑxᶠᵃᵃ(i, j, k, grid, ρ) * κᶠᶜᶜ * Axᵃᵃᶠ(i, j, k, grid) * ∂xᶠᵃᵃ(i, j, k, grid, C_over_ρ, ρ, C)
@inline diffusive_flux_y(i, j, k, grid, κᶜᶠᶜ, ρ, C) = ℑyᵃᶠᵃ(i, j, k, grid, ρ) * κᶜᶠᶜ * Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶠᵃ(i, j, k, grid, C_over_ρ, ρ, C)
@inline diffusive_flux_z(i, j, k, grid, κᶜᶜᶠ, ρ, C) = ℑzᵃᵃᶠ(i, j, k, grid, ρ) * κᶜᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, C_over_ρ, ρ, C)

@inline diffusive_tvar_flux_x(i, j, k, grid, κᶠᶜᶜ, tvar_diag, tracer_index, tvar, ρ̃, g, ρ, Ũ, C̃, C) = ℑxᶠᵃᵃ(i, j, k, grid, tvar_diag, tracer_index, tvar, ρ̃, g, ρ, Ũ, C̃) * κᶠᶜᶜ * Axᵃᵃᶠ(i, j, k, grid) * ∂xᶠᵃᵃ(i, j, k, grid, C_over_ρ, ρ, C)
@inline diffusive_tvar_flux_y(i, j, k, grid, κᶜᶠᶜ, tvar_diag, tracer_index, tvar, ρ̃, g, ρ, Ũ, C̃, C) = ℑyᵃᶠᵃ(i, j, k, grid, tvar_diag, tracer_index, tvar, ρ̃, g, ρ, Ũ, C̃) * κᶜᶠᶜ * Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶠᵃ(i, j, k, grid, C_over_ρ, ρ, C)
@inline diffusive_tvar_flux_z(i, j, k, grid, κᶜᶜᶠ, tvar_diag, tracer_index, tvar, ρ̃, g, ρ, Ũ, C̃, C) = ℑzᵃᵃᶠ(i, j, k, grid, tvar_diag, tracer_index, tvar, ρ̃, g, ρ, Ũ, C̃) * κᶜᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, C_over_ρ, ρ, C)

@inline diffusive_pressure_flux_x(i, j, k, grid, κᶠᶜᶜ, p_over_ρ_diag, tvar, ρ̃, g, ρ, Ũ, C̃) = ℑxᶠᵃᵃ(i, j, k, grid, ρ) * κᶠᶜᶜ * Axᵃᵃᶠ(i, j, k, grid) * ∂xᶠᵃᵃ(i, j, k, grid, p_over_ρ_diag, tvar, ρ̃, g, ρ, Ũ, C̃)
@inline diffusive_pressure_flux_y(i, j, k, grid, κᶜᶠᶜ, p_over_ρ_diag, tvar, ρ̃, g, ρ, Ũ, C̃) = ℑyᵃᶠᵃ(i, j, k, grid, ρ) * κᶜᶠᶜ * Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶠᵃ(i, j, k, grid, p_over_ρ_diag, tvar, ρ̃, g, ρ, Ũ, C̃)

@inline function diffusive_pressure_flux_z(i, j, k, grid, κᶜᶜᶠ, p_over_ρ_diag, tvar, ρ̃, g, ρ, Ũ, C̃)
    (k <= 1 || k > grid.Nz) && return 0.0
    return ℑzᵃᵃᶠ(i, j, k, grid, ρ) * κᶜᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, p_over_ρ_diag, tvar, ρ̃, g, ρ, Ũ, C̃)
end

@inline function ∂ⱼDᶜⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity,
                       ρ, C, tracer_index, args...)
    @inbounds κ = closure.κ[tracer_index]
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, diffusive_flux_x, κ, ρ, C) +
                                    δyᵃᶜᵃ(i, j, k, grid, diffusive_flux_y, κ, ρ, C) +
                                    δzᵃᵃᶜ(i, j, k, grid, diffusive_flux_z, κ, ρ, C))
end

@inline function ∂ⱼtᶜDᶜⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity,
                         tvar_diag, tracer_index, tvar, ρ̃, g, ρ, Ũ, C̃, C, args ...)
    @inbounds κ = closure.κ[tracer_index]
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, diffusive_tvar_flux_x, κ, tvar_diag, tracer_index, tvar, ρ̃, g, ρ, Ũ, C̃, C) +
                                    δyᵃᶜᵃ(i, j, k, grid, diffusive_tvar_flux_y, κ, tvar_diag, tracer_index, tvar, ρ̃, g, ρ, Ũ, C̃, C) +
                                    δzᵃᵃᶜ(i, j, k, grid, diffusive_tvar_flux_z, κ, tvar_diag, tracer_index, tvar, ρ̃, g, ρ, Ũ, C̃, C))
end

@inline function ∂ⱼDᵖⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity,
                       p_over_ρ_diag, tvar, tracer_index, g, Ũ, ρ̃, C̃, ρ, args ...)
    @inbounds κ = closure.κ[tracer_index]
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, diffusive_pressure_flux_x, κ, p_over_ρ_diag, tvar, ρ̃, g, ρ, Ũ, C̃) +
                                    δyᵃᶜᵃ(i, j, k, grid, diffusive_pressure_flux_y, κ, p_over_ρ_diag, tvar, ρ̃, g, ρ, Ũ, C̃) +
                                    δzᵃᵃᶜ(i, j, k, grid, diffusive_pressure_flux_z, κ, p_over_ρ_diag, tvar, ρ̃, g, ρ, Ũ, C̃))
end

#####
##### Momentum advection
#####

@inline momentum_flux_ρuu(i, j, k, grid, ρ, U)    = @inbounds ℑxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, U) * ℑxᶜᵃᵃ(i, j, k, grid, U) / ρ[i, j, k]
@inline momentum_flux_ρuv(i, j, k, grid, ρ, U, V) =           ℑxᶠᵃᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, V) * ℑyᵃᶠᵃ(i, j, k, grid, U) / ℑxyᶠᶠᵃ(i, j, k, grid, ρ)
@inline momentum_flux_ρuw(i, j, k, grid, ρ, U, W) =           ℑxᶠᵃᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, W) * ℑzᵃᵃᶠ(i, j, k, grid, U) / ℑxzᶠᵃᶠ(i, j, k, grid, ρ)

@inline momentum_flux_ρvu(i, j, k, grid, ρ, U, V) =           ℑyᵃᶠᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, U) * ℑxᶠᵃᵃ(i, j, k, grid, V) / ℑxyᶠᶠᵃ(i, j, k, grid, ρ)
@inline momentum_flux_ρvv(i, j, k, grid, ρ, V)    = @inbounds ℑyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, V) * ℑyᵃᶜᵃ(i, j, k, grid, V) / ρ[i, j, k]
@inline momentum_flux_ρvw(i, j, k, grid, ρ, V, W) =           ℑyᵃᶠᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, W) * ℑzᵃᵃᶠ(i, j, k, grid, V) / ℑyzᵃᶠᶠ(i, j, k, grid, ρ)

@inline momentum_flux_ρwu(i, j, k, grid, ρ, U, W) =           ℑzᵃᵃᶠ(i, j, k, grid, Ax_ψᵃᵃᶠ, U) * ℑxᶠᵃᵃ(i, j, k, grid, W) / ℑxzᶠᵃᶠ(i, j, k, grid, ρ)
@inline momentum_flux_ρwv(i, j, k, grid, ρ, V, W) =           ℑzᵃᵃᶠ(i, j, k, grid, Ay_ψᵃᵃᶠ, V) * ℑyᵃᶠᵃ(i, j, k, grid, W) / ℑyzᵃᶠᶠ(i, j, k, grid, ρ)
@inline momentum_flux_ρww(i, j, k, grid, ρ, W)    = @inbounds ℑzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, W) * ℑzᵃᵃᶜ(i, j, k, grid, W) / ρ[i, j, k]

@inline function div_ρuũ(i, j, k, grid, ρ, Ũ)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (  δxᶠᵃᵃ(i, j, k, grid, momentum_flux_ρuu, ρ, Ũ.ρu)
                                    + δyᵃᶜᵃ(i, j, k, grid, momentum_flux_ρuv, ρ, Ũ.ρu, Ũ.ρv)
                                    + δzᵃᵃᶜ(i, j, k, grid, momentum_flux_ρuw, ρ, Ũ.ρu, Ũ.ρw))
end

@inline function div_ρvũ(i, j, k, grid, ρ, Ũ)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (  δxᶜᵃᵃ(i, j, k, grid, momentum_flux_ρvu, ρ, Ũ.ρu, Ũ.ρv)
                                    + δyᵃᶠᵃ(i, j, k, grid, momentum_flux_ρvv, ρ, Ũ.ρv)
                                    + δzᵃᵃᶜ(i, j, k, grid, momentum_flux_ρvw, ρ, Ũ.ρv, Ũ.ρw))
end

@inline function div_ρwũ(i, j, k, grid, ρ, Ũ)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (  δxᶜᵃᵃ(i, j, k, grid, momentum_flux_ρwu, ρ, Ũ.ρu, Ũ.ρw)
                                    + δyᵃᶜᵃ(i, j, k, grid, momentum_flux_ρwv, ρ, Ũ.ρv, Ũ.ρw)
                                    + δzᵃᵃᶠ(i, j, k, grid, momentum_flux_ρww, ρ, Ũ.ρw))
end

#####
##### Viscous dissipation
#####

@inline strain_rate_tensor_ux(i, j, k, grid::AbstractGrid{FT}, ρ, U) where FT = FT(1/3) * ∂xᶜᵃᵃ(i, j, k, grid, U_over_ρ, ρ, U)
@inline strain_rate_tensor_vy(i, j, k, grid::AbstractGrid{FT}, ρ, V) where FT = FT(1/3) * ∂yᵃᶜᵃ(i, j, k, grid, V_over_ρ, ρ, V)
@inline strain_rate_tensor_wz(i, j, k, grid::AbstractGrid{FT}, ρ, W) where FT = FT(1/3) * ∂zᵃᵃᶜ(i, j, k, grid, W_over_ρ, ρ, W)

@inline strain_rate_tensor_uy(i, j, k, grid, ρ, U, V) = ∂yᵃᶠᵃ(i, j, k, grid, U_over_ρ, ρ, U) + ∂xᶠᵃᵃ(i, j, k, grid, V_over_ρ, ρ, V)
@inline strain_rate_tensor_uz(i, j, k, grid, ρ, U, W) = ∂zᵃᵃᶠ(i, j, k, grid, U_over_ρ, ρ, U) + ∂xᶠᵃᵃ(i, j, k, grid, W_over_ρ, ρ, W)
@inline strain_rate_tensor_vx(i, j, k, grid, ρ, V, U) = ∂xᶠᵃᵃ(i, j, k, grid, V_over_ρ, ρ, V) + ∂yᵃᶠᵃ(i, j, k, grid, U_over_ρ, ρ, U)
@inline strain_rate_tensor_vz(i, j, k, grid, ρ, V, W) = ∂zᵃᵃᶠ(i, j, k, grid, V_over_ρ, ρ, V) + ∂yᵃᶠᵃ(i, j, k, grid, W_over_ρ, ρ, W)
@inline strain_rate_tensor_wx(i, j, k, grid, ρ, W, U) = ∂xᶠᵃᵃ(i, j, k, grid, W_over_ρ, ρ, W) + ∂zᵃᵃᶠ(i, j, k, grid, U_over_ρ, ρ, U)
@inline strain_rate_tensor_wy(i, j, k, grid, ρ, W, V) = ∂yᵃᶠᵃ(i, j, k, grid, W_over_ρ, ρ, W) + ∂zᵃᵃᶠ(i, j, k, grid, V_over_ρ, ρ, V)

@inline viscous_flux_ux(i, j, k, grid, νᶜᶜᶜ, ρ, U)    = @inbounds ρ[i, j, k] * νᶜᶜᶜ * Axᵃᵃᶜ(i, j, k, grid) * strain_rate_tensor_ux(i, j, k, grid, ρ, U)
@inline viscous_flux_vy(i, j, k, grid, νᶜᶜᶜ, ρ, V)    = @inbounds ρ[i, j, k] * νᶜᶜᶜ * Ayᵃᵃᶜ(i, j, k, grid) * strain_rate_tensor_vy(i, j, k, grid, ρ, V)
@inline viscous_flux_wz(i, j, k, grid, νᶜᶜᶜ, ρ, W)    = @inbounds ρ[i, j, k] * νᶜᶜᶜ * Azᵃᵃᵃ(i, j, k, grid) * strain_rate_tensor_wz(i, j, k, grid, ρ, W)

@inline viscous_flux_uy(i, j, k, grid, νᶠᶠᶜ, ρ, U, V) = ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * νᶠᶠᶜ * Ayᵃᵃᶜ(i, j, k, grid) * strain_rate_tensor_uy(i, j, k, grid, ρ, U, V)
@inline viscous_flux_uz(i, j, k, grid, νᶠᶜᶠ, ρ, U, W) = ℑxzᶠᵃᶠ(i, j, k, grid, ρ) * νᶠᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * strain_rate_tensor_uz(i, j, k, grid, ρ, U, W)
@inline viscous_flux_vx(i, j, k, grid, νᶠᶠᶜ, ρ, V, U) = ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * νᶠᶠᶜ * Axᵃᵃᶜ(i, j, k, grid) * strain_rate_tensor_vx(i, j, k, grid, ρ, V, U)
@inline viscous_flux_vz(i, j, k, grid, νᶜᶠᶠ, ρ, V, W) = ℑyzᵃᶠᶠ(i, j, k, grid, ρ) * νᶜᶠᶠ * Azᵃᵃᵃ(i, j, k, grid) * strain_rate_tensor_vz(i, j, k, grid, ρ, V, W)
@inline viscous_flux_wx(i, j, k, grid, νᶠᶜᶠ, ρ, W, U) = ℑxzᶠᵃᶠ(i, j, k, grid, ρ) * νᶠᶜᶠ * Axᵃᵃᶠ(i, j, k, grid) * strain_rate_tensor_wx(i, j, k, grid, ρ, W, U)
@inline viscous_flux_wy(i, j, k, grid, νᶜᶠᶠ, ρ, W, V) = ℑyzᵃᶠᶠ(i, j, k, grid, ρ) * νᶜᶠᶠ * Ayᵃᵃᶠ(i, j, k, grid) * strain_rate_tensor_wy(i, j, k, grid, ρ, W, V)

@inline function ∂ⱼτ₁ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, ρ, Ũ, args...)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, viscous_flux_ux, closure.ν, ρ, Ũ.ρu) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_uy, closure.ν, ρ, Ũ.ρu, Ũ.ρv) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_uz, closure.ν, ρ, Ũ.ρu, Ũ.ρw))
end

@inline function ∂ⱼτ₂ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, ρ, Ũ, args...)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_vx, closure.ν, ρ, Ũ.ρv, Ũ.ρu) +
                                    δyᵃᶠᵃ(i, j, k, grid, viscous_flux_vy, closure.ν, ρ, Ũ.ρv) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_vz, closure.ν, ρ, Ũ.ρv, Ũ.ρw))
end

@inline function ∂ⱼτ₃ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, ρ, Ũ, args...)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_wx, closure.ν, ρ, Ũ.ρw, Ũ.ρu) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_wy, closure.ν, ρ, Ũ.ρw, Ũ.ρv) +
                                    δzᵃᵃᶠ(i, j, k, grid, viscous_flux_wz, closure.ν, ρ, Ũ.ρw))
end

@inline function u₁∂ⱼτ₁ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, ρ, Ũ, args...)
    return U_over_ρ(i, j, k, grid, ρ, Ũ.ρu) * ∂ⱼτ₁ⱼ(i, j, k, grid, closure, ρ, Ũ, args...)
end

@inline function u₂∂ⱼτ₂ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, ρ, Ũ, args...)
    return V_over_ρ(i, j, k, grid, ρ, Ũ.ρv) * ∂ⱼτ₂ⱼ(i, j, k, grid, closure, ρ, Ũ, args...)
end

@inline function u₃∂ⱼτ₃ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, ρ, Ũ, args...)
    return W_over_ρ(i, j, k, grid, ρ, Ũ.ρw) * ∂ⱼτ₃ⱼ(i, j, k, grid, closure, ρ, Ũ, args...)
end

@inline function Q_dissipation(i, j, k, grid, closure::ConstantIsotropicDiffusivity, ρ, Ũ, args...)
    return (ℑxᶜᵃᵃ(i, j, k, grid, u₁∂ⱼτ₁ⱼ, closure, ρ, Ũ, args...)
            + ℑyᵃᶜᵃ(i, j, k, grid, u₂∂ⱼτ₂ⱼ, closure, ρ, Ũ, args...)
            + ℑzᵃᵃᶜ(i, j, k, grid, u₃∂ⱼτ₃ⱼ, closure, ρ, Ũ, args...))
end
