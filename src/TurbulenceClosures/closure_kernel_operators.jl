using Oceananigans.Operators: Δy_qᶠᶜᶜ, Δx_qᶜᶠᶜ, Δx_qᶠᶜᶜ

# Interface for "conditional fluxes" (see ImmersedBoundaries module)
@inline _viscous_flux_ux(args...) = viscous_flux_ux(args...)
@inline _viscous_flux_uy(args...) = viscous_flux_uy(args...)
@inline _viscous_flux_uz(args...) = viscous_flux_uz(args...)
@inline _viscous_flux_vx(args...) = viscous_flux_vx(args...)
@inline _viscous_flux_vy(args...) = viscous_flux_vy(args...)
@inline _viscous_flux_vz(args...) = viscous_flux_vz(args...)
@inline _viscous_flux_wx(args...) = viscous_flux_wx(args...)
@inline _viscous_flux_wy(args...) = viscous_flux_wy(args...)
@inline _viscous_flux_wz(args...) = viscous_flux_wz(args...)

@inline _diffusive_flux_x(args...) = diffusive_flux_x(args...)
@inline _diffusive_flux_y(args...) = diffusive_flux_y(args...)
@inline _diffusive_flux_z(args...) = diffusive_flux_z(args...)

#####
##### Viscous flux divergences
#####

@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, K, clock, fields, buoyancy)
    disc = time_discretization(closure)
    return 1 / Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_qᶜᶜᶜ, _viscous_flux_ux, disc, closure, K, clock, fields, buoyancy) +
                                      δyᵃᶜᵃ(i, j, k, grid, Ay_qᶠᶠᶜ, _viscous_flux_uy, disc, closure, K, clock, fields, buoyancy) +
                                      δzᵃᵃᶜ(i, j, k, grid, Az_qᶠᶜᶠ, _viscous_flux_uz, disc, closure, K, clock, fields, buoyancy))
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, K, clock, fields, buoyancy)
    disc = time_discretization(closure)
    return 1 / Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶠᶜ, _viscous_flux_vx, disc, closure, K, clock, fields, buoyancy) +
                                      δyᵃᶠᵃ(i, j, k, grid, Ay_qᶜᶜᶜ, _viscous_flux_vy, disc, closure, K, clock, fields, buoyancy) +
                                      δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶠᶠ, _viscous_flux_vz, disc, closure, K, clock, fields, buoyancy))
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, K, clock, fields, buoyancy)
    disc = time_discretization(closure)
    return 1 / Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶠ, _viscous_flux_wx, disc, closure, K, clock, fields, buoyancy) +
                                      δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶠ, _viscous_flux_wy, disc, closure, K, clock, fields, buoyancy) +
                                      δzᵃᵃᶠ(i, j, k, grid, Az_qᶜᶜᶜ, _viscous_flux_wz, disc, closure, K, clock, fields, buoyancy))
end

@inline function ∇_dot_qᶜ(i, j, k, grid, closure::AbstractTurbulenceClosure, K, id, c, clock, fields, buoyancy)
    disc = time_discretization(closure)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, _diffusive_flux_x, disc, closure, K, id, c, clock, fields, buoyancy) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, _diffusive_flux_y, disc, closure, K, id, c, clock, fields, buoyancy) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, _diffusive_flux_z, disc, closure, K, id, c, clock, fields, buoyancy))
end
