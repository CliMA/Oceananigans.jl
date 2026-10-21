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
    disc = TimeSteppers.time_discretization(closure)
    return (δxᶠᶜᶜ(i, j, k, grid, Ax_qᶜᶜᶜ, _viscous_flux_ux, disc, closure, K, clock, fields, buoyancy) +
            δyᶠᶜᶜ(i, j, k, grid, Ay_qᶠᶠᶜ, _viscous_flux_uy, disc, closure, K, clock, fields, buoyancy) +
            δzᶠᶜᶜ(i, j, k, grid, Az_qᶠᶜᶠ, _viscous_flux_uz, disc, closure, K, clock, fields, buoyancy)) * V⁻¹ᶠᶜᶜ(i, j, k, grid)
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, K, clock, fields, buoyancy)
    disc = TimeSteppers.time_discretization(closure)
    return (δxᶜᶠᶜ(i, j, k, grid, Ax_qᶠᶠᶜ, _viscous_flux_vx, disc, closure, K, clock, fields, buoyancy) +
            δyᶜᶠᶜ(i, j, k, grid, Ay_qᶜᶜᶜ, _viscous_flux_vy, disc, closure, K, clock, fields, buoyancy) +
            δzᶜᶠᶜ(i, j, k, grid, Az_qᶜᶠᶠ, _viscous_flux_vz, disc, closure, K, clock, fields, buoyancy)) * V⁻¹ᶜᶠᶜ(i, j, k, grid)
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, K, clock, fields, buoyancy)
    disc = TimeSteppers.time_discretization(closure)
    return (δxᶜᶜᶠ(i, j, k, grid, Ax_qᶠᶜᶠ, _viscous_flux_wx, disc, closure, K, clock, fields, buoyancy) +
            δyᶜᶜᶠ(i, j, k, grid, Ay_qᶜᶠᶠ, _viscous_flux_wy, disc, closure, K, clock, fields, buoyancy) +
            δzᶜᶜᶠ(i, j, k, grid, Az_qᶜᶜᶜ, _viscous_flux_wz, disc, closure, K, clock, fields, buoyancy)) * V⁻¹ᶜᶜᶠ(i, j, k, grid)
end

@inline function ∇_dot_qᶜ(i, j, k, grid, closure::AbstractTurbulenceClosure, K, id, c, clock, fields, buoyancy)
    disc = TimeSteppers.time_discretization(closure)
    return (δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, _diffusive_flux_x, disc, closure, K, id, c, clock, fields, buoyancy) +
            δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, _diffusive_flux_y, disc, closure, K, id, c, clock, fields, buoyancy) +
            δzᶜᶜᶜ(i, j, k, grid, Az_qᶜᶜᶠ, _diffusive_flux_z, disc, closure, K, id, c, clock, fields, buoyancy)) * V⁻¹ᶜᶜᶜ(i, j, k, grid)
end
