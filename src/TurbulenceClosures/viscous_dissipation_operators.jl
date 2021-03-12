#####
##### Viscous flux divergences
#####

@inline function ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, clock, closure::AbstractTurbulenceClosure, args...)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_cᶜᶜᶜ, viscous_flux_ux, clock, closure, args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_ζᶠᶠᶜ, viscous_flux_uy, clock, closure, args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_ηᶠᶜᵃ, viscous_flux_uz, clock, closure, args...))
end

@inline function ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, clock, closure::AbstractTurbulenceClosure, args...)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_ζᶠᶠᶜ, viscous_flux_vx, clock, closure, args...) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_cᶜᶜᶜ, viscous_flux_vy, clock, closure, args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_ξᶜᶠᵃ, viscous_flux_vz, clock, closure, args...))
end

@inline function ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, clock, closure::AbstractTurbulenceClosure, args...)
    return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_ηᶠᶜᶠ, viscous_flux_wx, clock, closure, args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_ξᶜᶠᶠ, viscous_flux_wy, clock, closure, args...) +
                                    δzᵃᵃᶠ(i, j, k, grid, Az_cᶜᶜᵃ, viscous_flux_wz, clock, closure, args...))
end

#####
##### Viscosities at different locations
#####

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::Number) = ν

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::Function) = ν(xnode(Center, i, grid), ynode(Center, j, grid), znode(Center, k, grid), clock.time)
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::Function) = ν(xnode(Face,   i, grid), ynode(Face,   j, grid), znode(Center, k, grid), clock.time)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::Function) = ν(xnode(Face,   i, grid), ynode(Center, j, grid), znode(Face,   k, grid), clock.time)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::Function) = ν(xnode(Center, i, grid), ynode(Face,   j, grid), znode(Face,   k, grid), clock.time)

#####
##### Viscous fluxes
#####

@inline viscous_flux_ux(i, j, k, grid, clock, ν, u) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * ∂xᶜᶜᵃ(i, j, k, grid, u)
@inline viscous_flux_uy(i, j, k, grid, clock, ν, u) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * ∂yᶠᶠᵃ(i, j, k, grid, u)
@inline viscous_flux_uz(i, j, k, grid, clock, ν, u) = νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶠ(i, j, k, grid, u)

@inline viscous_flux_vx(i, j, k, grid, clock, ν, v) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * ∂xᶠᶠᵃ(i, j, k, grid, v)
@inline viscous_flux_vy(i, j, k, grid, clock, ν, v) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * ∂yᶜᶜᵃ(i, j, k, grid, v)
@inline viscous_flux_vz(i, j, k, grid, clock, ν, v) = νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶠ(i, j, k, grid, v)

@inline viscous_flux_wx(i, j, k, grid, clock, ν, w) = νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂xᶠᶜᵃ(i, j, k, grid, w)
@inline viscous_flux_wy(i, j, k, grid, clock, ν, w) = νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂yᶜᶠᵃ(i, j, k, grid, w)
@inline viscous_flux_wz(i, j, k, grid, clock, ν, w) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶜ(i, j, k, grid, w)

#####
##### Products of viscosity and divergence, vorticity, and vertical momentum gradients
#####

@inline ν_δᶜᶜᶜ(i, j, k, grid, clock, ν, u, v) = @inbounds νᶜᶜᶜ(i, j, k, grid, clock, ν) * div_xyᶜᶜᵃ(i, j, k, grid, u, v)
@inline ν_ζᶠᶠᶜ(i, j, k, grid, clock, ν, u, v) = @inbounds νᶠᶠᶜ(i, j, k, grid, clock, ν) * ζ₃ᶠᶠᵃ(i, j, k, grid, u, v)

@inline ν_uzᶠᶜᶠ(i, j, k, grid, clock, ν, u) = @inbounds νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶠ(i, j, k, grid, u)
@inline ν_vzᶜᶠᶠ(i, j, k, grid, clock, ν, v) = @inbounds νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶠ(i, j, k, grid, v)
