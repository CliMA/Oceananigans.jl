using Oceananigans.Operators: Δy_qᶠᶜᶜ, Δx_qᶜᶠᶜ, Δx_qᶠᶜᶜ, ℑxyᶠᶠᵃ, ℑxzᶠᵃᶠ, ℑyzᵃᶠᶠ

@inline _viscous_flux_ux(args...) = viscous_flux_ux(args...)
@inline _viscous_flux_uy(args...) = viscous_flux_uy(args...)
@inline _viscous_flux_uz(args...) = viscous_flux_uz(args...)
@inline _viscous_flux_vx(args...) = viscous_flux_vx(args...)
@inline _viscous_flux_vy(args...) = viscous_flux_vy(args...)
@inline _viscous_flux_vz(args...) = viscous_flux_vz(args...)
@inline _viscous_flux_wx(args...) = viscous_flux_wx(args...)
@inline _viscous_flux_wy(args...) = viscous_flux_wy(args...)
@inline _viscous_flux_wz(args...) = viscous_flux_wz(args...)

#####                                                            
##### Viscous flux divergences
#####

@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_qᶜᶜᶜ, _viscous_flux_ux, disc, closure, args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_qᶠᶠᶜ, _viscous_flux_uy, disc, closure, args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_qᶠᶜᶠ, _viscous_flux_uz, disc, closure, args...))
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶠᶜ, _viscous_flux_vx, disc, closure, args...) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_qᶜᶜᶜ, _viscous_flux_vy, disc, closure, args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶠᶠ, _viscous_flux_vz, disc, closure, args...))
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶠ, _viscous_flux_wx, disc, closure, args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶠ, _viscous_flux_wy, disc, closure, args...) +
                                    δzᵃᵃᶠ(i, j, k, grid, Az_qᶜᶜᶜ, _viscous_flux_wz, disc, closure, args...))
end

#####
##### Viscosities at different locations
#####

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::Number) = ν

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ν[i, j, k]
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ℑxyᶠᶠᵃ(i, j, k, grid, ν)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ℑxzᶠᵃᶠ(i, j, k, grid, ν)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ℑyzᵃᶠᶠ(i, j, k, grid, ν)

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Center(), i, grid), ynode(Center(), j, grid), znode(Center(), k, grid), clock.time)
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Face(),   i, grid), ynode(Face(),   j, grid), znode(Center(), k, grid), clock.time)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Face(),   i, grid), ynode(Center(), j, grid), znode(Face(),   k, grid), clock.time)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Center(), i, grid), ynode(Face(),   j, grid), znode(Face(),   k, grid), clock.time)

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, Center(), Center(), Center())
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, Face(),   Face(),   Center())
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, Face(),   Center(), Face())
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, Center(), Face(),   Face())

#####
##### Products of viscosity and stress, divergence, vorticity
#####

@inline ν_σᶜᶜᶜ(i, j, k, grid, clock, ν, σᶜᶜᶜ, args...) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * σᶜᶜᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶠᶜ(i, j, k, grid, clock, ν, σᶠᶠᶜ, args...) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * σᶠᶠᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶜᶠ(i, j, k, grid, clock, ν, σᶠᶜᶠ, args...) = νᶠᶜᶠ(i, j, k, grid, clock, ν) * σᶠᶜᶠ(i, j, k, grid, args...)
@inline ν_σᶜᶠᶠ(i, j, k, grid, clock, ν, σᶜᶠᶠ, args...) = νᶜᶠᶠ(i, j, k, grid, clock, ν) * σᶜᶠᶠ(i, j, k, grid, args...)

@inline ν_δᶜᶜᶜ(i, j, k, grid, clock, ν, u, v) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * div_xyᶜᶜᶜ(i, j, k, grid, u, v)
@inline ν_ζᶠᶠᶜ(i, j, k, grid, clock, ν, u, v) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
