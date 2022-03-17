using Oceananigans.Operators: Δy_qᶠᶜᶜ, Δx_qᶜᶠᶜ, Δx_qᶠᶜᶜ, ℑxyᶠᶠᵃ, ℑxzᶠᵃᶠ, ℑyzᵃᶠᶠ

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

@inline function ∇_dot_qᶜ(i, j, k, grid, closure::AbstractTurbulenceClosure, tracer_index, args...)
    disc = time_discretization(closure)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, _diffusive_flux_x, disc, closure, tracer_index, args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, _diffusive_flux_y, disc, closure, tracer_index, args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, _diffusive_flux_z, disc, closure, tracer_index, args...))
end

#####
##### Products of viscosity and stress, divergence, vorticity
#####

@inline ν_σᶜᶜᶜ(i, j, k, grid, closure, K, clock, σᶜᶜᶜ, args...) = νᶜᶜᶜ(i, j, k, grid, closure, K, clock) * σᶜᶜᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶠᶜ(i, j, k, grid, closure, K, clock, σᶠᶠᶜ, args...) = νᶠᶠᶜ(i, j, k, grid, closure, K, clock) * σᶠᶠᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, σᶠᶜᶠ, args...) = νᶠᶜᶠ(i, j, k, grid, closure, K, clock) * σᶠᶜᶠ(i, j, k, grid, args...)
@inline ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, σᶜᶠᶠ, args...) = νᶜᶠᶠ(i, j, k, grid, closure, K, clock) * σᶜᶠᶠ(i, j, k, grid, args...)

@inline ν_δᶜᶜᶜ(i, j, k, grid, closure, K, clock, u, v) = νᶜᶜᶜ(i, j, k, grid, closure, K, clock) * div_xyᶜᶜᶜ(i, j, k, grid, u, v)
@inline ν_ζᶠᶠᶜ(i, j, k, grid, closure, K, clock, u, v) = νᶠᶠᶜ(i, j, k, grid, closure, K, clock) * ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)

@inline κ_σᶠᶜᶜ(i, j, k, grid, closure, K, id, clock, σᶠᶜᶜ, args...) = κᶠᶜᶜ(i, j, k, grid, closure, K, id, clock) * σᶠᶜᶜ(i, j, k, grid, args...)
@inline κ_σᶜᶠᶜ(i, j, k, grid, closure, K, id, clock, σᶜᶠᶜ, args...) = κᶜᶠᶜ(i, j, k, grid, closure, K, id, clock) * σᶜᶠᶜ(i, j, k, grid, args...)
@inline κ_σᶜᶜᶠ(i, j, k, grid, closure, K, id, clock, σᶜᶜᶠ, args...) = κᶜᶜᶠ(i, j, k, grid, closure, K, id, clock) * σᶜᶜᶠ(i, j, k, grid, args...)

#####
##### Viscosity "extractors"
#####

# Number
@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::Number) = ν

@inline κᶠᶜᶜ(i, j, k, grid, clock, κ::Number) = κ
@inline κᶜᶠᶜ(i, j, k, grid, clock, κ::Number) = κ
@inline κᶜᶜᶠ(i, j, k, grid, clock, κ::Number) = κ

# Array / Field
@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ν[i, j, k]
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ℑxyᶠᶠᵃ(i, j, k, grid, ν)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ℑxzᶠᵃᶠ(i, j, k, grid, ν)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::AbstractArray) = @inbounds ℑyzᵃᶠᶠ(i, j, k, grid, ν)

@inline κᶠᶜᶜ(i, j, k, grid, clock, κ::AbstractArray) = ℑxᶠᵃᵃ(i, j, k, grid, κ)
@inline κᶜᶠᶜ(i, j, k, grid, clock, κ::AbstractArray) = ℑyᵃᶠᵃ(i, j, k, grid, κ)
@inline κᶜᶜᶠ(i, j, k, grid, clock, κ::AbstractArray) = ℑzᵃᵃᶠ(i, j, k, grid, κ)

# Function

const c = Center()
const f = Face()

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::F) where F<:Function = ν(node(c, c, c, i, j, k, grid)..., clock.time)
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::F) where F<:Function = ν(node(f, f, c, i, j, k, grid)..., clock.time)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::F) where F<:Function = ν(node(f, c, f, i, j, k, grid)..., clock.time)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::F) where F<:Function = ν(node(c, f, f, i, j, k, grid)..., clock.time)

@inline κᶠᶜᶜ(i, j, k, grid, clock, κ::F) where F<:Function = κ(node(f, c, c, i, j, k, grid)..., clock.time)
@inline κᶜᶠᶜ(i, j, k, grid, clock, κ::F) where F<:Function = κ(node(c, f, c, i, j, k, grid)..., clock.time)
@inline κᶜᶜᶠ(i, j, k, grid, clock, κ::F) where F<:Function = κ(node(c, c, f, i, j, k, grid)..., clock.time)

# "DiscreteDiffusionFunction"
@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, c, c, c)
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, f, f, c)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, f, c, f)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, c, f, f)

@inline κᶠᶜᶜ(i, j, k, grid, clock, κ::DiscreteDiffusionFunction) = κ.func(i, j, k, grid, f, c, c)
@inline κᶜᶠᶜ(i, j, k, grid, clock, κ::DiscreteDiffusionFunction) = κ.func(i, j, k, grid, c, f, c)
@inline κᶜᶜᶠ(i, j, k, grid, clock, κ::DiscreteDiffusionFunction) = κ.func(i, j, k, grid, c, c, f)

