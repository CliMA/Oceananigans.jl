using Oceananigans.Operators: Δy_qᶠᶜᶜ, Δx_qᶜᶠᶜ, Δx_qᶠᶜᶜ, ℑxyᶠᶠᵃ, ℑxzᶠᵃᶠ, ℑyzᵃᶠᶠ, div

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
    return 1 / Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_qᶜᶜᶜ, _viscous_flux_ux, disc, closure, args...) +
                                      δyᵃᶜᵃ(i, j, k, grid, Ay_qᶠᶠᶜ, _viscous_flux_uy, disc, closure, args...) +
                                      δzᵃᵃᶜ(i, j, k, grid, Az_qᶠᶜᶠ, _viscous_flux_uz, disc, closure, args...))
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1 / Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶠᶜ, _viscous_flux_vx, disc, closure, args...) +
                                      δyᵃᶠᵃ(i, j, k, grid, Ay_qᶜᶜᶜ, _viscous_flux_vy, disc, closure, args...) +
                                      δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶠᶠ, _viscous_flux_vz, disc, closure, args...))
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1 / Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶠ, _viscous_flux_wx, disc, closure, args...) +
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

@inline ν_σᶜᶜᶜ(i, j, k, grid, closure, K, clock, F, σᶜᶜᶜ, args...) = νᶜᶜᶜ(i, j, k, grid, closure, K, clock, F) * σᶜᶜᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶠᶜ(i, j, k, grid, closure, K, clock, F, σᶠᶠᶜ, args...) = νᶠᶠᶜ(i, j, k, grid, closure, K, clock, F) * σᶠᶠᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, F, σᶠᶜᶠ, args...) = νᶠᶜᶠ(i, j, k, grid, closure, K, clock, F) * σᶠᶜᶠ(i, j, k, grid, args...)
@inline ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, F, σᶜᶠᶠ, args...) = νᶜᶠᶠ(i, j, k, grid, closure, K, clock, F) * σᶜᶠᶠ(i, j, k, grid, args...)

@inline ν_δᶜᶜᶜ(i, j, k, grid, closure, K, clock, F, u, v) = νᶜᶜᶜ(i, j, k, grid, closure, K, clock, F) * div_xyᶜᶜᶜ(i, j, k, grid, u, v)
@inline ν_ζᶠᶠᶜ(i, j, k, grid, closure, K, clock, F, u, v) = νᶠᶠᶜ(i, j, k, grid, closure, K, clock, F) * ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)

@inline κ_σᶠᶜᶜ(i, j, k, grid, closure, K, id, clock, F, σᶠᶜᶜ, args...) = κᶠᶜᶜ(i, j, k, grid, closure, K, id, clock, F) * σᶠᶜᶜ(i, j, k, grid, args...)
@inline κ_σᶜᶠᶜ(i, j, k, grid, closure, K, id, clock, F, σᶜᶠᶜ, args...) = κᶜᶠᶜ(i, j, k, grid, closure, K, id, clock, F) * σᶜᶠᶜ(i, j, k, grid, args...)
@inline κ_σᶜᶜᶠ(i, j, k, grid, closure, K, id, clock, F, σᶜᶜᶠ, args...) = κᶜᶜᶠ(i, j, k, grid, closure, K, id, clock, F) * σᶜᶜᶠ(i, j, k, grid, args...)

#####
##### Viscosity "extractors"
#####

# Number

@inline νᶜᶜᶜ(i, j, k, grid, loc, ν::Number, args...) = ν
@inline νᶠᶜᶠ(i, j, k, grid, loc, ν::Number, args...) = ν
@inline νᶜᶠᶠ(i, j, k, grid, loc, ν::Number, args...) = ν
@inline νᶠᶠᶜ(i, j, k, grid, loc, ν::Number, args...) = ν

@inline κᶠᶜᶜ(i, j, k, grid, loc, κ::Number, args...) = κ
@inline κᶜᶠᶜ(i, j, k, grid, loc, κ::Number, args...) = κ
@inline κᶜᶜᶠ(i, j, k, grid, loc, κ::Number, args...) = κ

# Array / Field at `Center, Center, Center`
const Lᶜᶜᶜ = Tuple{Center, Center, Center}
@inline νᶜᶜᶜ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, args...) = @inbounds ν[i, j, k]
@inline νᶠᶜᶠ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, args...) = ℑxzᶠᵃᶠ(i, j, k, grid, ν)
@inline νᶜᶠᶠ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, args...) = ℑyzᵃᶠᶠ(i, j, k, grid, ν)
@inline νᶠᶠᶜ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, args...) = ℑxyᶠᶠᵃ(i, j, k, grid, ν)
                                        
@inline κᶠᶜᶜ(i, j, k, grid, ::Lᶜᶜᶜ, κ::AbstractArray, args...) = ℑxᶠᵃᵃ(i, j, k, grid, κ)
@inline κᶜᶠᶜ(i, j, k, grid, ::Lᶜᶜᶜ, κ::AbstractArray, args...) = ℑyᵃᶠᵃ(i, j, k, grid, κ)
@inline κᶜᶜᶠ(i, j, k, grid, ::Lᶜᶜᶜ, κ::AbstractArray, args...) = ℑzᵃᵃᶠ(i, j, k, grid, κ)

# Array / Field at `Center, Center, Face`
const Lᶜᶜᶠ = Tuple{Center, Center, Face}
@inline νᶜᶜᶜ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, args...) = ℑzᵃᵃᶜ(i, j, k, grid, ν)
@inline νᶠᶜᶠ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, args...) = ℑxᶠᵃᵃ(i, j, k, grid, ν)
@inline νᶜᶠᶠ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, args...) = ℑyᵃᶠᵃ(i, j, k, grid, ν)
@inline νᶠᶠᶜ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, args...) = ℑxyzᶠᶠᶜ(i, j, k, grid, ν)

@inline κᶠᶜᶜ(i, j, k, grid, ::Lᶜᶜᶠ, κ::AbstractArray, args...) = ℑxzᶠᵃᶠ(i, j, k, grid, κ)
@inline κᶜᶠᶜ(i, j, k, grid, ::Lᶜᶜᶠ, κ::AbstractArray, args...) = ℑyzᵃᶠᶠ(i, j, k, grid, κ)
@inline κᶜᶜᶠ(i, j, k, grid, ::Lᶜᶜᶠ, κ::AbstractArray, args...) = @inbounds κ[i, j, k]

# Function

const c = Center()
const f = Face()

@inline νᶜᶜᶜ(i, j, k, grid, loc, ν::F, clock, args...) where F<:Function = ν(node(c, c, c, i, j, k, grid)..., clock.time)
@inline νᶠᶜᶠ(i, j, k, grid, loc, ν::F, clock, args...) where F<:Function = ν(node(f, c, f, i, j, k, grid)..., clock.time)
@inline νᶜᶠᶠ(i, j, k, grid, loc, ν::F, clock, args...) where F<:Function = ν(node(c, f, f, i, j, k, grid)..., clock.time)
@inline νᶠᶠᶜ(i, j, k, grid, loc, ν::F, clock, args...) where F<:Function = ν(node(f, f, c, i, j, k, grid)..., clock.time)

@inline κᶠᶜᶜ(i, j, k, grid, loc, κ::F, clock, args...) where F<:Function = κ(node(f, c, c, i, j, k, grid)..., clock.time)
@inline κᶜᶠᶜ(i, j, k, grid, loc, κ::F, clock, args...) where F<:Function = κ(node(c, f, c, i, j, k, grid)..., clock.time)
@inline κᶜᶜᶠ(i, j, k, grid, loc, κ::F, clock, args...) where F<:Function = κ(node(c, c, f, i, j, k, grid)..., clock.time)

# "DiscreteDiffusionFunction"
@inline νᶜᶜᶜ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = ν.func(i, j, k, grid, c, c, c, fields)
@inline νᶠᶜᶠ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = ν.func(i, j, k, grid, f, c, f, fields)
@inline νᶜᶠᶠ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = ν.func(i, j, k, grid, c, f, f, fields)
@inline νᶠᶠᶜ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = ν.func(i, j, k, grid, f, f, c, fields)

@inline κᶠᶜᶜ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = κ.func(i, j, k, grid, f, c, c, fields)
@inline κᶜᶠᶜ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = κ.func(i, j, k, grid, c, f, c, fields)
@inline κᶜᶜᶠ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = κ.func(i, j, k, grid, c, c, f, fields)

#####
##### Immersed flux divergences
#####

@inline immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid, args...)   = zero(eltype(grid))
@inline immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, args...)   = zero(eltype(grid))
@inline immersed_∂ⱼ_τ₃ⱼ(i, j, k, grid, args...)   = zero(eltype(grid))
@inline immersed_∇_dot_qᶜ(i, j, k, grid, args...) = zero(eltype(grid))
