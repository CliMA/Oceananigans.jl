using Oceananigans.Operators: Δy_qᶠᶜᶜ, Δx_qᶜᶠᶜ, Δx_qᶠᶜᶜ, ℑxyᶠᶠᶜ, ℑxzᶠᶜᶠ, ℑyzᶜᶠᶠ, div

#####                                                            
##### Viscous flux divergences
#####

@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1 / Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᶜᶜ(i, j, k, grid, Ax_qᶜᶜᶜ, viscous_flux_ux, disc, closure, args...) +
                                      δyᶠᶜᶜ(i, j, k, grid, Ay_qᶠᶠᶜ, viscous_flux_uy, disc, closure, args...) +
                                      δzᶠᶜᶜ(i, j, k, grid, Az_qᶠᶜᶠ, viscous_flux_uz, disc, closure, args...))
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1 / Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᶠᶜ(i, j, k, grid, Ax_qᶠᶠᶜ, viscous_flux_vx, disc, closure, args...) +
                                      δyᶜᶠᶜ(i, j, k, grid, Ay_qᶜᶜᶜ, viscous_flux_vy, disc, closure, args...) +
                                      δzᶜᶠᶜ(i, j, k, grid, Az_qᶜᶠᶠ, viscous_flux_vz, disc, closure, args...))
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1 / Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᶜᶠ(i, j, k, grid, Ax_qᶠᶜᶠ, viscous_flux_wx, disc, closure, args...) +
                                      δyᶜᶜᶠ(i, j, k, grid, Ay_qᶜᶠᶠ, viscous_flux_wy, disc, closure, args...) +
                                      δzᶜᶜᶠ(i, j, k, grid, Az_qᶜᶜᶜ, viscous_flux_wz, disc, closure, args...))
end

@inline function ∇_dot_qᶜ(i, j, k, grid, closure::AbstractTurbulenceClosure, tracer_index, args...)
    disc = time_discretization(closure)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, diffusive_flux_x, disc, closure, tracer_index, args...) +
                                    δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, diffusive_flux_y, disc, closure, tracer_index, args...) +
                                    δzᶜᶜᶜ(i, j, k, grid, Az_qᶜᶜᶠ, diffusive_flux_z, disc, closure, tracer_index, args...))
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
@inline νᶜᶜᶜ(i, j, k, grid, clock, loc, ν::Number) = ν
@inline νᶠᶜᶠ(i, j, k, grid, clock, loc, ν::Number) = ν
@inline νᶜᶠᶠ(i, j, k, grid, clock, loc, ν::Number) = ν
@inline νᶠᶠᶜ(i, j, k, grid, clock, loc, ν::Number) = ν

@inline κᶠᶜᶜ(i, j, k, grid, clock, loc, κ::Number) = κ
@inline κᶜᶠᶜ(i, j, k, grid, clock, loc, κ::Number) = κ
@inline κᶜᶜᶠ(i, j, k, grid, clock, loc, κ::Number) = κ

# Array / Field at `Center, Center, Center`
const Lᶜᶜᶜ = Tuple{Center, Center, Center}
@inline νᶜᶜᶜ(i, j, k, grid, clock, ::Lᶜᶜᶜ, ν::AbstractArray) = @inbounds ν[i, j, k]
@inline νᶠᶜᶠ(i, j, k, grid, clock, ::Lᶜᶜᶜ, ν::AbstractArray) = ℑxzᶠᶜᶠ(i, j, k, grid, ν)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ::Lᶜᶜᶜ, ν::AbstractArray) = ℑyzᶜᶠᶠ(i, j, k, grid, ν)
@inline νᶠᶠᶜ(i, j, k, grid, clock, ::Lᶜᶜᶜ, ν::AbstractArray) = ℑxyᶠᶠᶜ(i, j, k, grid, ν)
                                        
@inline κᶠᶜᶜ(i, j, k, grid, clock, ::Lᶜᶜᶜ, κ::AbstractArray) = ℑxᶠᶜᶜ(i, j, k, grid, κ)
@inline κᶜᶠᶜ(i, j, k, grid, clock, ::Lᶜᶜᶜ, κ::AbstractArray) = ℑyᶜᶠᶜ(i, j, k, grid, κ)
@inline κᶜᶜᶠ(i, j, k, grid, clock, ::Lᶜᶜᶜ, κ::AbstractArray) = ℑzᶜᶜᶠ(i, j, k, grid, κ)

# Array / Field at `Center, Center, Face`
const Lᶜᶜᶠ = Tuple{Center, Center, Face}
@inline νᶜᶜᶜ(i, j, k, grid, clock, ::Lᶜᶜᶠ, ν::AbstractArray) = ℑzᶜᶜᶜ(i, j, k, grid, ν)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ::Lᶜᶜᶠ, ν::AbstractArray) = ℑxᶠᶜᶜ(i, j, k, grid, ν)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ::Lᶜᶜᶠ, ν::AbstractArray) = ℑyᶜᶠᶜ(i, j, k, grid, ν)
@inline νᶠᶠᶜ(i, j, k, grid, clock, ::Lᶜᶜᶠ, ν::AbstractArray) = ℑxyzᶠᶠᶜ(i, j, k, grid, ν)

@inline κᶠᶜᶜ(i, j, k, grid, clock, ::Lᶜᶜᶠ, κ::AbstractArray) = ℑxzᶠᶜᶠ(i, j, k, grid, κ)
@inline κᶜᶠᶜ(i, j, k, grid, clock, ::Lᶜᶜᶠ, κ::AbstractArray) = ℑyzᶜᶠᶠ(i, j, k, grid, κ)
@inline κᶜᶜᶠ(i, j, k, grid, clock, ::Lᶜᶜᶠ, κ::AbstractArray) = @inbounds κ[i, j, k]

# Function

const c = Center()
const f = Face()

@inline νᶜᶜᶜ(i, j, k, grid, clock, loc, ν::F) where F<:Function = ν(node(c, c, c, i, j, k, grid)..., clock.time)
@inline νᶠᶜᶠ(i, j, k, grid, clock, loc, ν::F) where F<:Function = ν(node(f, c, f, i, j, k, grid)..., clock.time)
@inline νᶜᶠᶠ(i, j, k, grid, clock, loc, ν::F) where F<:Function = ν(node(c, f, f, i, j, k, grid)..., clock.time)
@inline νᶠᶠᶜ(i, j, k, grid, clock, loc, ν::F) where F<:Function = ν(node(f, f, c, i, j, k, grid)..., clock.time)

@inline κᶠᶜᶜ(i, j, k, grid, clock, loc, κ::F) where F<:Function = κ(node(f, c, c, i, j, k, grid)..., clock.time)
@inline κᶜᶠᶜ(i, j, k, grid, clock, loc, κ::F) where F<:Function = κ(node(c, f, c, i, j, k, grid)..., clock.time)
@inline κᶜᶜᶠ(i, j, k, grid, clock, loc, κ::F) where F<:Function = κ(node(c, c, f, i, j, k, grid)..., clock.time)

# "DiscreteDiffusionFunction"
@inline νᶜᶜᶜ(i, j, k, grid, clock, loc, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, c, c, c)
@inline νᶠᶜᶠ(i, j, k, grid, clock, loc, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, f, c, f)
@inline νᶜᶠᶠ(i, j, k, grid, clock, loc, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, c, f, f)
@inline νᶠᶠᶜ(i, j, k, grid, clock, loc, ν::DiscreteDiffusionFunction) = ν.func(i, j, k, grid, f, f, c)

@inline κᶠᶜᶜ(i, j, k, grid, clock, loc, κ::DiscreteDiffusionFunction) = κ.func(i, j, k, grid, f, c, c)
@inline κᶜᶠᶜ(i, j, k, grid, clock, loc, κ::DiscreteDiffusionFunction) = κ.func(i, j, k, grid, c, f, c)
@inline κᶜᶜᶠ(i, j, k, grid, clock, loc, κ::DiscreteDiffusionFunction) = κ.func(i, j, k, grid, c, c, f)

#####
##### Immersed flux divergences
#####

@inline immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid, args...) = zero(eltype(grid))
@inline immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, args...) = zero(eltype(grid))
@inline immersed_∂ⱼ_τ₃ⱼ(i, j, k, grid, args...) = zero(eltype(grid))
@inline immersed_∇_dot_qᶜ(i, j, k, grid, args...) = zero(eltype(grid))
