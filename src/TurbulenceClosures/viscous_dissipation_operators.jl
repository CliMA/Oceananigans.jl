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

# @inline νᶜᶜᶜ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Center(), i, grid), ynode(Center(), j, grid), znode(Center(), k, grid), clock.time)
# @inline νᶠᶠᶜ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Face(),   i, grid), ynode(Face(),   j, grid), znode(Center(), k, grid), clock.time)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Face(),   i, grid), ynode(Center(), j, grid), znode(Face(),   k, grid), clock.time)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::F) where F<:Function = ν(xnode(Center(), i, grid), ynode(Face(),   j, grid), znode(Face(),   k, grid), clock.time)

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::F) where F<:Function = ν(i, j, k, grid)
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::F) where F<:Function = ν(i, j, k, grid)

#####
##### Convenience "vanilla" viscous flux functions
#####

@inline viscous_flux_ux(i, j, k, grid, clock, ν, u) = - νᶜᶜᶜ(i, j, k, grid, clock, ν) * ∂xᶜᶜᶜ(i, j, k, grid, u)
@inline viscous_flux_uy(i, j, k, grid, clock, ν, u) = - νᶠᶠᶜ(i, j, k, grid, clock, ν) * ∂yᶠᶠᶜ(i, j, k, grid, u)
@inline viscous_flux_uz(i, j, k, grid, clock, ν, u) = - νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂zᶠᶜᶠ(i, j, k, grid, u)

@inline viscous_flux_vx(i, j, k, grid, clock, ν, v) = - νᶠᶠᶜ(i, j, k, grid, clock, ν) * ∂xᶠᶠᶜ(i, j, k, grid, v)
@inline viscous_flux_vy(i, j, k, grid, clock, ν, v) = - νᶜᶜᶜ(i, j, k, grid, clock, ν) * ∂yᶜᶜᶜ(i, j, k, grid, v)
@inline viscous_flux_vz(i, j, k, grid, clock, ν, v) = - νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂zᶜᶠᶠ(i, j, k, grid, v)

@inline viscous_flux_wx(i, j, k, grid, clock, ν, w) = - νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂xᶠᶜᶠ(i, j, k, grid, w)
@inline viscous_flux_wy(i, j, k, grid, clock, ν, w) = - νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂yᶜᶠᶠ(i, j, k, grid, w)
@inline viscous_flux_wz(i, j, k, grid, clock, ν, w) = - νᶜᶜᶜ(i, j, k, grid, clock, ν) * ∂zᶜᶜᶜ(i, j, k, grid, w)

#####
##### Products of viscosity and divergence, vorticity, vertical momentum gradients
#####

@inline ν_δᶜᶜᶜ(i, j, k, grid, clock, ν, u, v) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * div_xyᶜᶜᶜ(i, j, k, grid, u, v)
@inline ν_ζᶠᶠᶜ(i, j, k, grid, clock, ν, u, v) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)

@inline ν_uzᶠᶜᶠ(i, j, k, grid, clock, ν, u) = νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂zᶠᶜᶠ(i, j, k, grid, u)
@inline ν_vzᶜᶠᶠ(i, j, k, grid, clock, ν, v) = νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂zᶜᶠᶠ(i, j, k, grid, v)

@inline ν_uzzzᶠᶜᶠ(i, j, k, grid, clock, ν, u) = νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂³zᶠᶜᶠ(i, j, k, grid, u)
@inline ν_vzzzᶜᶠᶠ(i, j, k, grid, clock, ν, v) = νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂³zᶜᶠᶠ(i, j, k, grid, v)

# See https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-dissipation
@inline function δ★ᶜᶜᶜ(i, j, k, grid, u, v)

    # These closures seem to be needed to help the compiler infer types
    # (either of u and v or of the function arguments)
    @inline Δy_∇²u(i, j, k, grid, u) = Δy_qᶠᶜᶜ(i, j, k, grid, ∇²hᶠᶜᶜ, u)
    @inline Δx_∇²v(i, j, k, grid, v) = Δx_qᶜᶠᶜ(i, j, k, grid, ∇²hᶜᶠᶜ, v)

    return 1 / Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, Δy_∇²u, u) +
                                       δyᵃᶜᵃ(i, j, k, Δx_∇²v, v))
end

@inline function ζ★ᶠᶠᶜ(i, j, k, grid, u, v)

    # These closures seem to be needed to help the compiler infer types
    # (either of u and v or of the function arguments)
    @inline Δy_∇²v(i, j, k, grid, v) = Δy_qᶜᶠᶜ(i, j, k, grid, ∇²hᶜᶠᶜ, v)
    @inline Δx_∇²u(i, j, k, grid, u) = Δx_qᶠᶜᶜ(i, j, k, grid, ∇²hᶠᶜᶜ, u)

    return 1 / Azᶠᶠᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, Δy_∇²v, v) -
                                       δyᵃᶠᵃ(i, j, k, Δx_∇²u, u))
end

@inline ν_δ★ᶜᶜᶜ(i, j, k, grid, clock, ν, u, v) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * δ★ᶜᶜᶜ(i, j, k, grid, u, v)
@inline ν_ζ★ᶠᶠᶜ(i, j, k, grid, clock, ν, u, v) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * ζ★ᶠᶠᶜ(i, j, k, grid, u, v)
