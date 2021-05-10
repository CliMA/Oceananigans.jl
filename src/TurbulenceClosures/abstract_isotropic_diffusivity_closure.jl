"""
    abstract type AbstractIsotropicDiffusivity <: AbstractTurbulenceClosure end

Abstract type for closures with *isotropic* diffusivities.
"""
abstract type AbstractIsotropicDiffusivity{TD} <: AbstractTurbulenceClosure{TD} end

const AID = AbstractIsotropicDiffusivity

"""
    viscosity(closure, diffusivities)

Returns the isotropic viscosity associated with `closure`.
"""
function viscosity end 

"""
    diffusivity(closure, diffusivities, ::Val{c_idx}) where c_idx

Returns the isotropic diffusivity associated with `closure` and tracer index `c_idx`.
"""
function diffusivity end 

#####
##### Viscosity-stress products
#####

"""
    ν_σᶜᶜᶜ(i, j, k, grid, ν, σᶜᶜᶜ, u, v, w)

Multiply the array `ν` located at `ᶜᶜᶜ` by a function

    `σᶜᶜᶜ(i, j, k, grid, u, v, w)`

at index `i, j, k` and location `ᶜᶜᶜ`.
"""
@inline ν_σᶜᶜᶜ(i, j, k, grid, clock, ν, σᶜᶜᶜ, u, v, w) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * σᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline ν_σᶠᶠᶜ(i, j, k, grid, clock, ν, σᶠᶠᶜ, u, v, w) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * σᶠᶠᶜ(i, j, k, grid, u, v, w)
@inline ν_σᶠᶜᶠ(i, j, k, grid, clock, ν, σᶠᶜᶠ, u, v, w) = νᶠᶜᶠ(i, j, k, grid, clock, ν) * σᶠᶜᶠ(i, j, k, grid, u, v, w)
@inline ν_σᶜᶠᶠ(i, j, k, grid, clock, ν, σᶜᶠᶠ, u, v, w) = νᶜᶠᶠ(i, j, k, grid, clock, ν) * σᶜᶠᶠ(i, j, k, grid, u, v, w)

#####
##### Stress divergences
#####

@inline viscous_flux_ux(i, j, k, grid, clock, closure::AID, U, K) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, K), Σ₁₁, U.u, U.v, U.w)
@inline viscous_flux_uy(i, j, k, grid, clock, closure::AID, U, K) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, K), Σ₁₂, U.u, U.v, U.w)
@inline viscous_flux_uz(i, j, k, grid, clock, closure::AID, U, K) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, K), Σ₁₃, U.u, U.v, U.w)

@inline viscous_flux_vx(i, j, k, grid, clock, closure::AID, U, K) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, K), Σ₂₁, U.u, U.v, U.w)
@inline viscous_flux_vy(i, j, k, grid, clock, closure::AID, U, K) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, K), Σ₂₂, U.u, U.v, U.w)
@inline viscous_flux_vz(i, j, k, grid, clock, closure::AID, U, K) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, K), Σ₂₃, U.u, U.v, U.w)

@inline viscous_flux_wx(i, j, k, grid, clock, closure::AID, U, K) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, K), Σ₃₁, U.u, U.v, U.w)
@inline viscous_flux_wy(i, j, k, grid, clock, closure::AID, U, K) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, K), Σ₃₂, U.u, U.v, U.w)
@inline viscous_flux_wz(i, j, k, grid, clock, closure::AID, U, K) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, K), Σ₃₃, U.u, U.v, U.w)

#####
##### Diffusive fluxes
#####

@inline diffusive_flux_x(i, j, k, grid, clock, closure::AID, c, ::Val{c_idx}, K, args...) where c_idx = diffusive_flux_x(i, j, k, grid, clock, diffusivity(closure, K, Val(c_idx)), c)
@inline diffusive_flux_y(i, j, k, grid, clock, closure::AID, c, ::Val{c_idx}, K, args...) where c_idx = diffusive_flux_y(i, j, k, grid, clock, diffusivity(closure, K, Val(c_idx)), c)
@inline diffusive_flux_z(i, j, k, grid, clock, closure::AID, c, ::Val{c_idx}, K, args...) where c_idx = diffusive_flux_z(i, j, k, grid, clock, diffusivity(closure, K, Val(c_idx)), c)

#####
##### Support for VerticallyImplicitTimeDiscretization
#####

const VITD = VerticallyImplicitTimeDiscretization
const VerticallyBoundedGrid{FT} = AbstractGrid{FT, <:Any, <:Any, <:Bounded}
const AG = AbstractGrid

@inline z_viscosity(closure::AID, diffusivities) = viscosity(closure, diffusivities)
@inline z_diffusivity(closure::AID, diffusivities, ::Val{c_idx}) where c_idx = diffusivity(closure, diffusivities, Val(c_idx))

@inline ivd_viscous_flux_uz(i, j, k, grid, clock, closure, U, K) = - ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, K), ∂xᶠᶜᵃ, U.w)
@inline ivd_viscous_flux_vz(i, j, k, grid, clock, closure, U, K) = - ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, K), ∂yᶜᶠᵃ, U.v)

# General functions (eg for vertically periodic)
@inline viscous_flux_uz(i, j, k, grid::AG, ::VITD, clock, closure::AID, U, K) = ivd_viscous_flux_uz(i, j, k, grid, clock, closure, U, K)
@inline viscous_flux_vz(i, j, k, grid::AG, ::VITD, clock, closure::AID, U, K) = ivd_viscous_flux_vz(i, j, k, grid, clock, closure, U, K)
@inline viscous_flux_wz(i, j, k, grid::AG{FT}, ::VITD, clock, closure::AID, U, K) where FT = zero(FT)
@inline diffusive_flux_z(i, j, k, grid::AG{FT}, ::VITD, clock, closure::AID, args...) where FT = zero(FT)
                  
# Vertically bounded grids
@inline function viscous_flux_uz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, clock, closure::AID, U, K)
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_uz(i, j, k, grid, ExplicitTimeDiscretization(), clock, closure, U, K), # on boundaries, calculate fluxes explicitly
                  ivd_viscous_flux_uz(i, j, k, grid, clock, closure, U, K))
end

@inline function viscous_flux_vz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, clock, closure::AID, U, K)
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_vz(i, j, k, grid, ExplicitTimeDiscretization(), clock, closure, U, K), # on boundaries, calculate fluxes explicitly
                  ivd_viscous_flux_vz(i, j, k, grid, clock, closure, U, K))
end

@inline function viscous_flux_wz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, clock, closure::AID, U, K) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_wz(i, j, k, grid, ExplicitTimeDiscretization(), clock, closure, U, K), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, clock, closure::AID, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), clock, closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end
