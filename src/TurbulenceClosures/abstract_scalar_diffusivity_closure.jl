"""
    abstract type AbstractScalarDiffusivity <: AbstractTurbulenceClosure end

Abstract type for closures with *isotropic* diffusivities.
"""
abstract type AbstractScalarDiffusivity{TD, F} <: AbstractTurbulenceClosure{TD} end

"""
    struct ThreeDimensional end

Specifies a three-dimensionally-isotropic `ScalarDiffusivity`.
"""
struct ThreeDimensionalFormulation end

"""
    struct Horizontal end

Specifies a horizontally-isotropic, `VectorInvariant`, `ScalarDiffusivity`.
"""
struct HorizontalFormulation end

"""
    struct Vertical end

Specifies a `ScalarDiffusivity` acting only in the vertical direction.
"""
struct VerticalFormulation end

"""
    viscosity(closure, diffusivities)

Returns the scalar viscosity associated with `closure`.
"""
function viscosity end 

"""
    diffusivity(closure, tracer_index, diffusivity_fields)

Returns the scalar diffusivity associated with `closure` and `tracer_index`.
"""
function diffusivity end 

@inline formulation(::AbstractScalarDiffusivity{TD, F}) where {TD, F} = F()

Base.summary(::VerticalFormulation) = "VerticalFormulation"
Base.summary(::HorizontalFormulation) = "HorizontalFormulation"
Base.summary(::ThreeDimensionalFormulation) = "ThreeDimensionalFormulation"

#####
##### Coefficient extractors
#####

@inline νᶜᶜᶜ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, clock) = νᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, K)) 
@inline νᶠᶠᶜ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, clock) = νᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, K))
@inline νᶠᶜᶠ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, clock) = νᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, K))
@inline νᶜᶠᶠ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, clock) = νᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, K))

@inline κᶠᶜᶜ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, id, clock) = κᶠᶜᶜ(i, j, k, grid, clock, diffusivity(closure, K, id))
@inline κᶜᶠᶜ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, id, clock) = κᶜᶠᶜ(i, j, k, grid, clock, diffusivity(closure, K, id))
@inline κᶜᶜᶠ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, id, clock) = κᶜᶜᶠ(i, j, k, grid, clock, diffusivity(closure, K, id))

@inline νzᶜᶜᶜ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, clock) = νᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, K)) 
@inline νzᶠᶠᶜ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, clock) = νᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, K))
@inline νzᶠᶜᶠ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, clock) = νᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, K))
@inline νzᶜᶠᶠ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, clock) = νᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, K))

@inline κzᶠᶜᶜ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, id, clock) = κᶠᶜᶜ(i, j, k, grid, clock, diffusivity(closure, K, id))
@inline κzᶜᶠᶜ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, id, clock) = κᶜᶠᶜ(i, j, k, grid, clock, diffusivity(closure, K, id))
@inline κzᶜᶜᶠ(i, j, k, grid, closure::AbstractScalarDiffusivity, K, id, clock) = κᶜᶜᶠ(i, j, k, grid, clock, diffusivity(closure, K, id))

#####
##### Stress divergences
#####

const AID = AbstractScalarDiffusivity{<:Any, <:ThreeDimensionalFormulation}
const AHD = AbstractScalarDiffusivity{<:Any, <:HorizontalFormulation}
const AVD = AbstractScalarDiffusivity{<:Any, <:VerticalFormulation}

@inline viscous_flux_ux(i, j, k, grid, closure::AID, K, U, C, clock, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, closure, K, clock, Σ₁₁, U.u, U.v, U.w)
@inline viscous_flux_vx(i, j, k, grid, closure::AID, K, U, C, clock, b) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, closure, K, clock, Σ₂₁, U.u, U.v, U.w)
@inline viscous_flux_wx(i, j, k, grid, closure::AID, K, U, C, clock, b) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, Σ₃₁, U.u, U.v, U.w)
@inline viscous_flux_uy(i, j, k, grid, closure::AID, K, U, C, clock, b) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, closure, K, clock, Σ₁₂, U.u, U.v, U.w)
@inline viscous_flux_vy(i, j, k, grid, closure::AID, K, U, C, clock, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, closure, K, clock, Σ₂₂, U.u, U.v, U.w)
@inline viscous_flux_wy(i, j, k, grid, closure::AID, K, U, C, clock, b) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, Σ₃₂, U.u, U.v, U.w)

@inline viscous_flux_ux(i, j, k, grid, closure::AHD, K, U, C, clock, b) = - ν_δᶜᶜᶜ(i, j, k, grid, closure, K, clock, U.u, U.v)
@inline viscous_flux_vx(i, j, k, grid, closure::AHD, K, U, C, clock, b) = - ν_ζᶠᶠᶜ(i, j, k, grid, closure, K, clock, U.u, U.v)
@inline viscous_flux_uy(i, j, k, grid, closure::AHD, K, U, C, clock, b) = + ν_ζᶠᶠᶜ(i, j, k, grid, closure, K, clock, U.u, U.v)   
@inline viscous_flux_vy(i, j, k, grid, closure::AHD, K, U, C, clock, b) = - ν_δᶜᶜᶜ(i, j, k, grid, closure, K, clock, U.u, U.v)
@inline viscous_flux_wx(i, j, k, grid, closure::AHD, K, U, C, clock, b) = - ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, ∂xᶠᶜᶠ, U.w)
@inline viscous_flux_wy(i, j, k, grid, closure::AHD, K, U, C, clock, b) = - ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, ∂yᶜᶠᶠ, U.w)

@inline viscous_flux_uz(i, j, k, grid, closure::AID, K, U, C, clock, b) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, Σ₁₃, U.u, U.v, U.w)
@inline viscous_flux_vz(i, j, k, grid, closure::AID, K, U, C, clock, b) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, Σ₂₃, U.u, U.v, U.w)
@inline viscous_flux_wz(i, j, k, grid, closure::AID, K, U, C, clock, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, closure, K, clock, Σ₃₃, U.u, U.v, U.w)

@inline viscous_flux_uz(i, j, k, grid, closure::AVD, K, U, C, clock, b) = - ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, ∂zᶠᶜᶠ, U.u)
@inline viscous_flux_vz(i, j, k, grid, closure::AVD, K, U, C, clock, b) = - ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, ∂zᶜᶠᶠ, U.v)
@inline viscous_flux_wz(i, j, k, grid, closure::AVD, K, U, C, clock, b) = - ν_σᶜᶜᶜ(i, j, k, grid, closure, K, clock, ∂zᶜᶜᶜ, U.w)

#####
##### Diffusive fluxes
#####

const AIDorAHD = Union{AID, AHD}
const AIDorAVD = Union{AID, AVD}

@inline diffusive_flux_x(i, j, k, grid, cl::AIDorAHD, K, ::Val{id}, U, C, clk, b) where id = - κᶠᶜᶜ(i, j, k, grid, cl, K, Val(id), clk) * ∂xᶠᶜᶜ(i, j, k, grid, C[id])
@inline diffusive_flux_y(i, j, k, grid, cl::AIDorAHD, K, ::Val{id}, U, C, clk, b) where id = - κᶜᶠᶜ(i, j, k, grid, cl, K, Val(id), clk) * ∂yᶜᶠᶜ(i, j, k, grid, C[id])
@inline diffusive_flux_z(i, j, k, grid, cl::AIDorAVD, K, ::Val{id}, U, C, clk, b) where id = - κᶜᶜᶠ(i, j, k, grid, cl, K, Val(id), clk) * ∂zᶜᶜᶠ(i, j, k, grid, C[id])

#####
##### Zero out not used fluxes
#####

for (dir, closure) in zip((:x, :y, :z), (:AVD, :AVD, :AHD))
    diffusive_flux = Symbol(:diffusive_flux_, dir)
    viscous_flux_u = Symbol(:viscous_flux_u, dir)
    viscous_flux_v = Symbol(:viscous_flux_v, dir)
    viscous_flux_w = Symbol(:viscous_flux_w, dir)
    @eval begin
        @inline $diffusive_flux(i, j, k, grid, closure::$closure, K, ::Val, args...) = zero(eltype(grid))
        @inline $viscous_flux_u(i, j, k, grid, closure::$closure, args...) = zero(eltype(grid))
        @inline $viscous_flux_v(i, j, k, grid, closure::$closure, args...) = zero(eltype(grid))
        @inline $viscous_flux_w(i, j, k, grid, closure::$closure, args...) = zero(eltype(grid))
    end
end

#####
##### Support for VerticallyImplicit
#####

const VITD = VerticallyImplicitTimeDiscretization

@inline ivd_viscous_flux_uz(i, j, k, grid, closure, K, U, C, clock, b) = - ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, ∂xᶠᶜᶠ, U.w)
@inline ivd_viscous_flux_vz(i, j, k, grid, closure, K, U, C, clock, b) = - ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, ∂yᶜᶠᶠ, U.w)

# General functions (eg for vertically periodic)
@inline viscous_flux_uz(i, j, k, grid,  ::VITD, closure::AIDorAVD, args...) = ivd_viscous_flux_uz(i, j, k, grid, closure, args...)
@inline viscous_flux_vz(i, j, k, grid,  ::VITD, closure::AIDorAVD, args...) = ivd_viscous_flux_vz(i, j, k, grid, closure, args...)
@inline viscous_flux_wz(i, j, k, grid,  ::VITD, closure::AIDorAVD, args...) = zero(eltype(grid))
@inline diffusive_flux_z(i, j, k, grid, ::VITD, closure::AIDorAVD, args...) = zero(eltype(grid))
                  
# Vertically bounded grids
#
# For vertically bounded grids, we calculate _explicit_ fluxes on the boundaries, 
# and elide the implicit vertical flux component everywhere else. This is consistent
# with the formulation of the tridiagonal solver, which requires explicit treatment
# of boundary contributions (eg boundary contributions must be moved to the right
# hand side of the tridiagonal system).

@inline function viscous_flux_uz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, args...)
    return ifelse((k == 1) | (k == grid.Nz+1), 
                  viscous_flux_uz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  ivd_viscous_flux_uz(i, j, k, grid, closure, args...))
end

@inline function viscous_flux_vz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, args...)
    return ifelse((k == 1) | (k == grid.Nz+1), 
                  viscous_flux_vz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  ivd_viscous_flux_vz(i, j, k, grid, closure, args...))
end

@inline function viscous_flux_wz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, args...)
    return ifelse((k == 1) | (k == grid.Nz+1), 
                  viscous_flux_wz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  zero(eltype(grid)))
end

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, args...)
    return ifelse((k == 1) | (k == grid.Nz+1), 
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  zero(eltype(grid)))
end

