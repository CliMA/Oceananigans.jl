"""
    abstract type AbstractScalarDiffusivity <: AbstractTurbulenceClosure end

Abstract type for closures with *isotropic* diffusivities.
"""
abstract type AbstractScalarBiharmonicDiffusivity{Iso} <: AbstractTurbulenceClosure{Explicit} end

@inline isotropy(::AbstractScalarBiharmonicDiffusivity{Iso}) where {Iso} = Iso()

#####
##### Stress divergences
#####

const AIBD = AbstractScalarBiharmonicDiffusivity{<:ThreeDimensional}
const AHBD = AbstractScalarBiharmonicDiffusivity{<:Horizontal}
const AVBD = AbstractScalarBiharmonicDiffusivity{<:Vertical}

@inline viscous_flux_ux(i, j, k, grid, closure::AIBD, clock, U, args...) = ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), ∂xᶜᶜᶜ, ∇²ᶠᶜᶜ, U.u)
@inline viscous_flux_vx(i, j, k, grid, closure::AIBD, clock, U, args...) = ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, args...), ∂xᶠᶠᶜ, ∇²ᶜᶠᶜ, U.v)
@inline viscous_flux_uy(i, j, k, grid, closure::AIBD, clock, U, args...) = ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, args...), ∂yᶠᶠᶜ, ∇²ᶠᶜᶜ, U.u)
@inline viscous_flux_vy(i, j, k, grid, closure::AIBD, clock, U, args...) = ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), ∂yᶜᶜᶜ, ∇²ᶜᶠᶜ, U.v)

@inline viscous_flux_ux(i, j, k, grid, closure::AHBD, clock, U, args...) = + ν_δ★ᶜᶜᶜ(i, j, k, grid, clock, closure.ν, U.u, U.v)   
@inline viscous_flux_vx(i, j, k, grid, closure::AHBD, clock, U, args...) = + ν_ζ★ᶠᶠᶜ(i, j, k, grid, clock, closure.ν, U.u, U.v)
@inline viscous_flux_uy(i, j, k, grid, closure::AHBD, clock, U, args...) = - ν_ζ★ᶠᶠᶜ(i, j, k, grid, clock, closure.ν, U.u, U.v)   
@inline viscous_flux_vy(i, j, k, grid, closure::AHBD, clock, U, args...) = + ν_δ★ᶜᶜᶜ(i, j, k, grid, clock, closure.ν, U.u, U.v)

@inline viscous_flux_uz(i, j, k, grid, closure::AIBD, clock, U, args...) = ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, args...), ∂zᶠᶜᶠ, ∇²ᶠᶜᶜ, U.u)
@inline viscous_flux_vz(i, j, k, grid, closure::AIBD, clock, U, args...) = ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, args...), ∂zᶜᶠᶠ, ∇²ᶜᶠᶜ, U.v)
@inline viscous_flux_wz(i, j, k, grid, closure::AIBD, clock, U, args...) = ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), ∂zᶜᶜᶜ, ∇²ᶜᶜᶠ, U.w)

@inline viscous_flux_uz(i, j, k, grid, closure::AVBD, clock, U, args...) = ν_uzzzᶠᶜᶠ(i, j, k, grid, clock, closure.ν, U.u)
@inline viscous_flux_vz(i, j, k, grid, closure::AVBD, clock, U, args...) = ν_vzzzᶜᶠᶠ(i, j, k, grid, clock, closure.ν, U.v)
@inline viscous_flux_wz(i, j, k, grid, closure::AVBD, clock, U, args...) = ν_wzzzᶜᶜᶜ(i, j, k, grid, clock, closure.ν, U.w)

# are we sure about this?? This might not be consistent for AHBD
@inline viscous_flux_wx(i, j, k, grid, closure::Union{AIBD, AHBD}, clock, U, args...) = ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, args...), ∂xᶠᶜᶠ, ∇²ᶜᶜᶠ, U.w)
@inline viscous_flux_wy(i, j, k, grid, closure::Union{AIBD, AHBD}, clock, U, args...) = ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, args...), ∂yᶜᶠᶠ, ∇²ᶜᶜᶠ, U.w)

#####
##### Diffusive fluxes
#####

@inline diffusive_flux_x(i, j, k, grid, closure::AIBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κᶠᶜᶜ(i, j, k, grid, clock, closure.κ[tracer_index]) * ∂xᶠᶜᶜ(i, j, k, grid, ∇²ᶜᶜᶜ, c)
@inline diffusive_flux_y(i, j, k, grid, closure::AIBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κᶜᶠᶜ(i, j, k, grid, clock, closure.κ[tracer_index]) * ∂yᶜᶠᶜ(i, j, k, grid, ∇²ᶜᶜᶜ, c)
@inline diffusive_flux_z(i, j, k, grid, closure::AIBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κᶜᶜᶠ(i, j, k, grid, clock, closure.κ[tracer_index]) * ∂zᶜᶜᶠ(i, j, k, grid, ∇²ᶜᶜᶜ, c)

@inline diffusive_flux_x(i, j, k, grid, closure::AHBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κᶠᶜᶜ(i, j, k, grid, clock, closure.κ[tracer_index]) * ∂x_∇²h_cᶠᶜᶜ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, closure::AHBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κᶜᶠᶜ(i, j, k, grid, clock, closure.κ[tracer_index]) * ∂y_∇²h_cᶜᶠᶜ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, closure::AVBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κᶜᶜᶠ(i, j, k, grid, clock, closure.κ[tracer_index]) * ∂³zᶜᶜᶠ(i, j, k, grid, c)

#####
##### Zero out not used fluxes
#####

for (dir, closure) in zip((:x, :y, :z), (:AVBD, :AVBD, :AHBD))
    diffusive_flux = Symbol(:diffusive_flux_, dir)
    viscous_flux_u = Symbol(:viscous_flux_u, dir)
    viscous_flux_v = Symbol(:viscous_flux_v, dir)
    viscous_flux_w = Symbol(:viscous_flux_w, dir)
    @eval begin
        @inline $diffusive_flux(i, j, k, grid, closure::$closure, c, c_idx, clock, args...) = zero(eltype(grid))
        @inline $viscous_flux_u(i, j, k, grid, closure::$closure, c, clock, U, args...)     = zero(eltype(grid))
        @inline $viscous_flux_v(i, j, k, grid, closure::$closure, c, clock, U, args...)     = zero(eltype(grid))
        @inline $viscous_flux_w(i, j, k, grid, closure::$closure, c, clock, U, args...)     = zero(eltype(grid))
    end
end
