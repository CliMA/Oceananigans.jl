using Oceananigans.Grids: solid_interface

"""
    abstract type AbstractScalarDiffusivity <: AbstractTurbulenceClosure end

Abstract type for closures with *isotropic* diffusivities.
"""
abstract type AbstractScalarBiharmonicDiffusivity{Iso} <: AbstractTurbulenceClosure{Explicit} end

@inline isotropy(::AbstractScalarBiharmonicDiffusivity{Iso}) where {Iso} = Iso()

#####
##### Stress divergences
#####

const AIBD = AbstractScalarBiharmonicDiffusivity{<:XYZDirections}
const AHBD = AbstractScalarBiharmonicDiffusivity{<:XYDirections}
const AVBD = AbstractScalarBiharmonicDiffusivity{<:ZDirection}

@inline viscous_flux_ux(i, j, k, grid, closure::AIBD, clock, U, args...) = + ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), ∂xᶜᶜᶜ, biharmonic_mask_x, ∇²ᶠᶜᶜ, U.u)
@inline viscous_flux_vx(i, j, k, grid, closure::AIBD, clock, U, args...) = + ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, args...), biharmonic_mask_x, ∂xᶠᶠᶜ, ∇²ᶜᶠᶜ, U.v)
@inline viscous_flux_wx(i, j, k, grid, closure::AIBD, clock, U, args...) = + ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, args...), biharmonic_mask_x, ∂xᶠᶜᶠ, ∇²ᶜᶜᶠ, U.w)
@inline viscous_flux_uy(i, j, k, grid, closure::AIBD, clock, U, args...) = + ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, args...), biharmonic_mask_y, ∂yᶠᶠᶜ, ∇²ᶠᶜᶜ, U.u)
@inline viscous_flux_vy(i, j, k, grid, closure::AIBD, clock, U, args...) = + ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), ∂yᶜᶜᶜ, biharmonic_mask_y, ∇²ᶜᶠᶜ, U.v)
@inline viscous_flux_wy(i, j, k, grid, closure::AIBD, clock, U, args...) = + ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, args...), biharmonic_mask_y, ∂yᶜᶠᶠ, ∇²ᶜᶜᶠ, U.w)
@inline viscous_flux_uz(i, j, k, grid, closure::AIBD, clock, U, args...) = + ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, args...), biharmonic_mask_z, ∂zᶠᶜᶠ, ∇²ᶠᶜᶜ, U.u)
@inline viscous_flux_vz(i, j, k, grid, closure::AIBD, clock, U, args...) = + ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, args...), biharmonic_mask_z, ∂zᶜᶠᶠ, ∇²ᶜᶠᶜ, U.v)
@inline viscous_flux_wz(i, j, k, grid, closure::AIBD, clock, U, args...) = + ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), ∂zᶜᶜᶜ, biharmonic_mask_z, ∇²ᶜᶜᶠ, U.w)

@inline viscous_flux_ux(i, j, k, grid, closure::AHBD, clock, U, args...) = + ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), δ★ᶜᶜᶜ, U.u, U.v)   
@inline viscous_flux_vx(i, j, k, grid, closure::AHBD, clock, U, args...) = + ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, args...), ζ★ᶠᶠᶜ, U.u, U.v)
@inline viscous_flux_wx(i, j, k, grid, closure::AHBD, clock, U, args...) = + ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, args...), biharmonic_mask_x, ∂xᶠᶜᶠ, ∇²ᶜᶜᶠ, U.w)
@inline viscous_flux_uy(i, j, k, grid, closure::AHBD, clock, U, args...) = - ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, args...), ζ★ᶠᶠᶜ, U.u, U.v)   
@inline viscous_flux_vy(i, j, k, grid, closure::AHBD, clock, U, args...) = + ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), δ★ᶜᶜᶜ, U.u, U.v)
@inline viscous_flux_wy(i, j, k, grid, closure::AHBD, clock, U, args...) = + ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, args...), biharmonic_mask_y, ∂yᶜᶠᶠ, ∇²ᶜᶜᶠ, U.w)

@inline viscous_flux_uz(i, j, k, grid, closure::AVBD, clock, U, args...) = + ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, args...), biharmonic_mask_z, ∂zᶠᶜᶠ, ∂²zᶠᶜᶜ, U.u)
@inline viscous_flux_vz(i, j, k, grid, closure::AVBD, clock, U, args...) = + ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, args...), biharmonic_mask_z, ∂zᶜᶠᶠ, ∂²zᶜᶠᶜ, U.v)
@inline viscous_flux_wz(i, j, k, grid, closure::AVBD, clock, U, args...) = + ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), ∂zᶜᶜᶜ, biharmonic_mask_z, ∂²zᶜᶜᶠ, U.w)

#####
##### Diffusive fluxes
#####

@inline diffusive_flux_x(i, j, k, grid, closure::AIBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κ_σᶠᶜᶜ(i, j, k, grid, clock, closure.κ[tracer_index], biharmonic_mask_x, ∂xᶠᶜᶜ, ∇²ᶜᶜᶜ, c)
@inline diffusive_flux_y(i, j, k, grid, closure::AIBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κ_σᶜᶠᶜ(i, j, k, grid, clock, closure.κ[tracer_index], biharmonic_mask_y, ∂yᶜᶠᶜ, ∇²ᶜᶜᶜ, c)
@inline diffusive_flux_z(i, j, k, grid, closure::AIBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κ_σᶜᶜᶠ(i, j, k, grid, clock, closure.κ[tracer_index], biharmonic_mask_z, ∂zᶜᶜᶠ, ∇²ᶜᶜᶜ, c)

@inline diffusive_flux_x(i, j, k, grid, closure::AHBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κ_σᶠᶜᶜ(i, j, k, grid, clock, closure.κ[tracer_index], biharmonic_mask_x, ∂x_∇²h_cᶠᶜᶜ, c)
@inline diffusive_flux_y(i, j, k, grid, closure::AHBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κ_σᶜᶠᶜ(i, j, k, grid, clock, closure.κ[tracer_index], biharmonic_mask_y, ∂y_∇²h_cᶜᶠᶜ, c)
@inline diffusive_flux_z(i, j, k, grid, closure::AVBD, c, ::Val{tracer_index}, clock, args...) where tracer_index = κ_σᶜᶜᶠ(i, j, k, grid, clock, closure.κ[tracer_index], biharmonic_mask_z, ∂³zᶜᶜᶠ, c)

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

#####
##### Biharmonic-specific viscous operators
#####

# See https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-dissipation
@inline function δ★ᶜᶜᶜ(i, j, k, grid, u, v)

    # These closures seem to be needed to help the compiler infer types
    # (either of u and v or of the function arguments)
    @inline Δy_∇²u(i, j, k, grid, u) = Δy_qᶠᶜᶜ(i, j, k, grid, biharmonic_mask_x, ∇²hᶠᶜᶜ, u)
    @inline Δx_∇²v(i, j, k, grid, v) = Δx_qᶜᶠᶜ(i, j, k, grid, biharmonic_mask_y, ∇²hᶜᶠᶜ, v)

    return 1 / Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Δy_∇²u, u) +
                                       δyᵃᶜᵃ(i, j, k, grid, Δx_∇²v, v))
end

@inline function ζ★ᶠᶠᶜ(i, j, k, grid, u, v)

    # These closures seem to be needed to help the compiler infer types
    # (either of u and v or of the function arguments)
    @inline Δy_∇²v(i, j, k, grid, v) = Δy_qᶜᶠᶜ(i, j, k, grid, biharmonic_mask_y, ∇²hᶜᶠᶜ, v)
    @inline Δx_∇²u(i, j, k, grid, u) = Δx_qᶠᶜᶜ(i, j, k, grid, biharmonic_mask_x, ∇²hᶠᶜᶜ, u)

    return 1 / Azᶠᶠᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Δy_∇²v, v) -
                                       δyᵃᶠᵃ(i, j, k, grid, Δx_∇²u, u))
end

#####
##### Biharmonic-specific diffusion operators
#####

@inline ∂x_∇²h_cᶠᶜᶜ(i, j, k, grid, c) = 1 / Azᶠᶜᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, Δy_qᶜᶜᶜ, ∇²hᶜᶜᶜ, c)
@inline ∂y_∇²h_cᶜᶠᶜ(i, j, k, grid, c) = 1 / Azᶜᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, Δx_qᶜᶜᶜ, ∇²hᶜᶜᶜ, c)

#####
##### Biharmonic-specific operators that enforce "no-flux" boundary conditions and "0-value" boundary conditions for the Laplacian operator
##### biharmonic_mask(∂(∇²(var))) ensures that fluxes are 0 on the boundaries
##### ∂(biharmonic_mask(∇²(var))) ensures that laplacians are 0 on the boundaries
#####

biharmonic_mask_x(i, j, k, grid, f, args...) = ifelse(solid_x(i, j, k, grid), zero(eltype(grid)), f(i, j, k, grid, args...))
biharmonic_mask_y(i, j, k, grid, f, args...) = ifelse(solid_y(i, j, k, grid), zero(eltype(grid)), f(i, j, k, grid, args...))
biharmonic_mask_z(i, j, k, grid, f, args...) = ifelse(solid_z(i, j, k, grid), zero(eltype(grid)), f(i, j, k, grid, args...))

solid_x(i, j, k, grid) = solid_interface(Face(), Center(), Center(), i, j, k, grid)
solid_y(i, j, k, grid) = solid_interface(Center(), Face(), Center(), i, j, k, grid)
solid_z(i, j, k, grid) = solid_interface(Center(), Center(), Face(), i, j, k, grid)
