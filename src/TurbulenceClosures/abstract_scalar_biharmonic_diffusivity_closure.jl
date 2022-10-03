using Oceananigans.Grids: peripheral_node

"""
    abstract type AbstractScalarDiffusivity <: AbstractTurbulenceClosure end

Abstract type for closures with *isotropic* diffusivities.
"""
abstract type AbstractScalarBiharmonicDiffusivity{F} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization} end

@inline formulation(::AbstractScalarBiharmonicDiffusivity{F}) where {F} = F()

const ASBD = AbstractScalarBiharmonicDiffusivity

#####
##### Coefficient extractors
#####

const ccc = (Center(), Center(), Center())
@inline νᶜᶜᶜ(i, j, k, grid, closure::ASBD, K, clock, fields) = νᶜᶜᶜ(i, j, k, grid, ccc, viscosity(closure, K), clock, fields) 
@inline νᶠᶠᶜ(i, j, k, grid, closure::ASBD, K, clock, fields) = νᶠᶠᶜ(i, j, k, grid, ccc, viscosity(closure, K), clock, fields)
@inline νᶠᶜᶠ(i, j, k, grid, closure::ASBD, K, clock, fields) = νᶠᶜᶠ(i, j, k, grid, ccc, viscosity(closure, K), clock, fields)
@inline νᶜᶠᶠ(i, j, k, grid, closure::ASBD, K, clock, fields) = νᶜᶠᶠ(i, j, k, grid, ccc, viscosity(closure, K), clock, fields)

@inline κᶠᶜᶜ(i, j, k, grid, closure::ASBD, K, id, clock, fields) = κᶠᶜᶜ(i, j, k, grid, ccc, diffusivity(closure, K, id), clock, fields)
@inline κᶜᶠᶜ(i, j, k, grid, closure::ASBD, K, id, clock, fields) = κᶜᶠᶜ(i, j, k, grid, ccc, diffusivity(closure, K, id), clock, fields)
@inline κᶜᶜᶠ(i, j, k, grid, closure::ASBD, K, id, clock, fields) = κᶜᶜᶠ(i, j, k, grid, ccc, diffusivity(closure, K, id), clock, fields)

#####
##### Stress divergences
#####

const AIBD = AbstractScalarBiharmonicDiffusivity{<:ThreeDimensionalFormulation}
const AHBD = AbstractScalarBiharmonicDiffusivity{<:HorizontalFormulation}
const ADBD = AbstractScalarBiharmonicDiffusivity{<:HorizontalDivergenceFormulation}
const AVBD = AbstractScalarBiharmonicDiffusivity{<:VerticalFormulation}

@inline viscous_flux_ux(i, j, k, grid, clo::AIBD, K, clk, fields, b) = + ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, ∂xᶜᶜᶜ, biharmonic_mask_x, ∇²ᶠᶜᶜ, fields.u)
@inline viscous_flux_vx(i, j, k, grid, clo::AIBD, K, clk, fields, b) = + ν_σᶠᶠᶜ(i, j, k, grid, clo, K, clk, fields, biharmonic_mask_x, ∂xᶠᶠᶜ, ∇²ᶜᶠᶜ, fields.v)
@inline viscous_flux_wx(i, j, k, grid, clo::AIBD, K, clk, fields, b) = + ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, fields, biharmonic_mask_x, ∂xᶠᶜᶠ, ∇²ᶜᶜᶠ, fields.w)
@inline viscous_flux_uy(i, j, k, grid, clo::AIBD, K, clk, fields, b) = + ν_σᶠᶠᶜ(i, j, k, grid, clo, K, clk, fields, biharmonic_mask_y, ∂yᶠᶠᶜ, ∇²ᶠᶜᶜ, fields.u)
@inline viscous_flux_vy(i, j, k, grid, clo::AIBD, K, clk, fields, b) = + ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, ∂yᶜᶜᶜ, biharmonic_mask_y, ∇²ᶜᶠᶜ, fields.v)
@inline viscous_flux_wy(i, j, k, grid, clo::AIBD, K, clk, fields, b) = + ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, fields, biharmonic_mask_y, ∂yᶜᶠᶠ, ∇²ᶜᶜᶠ, fields.w)
@inline viscous_flux_uz(i, j, k, grid, clo::AIBD, K, clk, fields, b) = + ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, fields, biharmonic_mask_z, ∂zᶠᶜᶠ, ∇²ᶠᶜᶜ, fields.u)
@inline viscous_flux_vz(i, j, k, grid, clo::AIBD, K, clk, fields, b) = + ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, fields, biharmonic_mask_z, ∂zᶜᶠᶠ, ∇²ᶜᶠᶜ, fields.v)
@inline viscous_flux_wz(i, j, k, grid, clo::AIBD, K, clk, fields, b) = + ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, ∂zᶜᶜᶜ, biharmonic_mask_z, ∇²ᶜᶜᶠ, fields.w)
@inline viscous_flux_ux(i, j, k, grid, clo::AHBD, K, clk, fields, b) = + ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, δ★ᶜᶜᶜ, fields.u, fields.v)   
@inline viscous_flux_vx(i, j, k, grid, clo::AHBD, K, clk, fields, b) = + ν_σᶠᶠᶜ(i, j, k, grid, clo, K, clk, fields, ζ★ᶠᶠᶜ, fields.u, fields.v)
@inline viscous_flux_wx(i, j, k, grid, clo::AHBD, K, clk, fields, b) = + ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, fields, biharmonic_mask_x, ∂xᶠᶜᶠ, ∇²ᶜᶜᶠ, fields.w)
@inline viscous_flux_uy(i, j, k, grid, clo::AHBD, K, clk, fields, b) = - ν_σᶠᶠᶜ(i, j, k, grid, clo, K, clk, fields, ζ★ᶠᶠᶜ, fields.u, fields.v)   
@inline viscous_flux_vy(i, j, k, grid, clo::AHBD, K, clk, fields, b) = + ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, δ★ᶜᶜᶜ, fields.u, fields.v)
@inline viscous_flux_wy(i, j, k, grid, clo::AHBD, K, clk, fields, b) = + ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, fields, biharmonic_mask_y, ∂yᶜᶠᶠ, ∇²ᶜᶜᶠ,  fields.w)
@inline viscous_flux_uz(i, j, k, grid, clo::AVBD, K, clk, fields, b) = + ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, fields, biharmonic_mask_z, ∂zᶠᶜᶠ, ∂²zᶠᶜᶜ, fields.u)
@inline viscous_flux_vz(i, j, k, grid, clo::AVBD, K, clk, fields, b) = + ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, fields, biharmonic_mask_z, ∂zᶜᶠᶠ, ∂²zᶜᶠᶜ, fields.v)
@inline viscous_flux_wz(i, j, k, grid, clo::AVBD, K, clk, fields, b) = + ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, ∂zᶜᶜᶜ, biharmonic_mask_z, ∂²zᶜᶜᶠ, fields.w)

@inline viscous_flux_ux(i, j, k, grid, clo::ADBD, K, clk, fields, b) = + ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, δ★ᶜᶜᶜ, fields.u, fields.v)   
@inline viscous_flux_vy(i, j, k, grid, clo::ADBD, K, clk, fields, b) = + ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, δ★ᶜᶜᶜ, fields.u, fields.v)

#####
##### Diffusive fluxes
#####

@inline diffusive_flux_x(i, j, k, grid, clo::AIBD, K, ::Val{id}, c, clk, fields, b) where id = κ_σᶠᶜᶜ(i, j, k, grid, clo, K, Val(id), clk, fields, biharmonic_mask_x, ∂xᶠᶜᶜ, ∇²ᶜᶜᶜ, c)
@inline diffusive_flux_y(i, j, k, grid, clo::AIBD, K, ::Val{id}, c, clk, fields, b) where id = κ_σᶜᶠᶜ(i, j, k, grid, clo, K, Val(id), clk, fields, biharmonic_mask_y, ∂yᶜᶠᶜ, ∇²ᶜᶜᶜ, c)
@inline diffusive_flux_z(i, j, k, grid, clo::AIBD, K, ::Val{id}, c, clk, fields, b) where id = κ_σᶜᶜᶠ(i, j, k, grid, clo, K, Val(id), clk, fields, biharmonic_mask_z, ∂zᶜᶜᶠ, ∇²ᶜᶜᶜ, c)
@inline diffusive_flux_x(i, j, k, grid, clo::AHBD, K, ::Val{id}, c, clk, fields, b) where id = κ_σᶠᶜᶜ(i, j, k, grid, clo, K, Val(id), clk, fields, biharmonic_mask_x, ∂x_∇²h_cᶠᶜᶜ, c)
@inline diffusive_flux_y(i, j, k, grid, clo::AHBD, K, ::Val{id}, c, clk, fields, b) where id = κ_σᶜᶠᶜ(i, j, k, grid, clo, K, Val(id), clk, fields, biharmonic_mask_y, ∂y_∇²h_cᶜᶠᶜ, c)
@inline diffusive_flux_z(i, j, k, grid, clo::AVBD, K, ::Val{id}, c, clk, fields, b) where id = κ_σᶜᶜᶠ(i, j, k, grid, clo, K, Val(id), clk, fields, biharmonic_mask_z, ∂³zᶜᶜᶠ, c)

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

@inline biharmonic_mask_x(i, j, k, grid, f, args...) = ifelse(x_peripheral_node(i, j, k, grid), zero(eltype(grid)), f(i, j, k, grid, args...))
@inline biharmonic_mask_y(i, j, k, grid, f, args...) = ifelse(y_peripheral_node(i, j, k, grid), zero(eltype(grid)), f(i, j, k, grid, args...))
@inline biharmonic_mask_z(i, j, k, grid, f, args...) = ifelse(z_peripheral_node(i, j, k, grid), zero(eltype(grid)), f(i, j, k, grid, args...))

@inline x_peripheral_node(i, j, k, grid) = peripheral_node(i, j, k, grid, Face(), Center(), Center())
@inline y_peripheral_node(i, j, k, grid) = peripheral_node(i, j, k, grid, Center(), Face(), Center())
@inline z_peripheral_node(i, j, k, grid) = peripheral_node(i, j, k, grid, Center(), Center(), Face())
