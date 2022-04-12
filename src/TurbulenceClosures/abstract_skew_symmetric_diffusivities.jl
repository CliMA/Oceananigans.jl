"""
    abstract type AbstractSkewSymmetricDiffusivity{Tapering} end

Abstract type for skew-symmetric diffusivities with tapering.
"""
abstract type AbstractSkewSymmetricDiffusivity{Tapering} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization} end

const OneASSD = AbstractSkewSymmetricDiffusivity
const ManyASSD = AbstractVector{<:OneASSD}
const ASSD = Union{OneASSD, ManyASSD}
const assd_coefficient_loc = (Center, Center, Center)

#####
##### Tapering
#####

struct FluxTapering{FT}
    max_slope :: FT
end

const FluxTaperedASSD = AbstractSkewSymmetricDiffusivity{FluxTapering}

"""
    taper_factor_ccc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, tapering::FluxTapering) 

Return the tapering factor `min(1, Sₘₐₓ² / slope²)`, where `slope² = slope_x² + slope_y²`
that multiplies all components of the isopycnal slope tensor. All slopes involved in the
tapering factor are computed at the cell centers.

References
==========
R. Gerdes, C. Koberle, and J. Willebrand. (1991), "The influence of numerical advection schemes
    on the results of ocean general circulation models", Clim. Dynamics, 5 (4), 211–226.
"""
@inline function taper_factor_ccc(i, j, k, grid, closure::FluxTaperedASSD, buoyancy, tracers)
    bx = ℑxᶜᵃᵃ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    by = ℑyᵃᶜᵃ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    
    slope_x = - bx / bz
    slope_y = - by / bz
    slope² = ifelse(bz <= 0, zero(FT), slope_x^2 + slope_y^2)

    tapering = flux_tapering(closure)

    return min(one(FT), tapering.max_slope^2 / slope²)
end

"""
    taper_factor_ccc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, ::Nothing) where FT

Returns 1 for the  isopycnal slope tapering factor, that is, no tapering is done.
"""
taper_factor_ccc(i, j, k, grid, args...) = one(eltype(grid))

# defined at fcc
@inline function diffusive_flux_x(i, j, k, grid, closure::ASSD, K, ::Val{id},
                                  velocities, tracers, clock, buoyancy) where id

    closure = getclosure(closure, i, j)
    κ_skew = skew_diffusivity(closure, id, K)
    κ_symmetric = symmetric_diffusivity(closure, id, K)
    isopycnals = isopycnal_tensor(closure)
    c = tracers[id]

    κ_skewᶠᶜᶜ = κᶠᶜᶜ(i, j, k, grid, clock, assd_coefficient_loc, κ_skew)
    κ_symmetricᶠᶜᶜ = κᶠᶜᶜ(i, j, k, grid, clock, assd_coefficient_loc, κ_symmetric)

    ∂x_c = ∂xᶠᶜᶜ(i, j, k, grid, c)
    ∂y_c = ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, c)
    ∂z_c = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)

    R₁₁ = one(eltype(grid))
    R₁₂ = zero(eltype(grid))
    R₁₃ = isopycnal_rotation_tensor_xz_fcc(i, j, k, grid, buoyancy, tracers, isopycnals)
    
    ϵ = taper_factor_ccc(i, j, k, grid, closure, buoyancy, tracers)

    return - ϵ * (              κ_symmetricᶠᶜᶜ * R₁₁ * ∂x_c +
                                κ_symmetricᶠᶜᶜ * R₁₂ * ∂y_c +
                  (κ_symmetricᶠᶜᶜ - κ_skewᶠᶜᶜ) * R₁₃ * ∂z_c)
end

# defined at cfc
@inline function diffusive_flux_y(i, j, k, grid, closure::ASSD, K, ::Val{id},
                                  velocities, tracers, clock, buoyancy) where id

    closure = getclosure(closure, i, j)
    κ_skew = skew_diffusivity(closure, id, K)
    κ_symmetric = symmetric_diffusivity(closure, id, K)
    isopycnals = isopycnal_tensor(closure)
    c = tracers[id]

    κ_skewᶜᶠᶜ = κᶜᶠᶜ(i, j, k, grid, clock, assd_coefficient_loc, κ_skew)
    κ_symmetricᶜᶠᶜ = κᶜᶠᶜ(i, j, k, grid, clock, assd_coefficient_loc, κ_symmetric)

    ∂x_c = ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, c)
    ∂y_c = ∂yᶜᶠᶜ(i, j, k, grid, c)
    ∂z_c = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)

    R₂₁ = zero(eltype(grid))
    R₂₂ = one(eltype(grid))
    R₂₃ = isopycnal_rotation_tensor_yz_cfc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    ϵ = taper_factor_ccc(i, j, k, grid, closure, buoyancy, tracers)

    return - ϵ * (              κ_symmetricᶜᶠᶜ * R₂₁ * ∂x_c +
                                κ_symmetricᶜᶠᶜ * R₂₂ * ∂y_c +
                  (κ_symmetricᶜᶠᶜ - κ_skewᶜᶠᶜ) * R₂₃ * ∂z_c)
end

# defined at ccf
@inline function diffusive_flux_z(i, j, k, grid, closure::ASSD, K, ::Val{id},
                                  velocities, tracers, clock, buoyancy) where id

    closure = getclosure(closure, i, j)
    κ_skew = skew_diffusivity(closure, id, K)
    κ_symmetric = symmetric_diffusivity(closure, id, K)
    isopycnals = isopycnal_tensor(closure)
    c = tracers[id]

    κ_skewᶜᶜᶠ = κᶜᶜᶠ(i, j, k, grid, clock, assd_coefficient_loc, κ_skew)
    κ_symmetricᶜᶜᶠ = κᶜᶜᶠ(i, j, k, grid, clock, assd_coefficient_loc, κ_symmetric)

    ∂x_c = ℑxzᶜᵃᶠ(i, j, k, grid, ∂xᶠᶜᶜ, c)
    ∂y_c = ℑyzᵃᶜᶠ(i, j, k, grid, ∂yᶜᶠᶜ, c)
    ∂z_c = ∂zᶜᶜᶠ(i, j, k, grid, c)

    R₃₁ = isopycnal_rotation_tensor_xz_ccf(i, j, k, grid, buoyancy, tracers, isopycnals)
    R₃₂ = isopycnal_rotation_tensor_yz_ccf(i, j, k, grid, buoyancy, tracers, isopycnals)
    R₃₃ = isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, tracers, isopycnals)

    ϵ = taper_factor_ccc(i, j, k, grid, closure, buoyancy, tracers)

    return - ϵ * ((κ_symmetricᶜᶜᶠ + κ_skewᶜᶜᶠ) * R₃₁ * ∂x_c +
                  (κ_symmetricᶜᶜᶠ + κ_skewᶜᶜᶠ) * R₃₂ * ∂y_c +
                                κ_symmetricᶜᶜᶠ * R₃₃ * ∂z_c)
end

@inline viscous_flux_ux(i, j, k, grid, closure::ASSD, args...) = zero(eltype(grid))
@inline viscous_flux_uy(i, j, k, grid, closure::ASSD, args...) = zero(eltype(grid))
@inline viscous_flux_uz(i, j, k, grid, closure::ASSD, args...) = zero(eltype(grid))

@inline viscous_flux_vx(i, j, k, grid, closure::ASSD, args...) = zero(eltype(grid))
@inline viscous_flux_vy(i, j, k, grid, closure::ASSD, args...) = zero(eltype(grid))
@inline viscous_flux_vz(i, j, k, grid, closure::ASSD, args...) = zero(eltype(grid))

@inline viscous_flux_wx(i, j, k, grid, closure::ASSD, args...) = zero(eltype(grid))
@inline viscous_flux_wy(i, j, k, grid, closure::ASSD, args...) = zero(eltype(grid))
@inline viscous_flux_wz(i, j, k, grid, closure::ASSD, args...) = zero(eltype(grid))

#####
##### Components of the Redi rotation tensor
#####

"""
    AbstractIsopycnalTensor

Abstract supertype for an isopycnal rotation model.
"""
abstract type AbstractIsopycnalTensor end

"""
    struct IsopycnalTensor{FT} <: AbstractIsopycnalTensor

A tensor that rotates a vector into the isopycnal plane using the local slopes
of the buoyancy field.

Slopes are computed via `slope_x = - ∂b/∂x / ∂b/∂z` and `slope_y = - ∂b/∂y / ∂b/∂z`,
with the negative sign to account for the stable stratification (`∂b/∂z < 0`).
Then, the components of the isopycnal rotation tensor are:

```
               ⎡     1 + slope_y²         - slope_x slope_y      slope_x ⎤ 
(1 + slope²)⁻¹ | - slope_x slope_y          1 + slope_x²         slope_y |
               ⎣       slope_x                 slope_y            slope² ⎦
```

where `slope² = slope_x² + slope_y²`.
"""
struct IsopycnalTensor{FT} <: AbstractIsopycnalTensor
    minimum_bz :: FT
end

"""
    struct SmallSlopeIsopycnalTensor{FT} <: AbstractIsopycnalTensor

A tensor that rotates a vector into the isopycnal plane using the local slopes
of the buoyancy field and employing the small-slope approximation, i.e., that
the horizontal isopycnal slopes, `slope_x` and `slope_y` are ``≪ 1``. Slopes are
computed via `slope_x = - ∂b/∂x / ∂b/∂z` and `slope_y = - ∂b/∂y / ∂b/∂z`, with
the negative sign to account for the stable stratification (`∂b/∂z < 0`). Then,
by utilizing the small-slope appoximation, the components of the isopycnal
rotation tensor are:

```
⎡   1            0         slope_x ⎤ 
|   0            1         slope_y |
⎣ slope_x      slope_y      slope² ⎦
```

where `slope² = slope_x² + slope_y²`.
"""
struct SmallSlopeIsopycnalTensor{FT} <: AbstractIsopycnalTensor
    minimum_bz :: FT
end

SmallSlopeIsopycnalTensor(; minimum_bz = 0) = SmallSlopeIsopycnalTensor(minimum_bz)

# Default
@inline isopycnal_tensor(closure) = SmallSlopeIsopycnalTensor(1e-3)

@inline function isopycnal_rotation_tensor_xz_fcc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, isopycnals::SmallSlopeIsopycnalTensor) where FT
    bx = ∂x_b(i, j, k, grid, buoyancy, tracers)
    bz = ∂zᶠᶜᶜ(i, j, k, grid, ℑxzᶠᵃᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    bz = max(bz, isopycnals.minimum_bz)
    
    slope_x = - bx / bz
    
    return ifelse(bz == 0, zero(FT), slope_x)
end

@inline function isopycnal_rotation_tensor_xz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, isopycnals::SmallSlopeIsopycnalTensor) where FT
    bx = ∂xᶜᶜᶠ(i, j, k, grid, ℑxzᶠᵃᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, isopycnals.minimum_bz)
    
    slope_x = - bx / bz
    
    return ifelse(bz == 0, zero(FT), slope_x)
end

@inline function isopycnal_rotation_tensor_yz_cfc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, isopycnals::SmallSlopeIsopycnalTensor) where FT
    by = ∂y_b(i, j, k, grid, buoyancy, tracers)
    bz = ∂zᶜᶠᶜ(i, j, k, grid, ℑyzᵃᶠᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    bz = max(bz, isopycnals.minimum_bz)
    
    slope_y = - by / bz
    
    return ifelse(bz == 0, zero(FT), slope_y)
end

@inline function isopycnal_rotation_tensor_yz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, isopycnals::SmallSlopeIsopycnalTensor) where FT
    by = ∂yᶜᶜᶠ(i, j, k, grid, ℑyzᵃᶠᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, isopycnals.minimum_bz)
    
    slope_y = - by / bz
    
    return ifelse(bz == 0, zero(FT), slope_y)
end

@inline function isopycnal_rotation_tensor_zz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, isopycnals::SmallSlopeIsopycnalTensor) where FT
    bx = ∂xᶜᶜᶠ(i, j, k, grid, ℑxzᶠᵃᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    by = ∂yᶜᶜᶠ(i, j, k, grid, ℑyzᵃᶠᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, isopycnals.minimum_bz)

    slope_x = - bx / bz
    slope_y = - by / bz
    slope² = slope_x^2 + slope_y^2
    
    return ifelse(bz == 0, zero(FT), slope²)
end
