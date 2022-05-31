using Oceananigans.Operators: ℑxzᶠᶜᶜ, ℑxzᶜᶜᶠ, ℑyzᶜᶠᶜ, ℑyzᶜᶜᶠ, ℑxyzᶠᶜᶠ

# tracer components of the Redi rotation tensor

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

#####
##### _triads_
#####
##### There are two horizontal slopes: Sx and Sy
#####
##### Both slopes are "located" at tracer cell centers.
#####
##### The slopes are computed by a directional derivative, which lends an
##### "orientation" to the slope. For example, the x-slope `Sx` computed
##### with a "+" directional derivative in x, and a "+" directional derivative
##### in z, is
#####
##### Sx⁺⁺ᵢₖ = Δz / Δx * (bᵢ₊₁ - bᵢ) / (bₖ₊₁ - bₖ)
#####
##### The superscript codes ⁺⁺, ⁺⁻, ⁻⁺, ⁻⁻, denote the direction of the derivative
##### in (h, z).
#####

@inline function triad_Sx(ix, jx, kx, iz, jz, kz, grid, buoyancy, tracers)
    bx = ∂x_b(ix, jx, kx, grid, buoyancy, tracers)
    bz = ∂z_b(iz, jz, kz, grid, buoyancy, tracers)
    bz = max(bz, zero(grid))
    return ifelse(bz == 0, zero(grid), - bx / bz)
end

@inline function triad_Sy(iy, jy, ky, iz, jz, kz, grid, buoyancy, tracers)
    by = ∂y_b(iy, jy, ky, grid, buoyancy, tracers)
    bz = ∂z_b(iz, jz, kz, grid, buoyancy, tracers)
    bz = max(bz, zero(grid))
    return ifelse(bz == 0, zero(grid), - by / bz)
end

@inline Sx⁺⁺(i, j, k, grid, buoyancy, tracers) = triad_Sx(i+1, j, k, i, j, k+1, grid, buoyancy, tracers)
@inline Sx⁺⁻(i, j, k, grid, buoyancy, tracers) = triad_Sx(i+1, j, k, i, j, k,   grid, buoyancy, tracers)
@inline Sx⁻⁺(i, j, k, grid, buoyancy, tracers) = triad_Sx(i,   j, k, i, j, k+1, grid, buoyancy, tracers)
@inline Sx⁻⁻(i, j, k, grid, buoyancy, tracers) = triad_Sx(i,   j, k, i, j, k,   grid, buoyancy, tracers)

@inline Sy⁺⁺(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j+1, k, i, j, k+1, grid, buoyancy, tracers)
@inline Sy⁺⁻(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j+1, k, i, j, k,   grid, buoyancy, tracers)
@inline Sy⁻⁺(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j,   k, i, j, k+1, grid, buoyancy, tracers)
@inline Sy⁻⁻(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j,   k, i, j, k,   grid, buoyancy, tracers)

# Tensor components
@inline Sxᶠᶜᶜ(i, j, k, grid, args...) = (Sx⁻⁺(i,   j, k, grid, args...) + Sx⁻⁻(i,   j, k, grid, args...) +    # west cell, z-average
                                         Sx⁻⁺(i-1, j, k, grid, args...) + Sx⁻⁻(i-1, j, k, grid, args...)) / 4 # east cell, z-average

@inline Syᶜᶠᶜ(i, j, k, grid, args...) = (Sy⁻⁺(i, j,   k, grid, args...) + Sy⁻⁻(i, j,   k, grid, args...) +    # south cell, z-average
                                         Sy⁻⁺(i, j-1, k, grid, args...) + Sy⁻⁻(i, j-1, k, grid, args...)) / 4 # north cell, z-average

@inline Syᶜᶜᶠ(i, j, k, grid, args...) = (Sy⁺⁻(i, j, k,   grid, args...) + Sy⁻⁻(i, j, k,   grid, args...) +    # top cell, y-average
                                         Sy⁺⁺(i, j, k-1, grid, args...) + Sy⁻⁺(i, j, k-1, grid, args...)) / 4 # bottom cell, y-average

@inline Sxᶜᶜᶠ(i, j, k, grid, args...) = (Sx⁺⁻(i, j, k,   grid, args...) + Sx⁻⁻(i, j, k,   grid, args...) +    # top cell, x-average
                                         Sx⁺⁺(i, j, k-1, grid, args...) + Sx⁻⁺(i, j, k-1, grid, args...)) / 4 # bottom cell, x-average

@inline Syᶜᶜᶠ(i, j, k, grid, args...) = (Sy⁺⁻(i, j, k,   grid, args...) + Sy⁻⁻(i, j, k,   grid, args...) +    # top cell, y-average
                                         Sy⁺⁺(i, j, k-1, grid, args...) + Sy⁻⁺(i, j, k-1, grid, args...)) / 4 # bottom cell, y-average

@inline function isopycnal_rotation_tensor_xz_fcc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    bx = ∂x_b(i, j, k, grid, buoyancy, tracers)

    # "Gradient of the average" stencil
    #bz = ∂zᶠᶜᶜ(i, j, k, grid, ℑxzᶠᶜᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    
    bz = ℑxzᶠᶜᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_x = - bx / bz
    
    return ifelse(bz == 0, zero(FT), slope_x)
end

@inline function isopycnal_rotation_tensor_xz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    # "Gradient of the average" stencil
    #bx = ∂xᶜᶜᶠ(i, j, k, grid, ℑxzᶠᶜᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    
    bx = ℑxzᶜᶜᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_x = - bx / bz
    
    return ifelse(bz == 0, zero(FT), slope_x)
end

@inline function isopycnal_rotation_tensor_xz_fcf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    # fcc -> fcf
    bx = ℑzᶠᶜᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)

    # ccf -> fcf
    bz = ℑxᶠᶜᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_x = - bx / bz
    
    return ifelse(bz == 0, zero(FT), slope_x)
end

@inline function isopycnal_rotation_tensor_xz_cff(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    # fcc -> cff
    bx = ℑxyzᶜᶠᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)

    # ccf -> cff
    bz = ℑyᶜᶠᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_x = - bx / bz
    
    return ifelse(bz == 0, zero(FT), slope_x)
end

@inline function isopycnal_rotation_tensor_yz_ffc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    # cfc -> ffc
    by = ℑxzᶠᶠᶜ(i, j, k, grid, ∂y_b, buoyancy, tracers)

    # ccf -> ffc
    bz = ℑxyzᶠᶠᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_y = - by / bz
    
    return ifelse(bz == 0, zero(FT), slope_y)
end

@inline function isopycnal_rotation_tensor_yz_cfc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    by = ∂y_b(i, j, k, grid, buoyancy, tracers)

    # "Gradient of the average" stencil
    #bz = ∂zᶜᶠᶜ(i, j, k, grid, ℑyzᶜᶠᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    
    bz = ℑyzᶜᶠᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_y = - by / bz
    
    return ifelse(bz == 0, zero(FT), slope_y)
end

@inline function isopycnal_rotation_tensor_yz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    by = ℑyzᶜᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)

    # "Gradient of the average" stencil
    #by = ∂yᶜᶜᶠ(i, j, k, grid, ℑyzᶜᶠᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_y = - by / bz
    
    return ifelse(bz == 0, zero(FT), slope_y)
end

@inline function isopycnal_rotation_tensor_yz_fcf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    # cfc -> fcf
    by = ℑxyzᶠᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)

    # ccf -> fcf
    bz = ℑxᶠᶜᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_y = - by / bz
    
    return ifelse(bz == 0, zero(FT), slope_y)
end

@inline function isopycnal_rotation_tensor_yz_cff(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    # cfc -> cff
    by = ℑzᶜᶠᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)

    # ccf -> cff
    bz = ℑyᶜᶠᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_y = - by / bz
    
    return ifelse(bz == 0, zero(FT), slope_y)
end

@inline function isopycnal_rotation_tensor_zz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    bx = ℑxzᶜᶜᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    by = ℑyzᶜᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)

    # "Gradient of the average" stencil
    #bx = ∂xᶜᶜᶠ(i, j, k, grid, ℑxzᶠᶜᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    #by = ∂yᶜᶜᶠ(i, j, k, grid, ℑyzᶜᶠᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)

    slope_x = - bx / bz
    slope_y = - by / bz
    slope² = slope_x^2 + slope_y^2
    
    return ifelse(bz == 0, zero(FT), slope²)
end

@inline function isopycnal_rotation_tensor_zz_fcf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    # fcc -> fcf
    bx = ℑzᶠᶜᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)

    # cfc -> fcf
    by = ℑxyzᶠᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)

    # ccf -> fcf
    bz = ℑxᶠᶜᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)

    slope_x = - bx / bz
    slope_y = - by / bz
    slope² = slope_x^2 + slope_y^2
    
    return ifelse(bz == 0, zero(FT), slope²)
end

@inline function isopycnal_rotation_tensor_zz_cff(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    # fcc -> cff
    bx = ℑxyzᶜᶠᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)

    # cfc -> cff
    by = ℑzᶜᶠᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)

    # ccf -> cff
    bz = ℑyᶜᶠᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)

    slope_x = - bx / bz
    slope_y = - by / bz
    slope² = slope_x^2 + slope_y^2
    
    return ifelse(bz == 0, zero(FT), slope²)
end

