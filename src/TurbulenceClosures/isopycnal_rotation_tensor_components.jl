# tracer components of the Redi rotation tensor

"""
    AbstractIsopycnalTensor

Abstract supertype for an isopycnal rotation model.
"""
abstract type AbstractIsopycnalTensor end

"""
    A tensor that rotates a vector into the isopycnal plane using the local slopes
    of the buoyancy field. Slopes are computed via `slope_x = - ∂b/∂x / ∂b/∂z` and
    `slope_y = - ∂b/∂y / ∂b/∂z`, with the negative sign to account for the stable
    stratification (`∂b/∂z < 0`). Then, the components of the isopycnal rotation
    tensor are:
    
                     ⎡     1 + slope_y²         - slope_x slope_y      slope_x ⎤ 
      (1 + slope²)⁻¹ | - slope_x slope_y          1 + slope_x²         slope_y |
                     ⎣       slope_x                 slope_y            slope² ⎦
    
    where `slope² = slope_x² + slope_y²`.
"""
struct IsopycnalTensor{FT} <: AbstractIsopycnalTensor
    minimum_bz :: FT
end

"""
    A tensor that rotates a vector into the isopycnal plane using the local slopes
    of the buoyancy field and employing the small-slope approximation, i.e., that
    the horizontal isopycnal slopes, `slope_x` and `slope_y` are ``≪ 1``. Slopes are
    computed via `slope_x = - ∂b/∂x / ∂b/∂z` and `slope_y = - ∂b/∂y / ∂b/∂z`, with
    the negative sign to account for the stable stratification (`∂b/∂z < 0`). Then,
    by utilizing the small-slope appoximation, the components of the isopycnal
    rotation tensor are:
    
      ⎡   1            0         slope_x ⎤ 
      |   0            1         slope_y |
      ⎣ slope_x      slope_y      slope² ⎦
    
    where `slope² = slope_x² + slope_y²`.
"""
struct SmallSlopeIsopycnalTensor{FT} <: AbstractIsopycnalTensor
    minimum_bz :: FT
end

SmallSlopeIsopycnalTensor(; minimum_bz = 0) = SmallSlopeIsopycnalTensor(minimum_bz)

@inline function isopycnal_rotation_tensor_xz_fcc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    bx = ∂x_b(i, j, k, grid, buoyancy, tracers)
    bz = ∂zᵃᵃᶜ(i, j, k, grid, ℑxzᶠᵃᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_x = - bx / bz
    
    return ifelse(bz == 0, zero(FT), slope_x)
end

@inline function isopycnal_rotation_tensor_xz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    bx = ∂xᶜᶜᵃ(i, j, k, grid, ℑxzᶠᵃᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_x = - bx / bz
    
    return ifelse(bz == 0, zero(FT), slope_x)
end

@inline function isopycnal_rotation_tensor_yz_cfc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    by = ∂y_b(i, j, k, grid, buoyancy, tracers)
    bz = ∂zᵃᵃᶜ(i, j, k, grid, ℑyzᵃᶠᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_y = - by / bz
    
    return ifelse(bz == 0, zero(FT), slope_y)
end

@inline function isopycnal_rotation_tensor_yz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    by = ∂yᶜᶜᵃ(i, j, k, grid, ℑyzᵃᶠᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_y = - by / bz
    
    return ifelse(bz == 0, zero(FT), slope_y)
end

@inline function isopycnal_rotation_tensor_zz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, slope_model::SmallSlopeIsopycnalTensor) where FT
    bx = ∂xᶜᶜᵃ(i, j, k, grid, ℑxzᶠᵃᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    by = ∂yᶜᶜᵃ(i, j, k, grid, ℑyzᵃᶠᶠ, buoyancy_perturbation, buoyancy.model, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)

    slope_x = - bx / bz
    slope_y = - by / bz
    slope² = slope_x^2 + slope_y^2
    
    return ifelse(bz == 0, zero(FT), slope²)
end
