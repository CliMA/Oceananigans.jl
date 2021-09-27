# tracer components of the Redi rotation tensor

"""
    AbstractIcopycnalModel

Abstract supertype for isopycnal model.
"""
abstract type AbstractIcopycnalModel end

"""
    An isopycnal model using the local slopes of the buoyancy field. Slopes are
    computed via `slope_x = - ∂b/∂x / ∂b/∂z` and `slope_y = - ∂b/∂y / ∂b/∂z`,
    with the negative sign to account for the stable stratification. Then, the
    components of the isopycnal rotation tensor are:
    
                     ⎡     1 + slope_y²         - slope_x slope_y      slope_x ⎤ 
      (1 + slope²)⁻¹ | - slope_x slope_y          1 + slope_x²         slope_y |
                     ⎣       slope_x                 slope_y            slope² ⎦
    
    where `slope² = slope_x² + slope_y²`.
"""
struct SlopeApproximation <: AbstractIcopycnalModel end

"""
    An isopycnal model using the local slopes of the buoyancy field that assumes
    utilizes the small-slope approximation, i.e., that the horizontal isopycnal
    slopes, `slope_x` and `slope_y` are ``≪ 1``. Slopes are computed via
    `slope_x = - ∂b/∂x / ∂b/∂z` and `slope_y = - ∂b/∂y / ∂b/∂z`, with the
    negative sign to account for the stable stratification. Then, keeping only
    terms up to linear in `slope_x` or `slope_y`, the components of the isopycnal
    rotation tensor are:
    
      ⎡   1            0         slope_x ⎤ 
      |   0            1         slope_y |
      ⎣ slope_x      slope_y      slope² ⎦
    
    where `slope² = slope_x² + slope_y²`.
"""
struct SmallSlopeApproximation <: AbstractIcopycnalModel end

@inline function isopycnal_rotation_tensor_xz_fcc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, ::SmallSlopeApproximation) where FT
    bx = ∂x_b(i, j, k, grid, buoyancy, tracers)
    bz = ℑxzᶠᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    slope_x = - bx / bz
    return ifelse(bz == 0, zero(FT), slope_x)
end

@inline function isopycnal_rotation_tensor_xz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, ::SmallSlopeApproximation) where FT
    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    slope_x = - bx / bz
    return ifelse(bz == 0, zero(FT), slope_x)
end

@inline function isopycnal_rotation_tensor_yz_cfc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, ::SmallSlopeApproximation) where FT
    by = ∂y_b(i, j, k, grid, buoyancy, tracers)
    bz = ℑyzᵃᶠᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    slope_y = - by / bz
    return ifelse(bz == 0, zero(FT), slope_y)
end

@inline function isopycnal_rotation_tensor_yz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, ::SmallSlopeApproximation) where FT
    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    slope_y = - by / bz
    return ifelse(bz == 0, zero(FT), slope_y)
end

@inline function isopycnal_rotation_tensor_zz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, ::SmallSlopeApproximation) where FT
    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    slope_x = - bx / bz
    slope_y = - by / bz
    slope² = slope_x^2 + slope_y^2
    return ifelse(bz == 0, zero(FT), slope²)
end
