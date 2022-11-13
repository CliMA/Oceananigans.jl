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

The slopes are tapered using the `slope_limiter.max_slope`, i.e., the tapering factor is
`min(1, slope_limiter.max_slope² / slope²)`, where `slope² = slope_x² + slope_y²`
that multiplies all components of the isopycnal slope tensor.

References
==========
R. Gerdes, C. Koberle, and J. Willebrand. (1991), "The influence of numerical advection schemes
    on the results of ocean general circulation models", Clim. Dynamics, 5 (4), 211–226.
"""
struct SmallSlopeIsopycnalTensor{FT} <: AbstractIsopycnalTensor
    minimum_bz :: FT
end

SmallSlopeIsopycnalTensor(FT::DataType=Float64; minimum_bz = FT(0)) = SmallSlopeIsopycnalTensor(minimum_bz)

@inline function isopycnal_rotation_tensor_xz_fcc(i, j, k, grid::AbstractGrid, buoyancy, tracers,
                                                  slope_model::SmallSlopeIsopycnalTensor)
    bx = ∂x_b(i, j, k, grid, buoyancy, tracers)
    by = ℑxyᶠᶜᵃ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ℑxzᶠᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_x = - bx / bz

    return ifelse(bz == 0, zero(grid), slope_x)
end

@inline function isopycnal_rotation_tensor_xz_ccf(i, j, k, grid::AbstractGrid, buoyancy, tracers,
                                                  slope_model::SmallSlopeIsopycnalTensor)

    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_x = - bx / bz

    return ifelse(bz == 0, zero(grid), slope_x)
end

@inline function isopycnal_rotation_tensor_yz_cfc(i, j, k, grid::AbstractGrid, buoyancy, tracers,
                                                  slope_model::SmallSlopeIsopycnalTensor)

    by = ∂y_b(i, j, k, grid, buoyancy, tracers)
    bz = ℑyzᵃᶠᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_y = - by / bz
    
    return ifelse(bz == 0, zero(grid), slope_y)
end

@inline function isopycnal_rotation_tensor_yz_ccf(i, j, k, grid::AbstractGrid, buoyancy, tracers,
                                                  slope_model::SmallSlopeIsopycnalTensor)

    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)
    
    slope_y = - by / bz
    
    return ifelse(bz == 0, zero(grid), slope_y)
end

@inline function isopycnal_rotation_tensor_zz_ccf(i, j, k, grid::AbstractGrid, buoyancy, tracers,
                                                  slope_model::SmallSlopeIsopycnalTensor)

    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    bz = max(bz, slope_model.minimum_bz)

    slope_x = - bx / bz
    slope_y = - by / bz
    slope² = slope_x^2 + slope_y^2

    return ifelse(bz == 0, zero(grid), slope²)
end
