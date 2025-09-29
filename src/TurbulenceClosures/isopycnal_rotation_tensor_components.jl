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