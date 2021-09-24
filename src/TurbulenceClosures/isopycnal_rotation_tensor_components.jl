# tracer components of the Redi rotation tensor

"""
    AbstractTurbulenceClosure

Abstract supertype for turbulence closures.
"""
abstract type AbstractIcopycnalModel end

struct SlopeApproximation <: AbstractIcopycnalModel end
struct SmallSlopeApproximation <: AbstractIcopycnalModel end

# SmallSlopeApproximation approximation
#    1            0         slope_x
#    0            1         slope_y
# slope_x      slope_y      slope²

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

# (full)-SlopeApproximation approximation
#     (1 + slope_y^2) / (1 + slope²)     - slope_x * slope_y / (1 + slope²)   slope_x / (1 + slope²)
# - slope_x * slope_y / (1 + slope²)         (1 + slope_x^2) / (1 + slope²)   slope_y / (1 + slope²)
#             slope_x / (1 + slope²)                 slope_y / (1 + slope²)    slope² / (1 + slope²)
