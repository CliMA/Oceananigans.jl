#####
##### This file provides functions that conditionally-evaluate interpolation operators
##### near boundaries in bounded directions
#####

using Oceananigans.Grids: AbstractGrid,
                          AbstractUnderlyingGrid,
                          Bounded,
                          RightConnected,
                          LeftConnected,
                          topology,
                          architecture

const AG  = AbstractGrid
const AUG = AbstractUnderlyingGrid

# topologies bounded at least on one side
const BT = Union{Bounded, RightConnected, LeftConnected}

# Bounded Grids
const AGX = AUG{<:Any, <:BT}
const AGY = AUG{<:Any, <:Any, <:BT}
const AGZ = AUG{<:Any, <:Any, <:Any, <:BT}

# Reduction of the order near boundaries
#
# For faces reconstructions with NoBias (tracers for example):
#               B                                                           B
#  cells:   --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
#  order:       1     1     2     3   ....        ....    3     2     1     1
#
# For faces reconstructions with LeftBias (tracers for example):
#               B                                                           B
#  cells:   --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
#  order:       1     1     2     3   ....        ....    4     3     2     1
#
# For faces reconstructions with RightBias (tracers for example):
#               B                                                           B
#  cells:   --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
#  order:       1     2     3     4   ....        ....    3     2     1     1
#
# For center reconstructions the bias does not matter (vorticity for example):
#               B                                                           B
#  cells:   --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
#  order:    1     1     2     3    ...               ...    3     2     1     1
#
@inline reduced_face_order(i, ::Type{RightConnected}, N, B, ::NoBias) = max(1, min(B, i-1))
@inline reduced_face_order(i, ::Type{LeftConnected},  N, B, ::NoBias) = max(1, min(B, N+1-i))
@inline reduced_face_order(i, ::Type{Bounded},        N, B, ::NoBias) = max(1, min(B, i-1, N+1-i))

@inline reduced_face_order(i, ::Type{RightConnected}, N, B, ::LeftBias) = max(1, min(B, i-1))
@inline reduced_face_order(i, ::Type{LeftConnected},  N, B, ::LeftBias) = max(1, min(B, N+2-i))
@inline reduced_face_order(i, ::Type{Bounded},        N, B, ::LeftBias) = max(1, min(B, i-1, N+2-i))

@inline reduced_face_order(i, ::Type{RightConnected}, N, B, ::RightBias) = max(1, min(B, i))
@inline reduced_face_order(i, ::Type{LeftConnected},  N, B, ::RightBias) = max(1, min(B, N+1-i))
@inline reduced_face_order(i, ::Type{Bounded},        N, B, ::RightBias) = max(1, min(B, i, N+1-i))

@inline reduced_center_order(i, ::Type{RightConnected}, N, B, bias) = max(1, min(B, i))
@inline reduced_center_order(i, ::Type{LeftConnected},  N, B, bias) = max(1, min(B, N+1-i))
@inline reduced_center_order(i, ::Type{Bounded},        N, B, bias) = max(1, min(B, i, N+1-i))

const A{B} = AbstractAdvectionScheme{B}

# Clamping of reduced order for schemes with a minimum buffer (e.g., WENO with minimum_buffer_upwind_order).
# When red_order drops below the minimum, set it to 0 (sentinel for centered 2nd-order fallback).
@inline clamp_reduced_order(scheme, red_order) = red_order
@inline clamp_reduced_order(scheme::WENO, red_order) = ifelse(red_order < scheme.minimum_buffer_upwind_order, 0, red_order)
@inline clamp_reduced_order(scheme::UpwindBiased, red_order) = ifelse(red_order < scheme.minimum_buffer_upwind_order, 0, red_order)

# Fallback for periodic underlying grids
@inline compute_face_reduced_order_x(i, j, k, grid::AUG, ::A{B}, bias) where B = B
@inline compute_face_reduced_order_y(i, j, k, grid::AUG, ::A{B}, bias) where B = B
@inline compute_face_reduced_order_z(i, j, k, grid::AUG, ::A{B}, bias) where B = B

# Fallback for periodic underlying grids
@inline compute_center_reduced_order_x(i, j, k, grid::AUG, ::A{B}, bias) where B = B
@inline compute_center_reduced_order_y(i, j, k, grid::AUG, ::A{B}, bias) where B = B
@inline compute_center_reduced_order_z(i, j, k, grid::AUG, ::A{B}, bias) where B = B

# Bounded grids
@inline compute_face_reduced_order_x(i, j, k, grid::AGX, ::A{B}, bias) where B = reduced_face_order(i, topology(grid, 1), size(grid, 1), B, bias)
@inline compute_face_reduced_order_y(i, j, k, grid::AGY, ::A{B}, bias) where B = reduced_face_order(j, topology(grid, 2), size(grid, 2), B, bias)
@inline compute_face_reduced_order_z(i, j, k, grid::AGZ, ::A{B}, bias) where B = reduced_face_order(k, topology(grid, 3), size(grid, 3), B, bias)

# Fallback for periodic underlying grids
@inline compute_center_reduced_order_x(i, j, k, grid::AGX, ::A{B}, bias) where B = reduced_center_order(i, topology(grid, 1), size(grid, 1), B, bias)
@inline compute_center_reduced_order_y(i, j, k, grid::AGY, ::A{B}, bias) where B = reduced_center_order(j, topology(grid, 2), size(grid, 2), B, bias)
@inline compute_center_reduced_order_z(i, j, k, grid::AGZ, ::A{B}, bias) where B = reduced_center_order(k, topology(grid, 3), size(grid, 3), B, bias)

@inline function _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias, args...)
    red_order = compute_face_reduced_order_x(i, j, k, grid, scheme, bias)
    red_order = clamp_reduced_order(scheme, red_order)
    return biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, red_order, bias, args...)
end

@inline function _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias, args...)
    red_order = compute_face_reduced_order_y(i, j, k, grid, scheme, bias)
    red_order = clamp_reduced_order(scheme, red_order)
    return biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, red_order, bias, args...)
end

@inline function _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, bias, args...)
    red_order = compute_face_reduced_order_z(i, j, k, grid, scheme, bias)
    red_order = clamp_reduced_order(scheme, red_order)
    return biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, red_order, bias, args...)
end

@inline function _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, bias, args...)
    red_order = compute_center_reduced_order_x(i, j, k, grid, scheme, bias)
    red_order = clamp_reduced_order(scheme, red_order)
    return biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, red_order, bias, args...)
end

@inline function _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, bias, args...)
    red_order = compute_center_reduced_order_y(i, j, k, grid, scheme, bias)
    red_order = clamp_reduced_order(scheme, red_order)
    return biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, red_order, bias, args...)
end

@inline function _biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, bias, args...)
    red_order = compute_center_reduced_order_z(i, j, k, grid, scheme, bias)
    red_order = clamp_reduced_order(scheme, red_order)
    return biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, red_order, bias, args...)
end

@inline function _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, args...)
    red_order = compute_face_reduced_order_x(i, j, k, grid, scheme, NoBias())
    return symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, args...)
    red_order = compute_face_reduced_order_y(i, j, k, grid, scheme, NoBias())
    return symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, args...)
    red_order = compute_face_reduced_order_z(i, j, k, grid, scheme, NoBias())
    return symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, args...)
    red_order = compute_center_reduced_order_x(i, j, k, grid, scheme, NoBias())
    return symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, args...)
    red_order = compute_center_reduced_order_y(i, j, k, grid, scheme, NoBias())
    return symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, args...)
    red_order = compute_center_reduced_order_z(i, j, k, grid, scheme, NoBias())
    return symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, red_order, args...)
end
