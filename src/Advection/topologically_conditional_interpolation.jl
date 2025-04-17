#####
##### This file provides functions that conditionally-evaluate interpolation operators
##### near boundaries in bounded directions.
#####
##### For example, the function _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c) either
#####
#####     1. Always returns symmetric_interpolate_xᶠᵃᵃ if the x-direction is Periodic; or
#####
#####     2. Returns symmetric_interpolate_xᶠᵃᵃ if the x-direction is Bounded and index i is not
#####        close to the boundary, or a second-order interpolation if i is close to a boundary.
#####

using Oceananigans.Grids: AbstractUnderlyingGrid, 
                          Bounded, 
                          RightConnected, 
                          LeftConnected, 
                          topology,
                          architecture

const AUG = AbstractUnderlyingGrid

# topologies bounded at least on one side
const BT = Union{Bounded, RightConnected, LeftConnected}

# Bounded underlying Grids
const AUGX   = AUG{<:Any, <:BT}
const AUGY   = AUG{<:Any, <:Any, <:BT}
const AUGZ   = AUG{<:Any, <:Any, <:Any, <:BT}
const AUGXY  = AUG{<:Any, <:BT, <:BT}
const AUGXZ  = AUG{<:Any, <:BT, <:Any, <:BT}
const AUGYZ  = AUG{<:Any, <:Any, <:BT, <:BT}
const AUGXYZ = AUG{<:Any, <:BT, <:BT, <:BT}

# Bounded underlying grids of all types
const AUGB = Union{AUGX, AUGY, AUGZ, AUGXY, AUGXZ, AUGYZ, AUGXYZ}

@inline reduced_order(i, ::Type{RightConnected}, N, B) = min(B, i)
@inline reduced_order(i, ::Type{LeftConnected},  N, B) = min(B, N-i)
@inline reduced_order(i, ::Type{Bounded},        N, B) = min(B, i, N-i)

const A{B} = AbstractAdvectionScheme{B} 

# Fallback for periodic underlying grids
@inline compute_face_reduced_order_x(i, j, k, grid, ::A{B}) where B = B
@inline compute_face_reduced_order_y(i, j, k, grid, ::A{B}) where B = B
@inline compute_face_reduced_order_z(i, j, k, grid, ::A{B}) where B = B

# Fallback for periodic underlying grids
@inline compute_center_reduced_order_x(i, j, k, grid, ::A{B}) where B = B
@inline compute_center_reduced_order_y(i, j, k, grid, ::A{B}) where B = B
@inline compute_center_reduced_order_z(i, j, k, grid, ::A{B}) where B = B

# Fallback for lower order advection on bounded grids
@inline compute_face_reduced_order_x(i, j, k, ::AUGB, ::A{1}) = 1
@inline compute_face_reduced_order_y(i, j, k, ::AUGB, ::A{1}) = 1
@inline compute_face_reduced_order_z(i, j, k, ::AUGB, ::A{1}) = 1

# Fallback for lower order advection on bounded grids
@inline compute_center_reduced_order_x(i, j, k, ::AUGB, ::A{1}) = 1
@inline compute_center_reduced_order_y(i, j, k, ::AUGB, ::A{1}) = 1
@inline compute_center_reduced_order_z(i, j, k, ::AUGB, ::A{1}) = 1

# Bounded grids
@inline compute_face_reduced_order_x(i, j, k, grid::AUGX, ::A{B}) where B = reduced_order(i, topology(grid, 1), size(grid, 1), B)
@inline compute_face_reduced_order_y(i, j, k, grid::AUGY, ::A{B}) where B = reduced_order(j, topology(grid, 2), size(grid, 2), B)
@inline compute_face_reduced_order_z(i, j, k, grid::AUGZ, ::A{B}) where B = reduced_order(k, topology(grid, 3), size(grid, 3), B)

# Fallback for periodic underlying grids
@inline compute_center_reduced_order_x(i, j, k, grid::AUGX, ::A{B}) where B = reduced_order(i, topology(grid, 1), size(grid, 1), B)
@inline compute_center_reduced_order_y(i, j, k, grid::AUGY, ::A{B}) where B = reduced_order(j, topology(grid, 2), size(grid, 2), B)
@inline compute_center_reduced_order_z(i, j, k, grid::AUGZ, ::A{B}) where B = reduced_order(k, topology(grid, 3), size(grid, 3), B)

@inline function _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, args...)
    R = compute_face_reduced_order_x(i, j, k, grid, scheme)
    return biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, Val(R), args...)
end

@inline function _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, args...) 
    R = compute_face_reduced_order_y(i, j, k, grid, scheme)
    return biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, Val(R), args...)
end

@inline function _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, args...)
    R = compute_face_reduced_order_z(i, j, k, grid, scheme)
    return biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, Val(R), args...)
end

@inline function _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, args...)
    R = compute_center_reduced_order_x(i, j, k, grid, scheme)
    return biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, Val(R), args...)
end

@inline function _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, args...) 
    R = compute_center_reduced_order_y(i, j, k, grid, scheme)
    return biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, Val(R), args...)
end

@inline function _biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, args...) 
    R = compute_center_reduced_order_z(i, j, k, grid, scheme)
    return biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, Val(R), args...)
end

@inline function _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, args...)
    R = compute_face_reduced_order_x(i, j, k, grid, scheme)
    return symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, Val(R), args...)
end

@inline function _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, args...)
    R = compute_face_reduced_order_y(i, j, k, grid, scheme)
    return symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, Val(R), args...)
end

@inline function _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, args...)
    R = compute_face_reduced_order_z(i, j, k, grid, scheme)
    return symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, Val(R), args...)
end

@inline function _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, args...)
    R = compute_center_reduced_order_x(i, j, k, grid, scheme)
    return symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, Val(R), args...)
end

@inline function _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, args...)
    R = compute_center_reduced_order_y(i, j, k, grid, scheme)
    return symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, Val(R), args...)
end

@inline function _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, args...)
    R = compute_center_reduced_order_z(i, j, k, grid, scheme)
    return symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, Val(R), args...)
end