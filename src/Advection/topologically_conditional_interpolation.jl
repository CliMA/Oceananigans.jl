#####
##### This file provides functions that conditionally-evaluate interpolation operators
##### near boundaries in bounded directions
#####

using Oceananigans.Grids: AbstractGrid,
                          Bounded,
                          RightConnected,
                          LeftConnected,
                          topology,
                          architecture

const AG = AbstractGrid

# topologies bounded at least on one side
const BT = Union{Bounded, RightConnected, LeftConnected}

# Bounded Grids
const AGX   = AG{<:Any, <:BT}
const AGY   = AG{<:Any, <:Any, <:BT}
const AGZ   = AG{<:Any, <:Any, <:Any, <:BT}
const AGXY  = AG{<:Any, <:BT,  <:BT}
const AGXZ  = AG{<:Any, <:BT,  <:Any, <:BT}
const AGYZ  = AG{<:Any, <:Any, <:BT,  <:BT}
const AGXYZ = AG{<:Any, <:BT,  <:BT,  <:BT}

# Reduction of the order near boundaries
#
# For faces reconstructions (tracers for example):
#               B                                                           B
#  cells:   --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
#  order:       1     2     3     4   ....        ....    4     3     2     1
#
# For center reconstructions (vorticity for example):
#               B                                                           B
#  cells:   --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
#  order:    1     1     2     3    ...               ...    3     2     1     1

@inline reduced_order(i, ::Type{RightConnected}, N, B) = max(1, min(B, i))
@inline reduced_order(i, ::Type{LeftConnected},  N, B) = max(1, min(B, N+1-i))
@inline reduced_order(i, ::Type{Bounded},        N, B) = max(1, min(B, i, N+1-i))

const A{B} = AbstractAdvectionScheme{B} 

# Fallback for periodic underlying grids
@inline compute_face_reduced_order_x(i, j, k, grid, ::A{B}) where B = B
@inline compute_face_reduced_order_y(i, j, k, grid, ::A{B}) where B = B
@inline compute_face_reduced_order_z(i, j, k, grid, ::A{B}) where B = B

# Fallback for periodic underlying grids
@inline compute_center_reduced_order_x(i, j, k, grid, ::A{B}) where B = B
@inline compute_center_reduced_order_y(i, j, k, grid, ::A{B}) where B = B
@inline compute_center_reduced_order_z(i, j, k, grid, ::A{B}) where B = B

# Bounded grids
@inline compute_face_reduced_order_x(i, j, k, grid::AGX, ::A{B}) where B = reduced_order(i, topology(grid, 1), size(grid, 1), B)
@inline compute_face_reduced_order_y(i, j, k, grid::AGY, ::A{B}) where B = reduced_order(j, topology(grid, 2), size(grid, 2), B)
@inline compute_face_reduced_order_z(i, j, k, grid::AGZ, ::A{B}) where B = reduced_order(k, topology(grid, 3), size(grid, 3), B)

# Fallback for periodic underlying grids
@inline compute_center_reduced_order_x(i, j, k, grid::AGX, ::A{B}) where B = reduced_order(i, topology(grid, 1), size(grid, 1), B)
@inline compute_center_reduced_order_y(i, j, k, grid::AGY, ::A{B}) where B = reduced_order(j, topology(grid, 2), size(grid, 2), B)
@inline compute_center_reduced_order_z(i, j, k, grid::AGZ, ::A{B}) where B = reduced_order(k, topology(grid, 3), size(grid, 3), B)

@inline function _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, args...)
    red_order = compute_face_reduced_order_x(i, j, k, grid, scheme)
    return biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, args...) 
    red_order = compute_face_reduced_order_y(i, j, k, grid, scheme)
    return biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, args...)
    red_order = compute_face_reduced_order_z(i, j, k, grid, scheme)
    return biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, args...)
    red_order = compute_center_reduced_order_x(i, j, k, grid, scheme)
    return biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, args...) 
    red_order = compute_center_reduced_order_y(i, j, k, grid, scheme)
    return biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, args...) 
    red_order = compute_center_reduced_order_z(i, j, k, grid, scheme)
    return biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, args...)
    red_order = compute_face_reduced_order_x(i, j, k, grid, scheme)
    return symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, args...)
    red_order = compute_face_reduced_order_y(i, j, k, grid, scheme)
    return symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, args...)
    red_order = compute_face_reduced_order_z(i, j, k, grid, scheme)
    return symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, args...)
    red_order = compute_center_reduced_order_x(i, j, k, grid, scheme)
    return symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, args...)
    red_order = compute_center_reduced_order_y(i, j, k, grid, scheme)
    return symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, red_order, args...)
end

@inline function _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, args...)
    red_order = compute_center_reduced_order_z(i, j, k, grid, scheme)
    return symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, red_order, args...)
end
