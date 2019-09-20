#####
##### Functions for non-rotating models
#####

@inline x_f_cross_U(i, j, k, grid::AbstractGrid{T}, ::Nothing, U) where T = zero(T)
@inline y_f_cross_U(i, j, k, grid::AbstractGrid{T}, ::Nothing, U) where T = zero(T)
@inline z_f_cross_U(i, j, k, grid::AbstractGrid{T}, ::Nothing, U) where T = zero(T)

#####
##### The 'FPlane' approximation. This is equivalent to a model with a constant
##### rotation rate around its vertical axis.
#####

"""
    FPlane{T} <: AbstractRotation

A parameter object for constant rotation around a vertical axis.
"""
struct FPlane{T} <: AbstractRotation
    f :: T
end

"""
    FPlane([T=Float64;] f)

Returns a parameter object for constant rotation at the angular frequency
`2f`, and therefore with background vorticity `f`, around a vertical axis.

Also called `FPlane`, after the "f-plane" approximation for the local effect of 
Earth's rotation in a planar coordinate system tangent to the Earth's surface.
"""
function FPlane(T=Float64; f) 
    return FPlane{T}(f)
end

@inline fv(i, j, k, grid::RegularCartesianGrid{T}, f, v) where T = 
    T(0.5) * f * (avgy_f2c(grid, v, i-1,  j, k) + avgy_f2c(grid, v, i, j, k))

@inline fu(i, j, k, grid::RegularCartesianGrid{T}, f, u) where T = 
    T(0.5) * f * (avgx_f2c(grid, u, i,  j-1, k) + avgx_f2c(grid, u, i, j, k))

@inline x_f_cross_U(i, j, k, grid, rotation::FPlane, U) = -fv(i, j, k, grid, rotation.f, U.v)
@inline y_f_cross_U(i, j, k, grid, rotation::FPlane, U) =  fu(i, j, k, grid, rotation.f, U.u)
@inline z_f_cross_U(i, j, k, grid::AbstractGrid{T}, ::FPlane, U) where T = zero(T)

