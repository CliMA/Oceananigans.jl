using .TurbulenceClosures: ▶xy_cfa, ▶xy_fca

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
function FPlane(T::DataType=Float64; f) 
    return FPlane{T}(f)
end

@inline fv(i, j, k, grid::RegularCartesianGrid{T}, f, v) where T = 
    T(0.5) * f * (avgy_f2c(grid, v, i-1,  j, k) + avgy_f2c(grid, v, i, j, k))

@inline fu(i, j, k, grid::RegularCartesianGrid{T}, f, u) where T = 
    T(0.5) * f * (avgx_f2c(grid, u, i,  j-1, k) + avgx_f2c(grid, u, i, j, k))

@inbounds @inline fv(i, j, k, grid::RegularCartesianGrid{T}, f₀, β, v) where T =

@inbounds @inline fu(i, j, k, grid::RegularCartesianGrid{T}, f₀, β, u) where T =
     T(0.5) * (f₀ + β * grid.yF[j]) * (avgx_f2c(grid, u, i,  j-1, k) + avgx_f2c(grid, u, i, j, k))

@inline x_f_cross_U(i, j, k, grid, coriolis::FPlane, U) = -fv(i, j, k, grid, coriolis.f, U.v)
@inline y_f_cross_U(i, j, k, grid, coriolis::FPlane, U) =  fu(i, j, k, grid, coriolis.f, U.u)
@inline z_f_cross_U(i, j, k, grid::AbstractGrid{T}, ::FPlane, U) where T = zero(T)

#####
##### The Beta Plane
#####

"""
    BetaPlane{T} <: AbstractRotation

A parameter object for meridionally increasing Coriolis
parameter (`f = f₀ + βy`).
"""
struct BetaPlane{T} <: AbstractRotation
    f₀ :: T
     β :: T
end

"""
    BetaPlane([T=Float64;] f₀=nothing, β=nothing, 
                           rotation_rate=nothing, latitude=nothing, radius=nothing)

A parameter object for meridionally increasing Coriolis parameter (`f = f₀ + βy`).

The user may specify both `f₀` and `β`, or the three parameters
`rotation_rate`, `latitude`, and `radius` that specify the rotation rate and radius 
of a planet, and the central latitude at which the `β`-plane approximation is to be made.
"""
function BetaPlane(T=Float64; f₀=nothing, β=nothing, 
                              rotation_rate=nothing, latitude=nothing, radius=nothing)

    f_and_β = f₀ != nothing && β != nothing
    planet_parameters = rotation_rate != nothing && latitude != nothing && radius != nothing

    (f_and_β && all(Tuple(p === nothing for p in (rotation_rate, latitude, radius)))) || 
    (planet_parameters && all(Tuple(p === nothing for p in (f₀, β)))) ||
        throw(ArgumentError("Either both keywords f₀ and β must be specified, 
                            *or* all of rotation_rate, latitude, and radius."))

    if planet_parameters
        f₀ = 2rotation_rate * sind(latitude)
         β = 2rotation_rate * cosd(latitude) / radius
     end

    return BetaPlane{T}(f₀, β)
end

@inline x_f_cross_U(i, j, k, grid, coriolis::BetaPlane, U) =
    @inbounds (coriolis.f₀ + coriolis.β * grid.yC[j]) * ▶xy_fca(i, j, k, grid, U.v)

@inline y_f_cross_U(i, j, k, grid, coriolis::BetaPlane, U) =
    @inbounds (coriolis.f₀ + coriolis.β * grid.yF[j]) * ▶xy_cfa(i, j, k, grid, U.v) 

@inline z_f_cross_U(i, j, k, grid::AbstractGrid{T}, ::BetaPlane, U) where T = zero(T)
