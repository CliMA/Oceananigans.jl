#####
##### Functions for non-rotating models
#####

@inline x_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U) where FT = zero(FT)
@inline y_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U) where FT = zero(FT)
@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U) where FT = zero(FT)

#####
##### The 'FPlane' approximation. This is equivalent to a model with a constant
##### rotation rate around its vertical axis.
#####

"""
    FPlane{FT} <: AbstractRotation

A parameter object for constant rotation around a vertical axis.
"""
struct FPlane{FT} <: AbstractRotation
    f :: FT
end

"""
    FPlane([FT=Float64;] f)

Returns a parameter object for constant rotation at the angular frequency
`f/2`, and therefore with background vorticity `f`, around a vertical axis.

Also called `FPlane`, after the "f-plane" approximation for the local effect of
Earth's rotation in a planar coordinate system tangent to the Earth's surface.
"""
function FPlane(FT=Float64; f)
    return FPlane{FT}(f)
end

"""
    FPlane([FT=Float64;] Ω, latitude)

Returns a parameter object for constant rotation at the angular frequency
`Ωsin(latitude), and therefore with background vorticity `f = 2Ωsin(latitude),
around a vertical axis.

Also called `FPlane`, after the "f-plane" approximation for the local effect of
Earth's rotation in a planar coordinate system tangent to the Earth's surface.
"""
function FPlane(FT=Float64; Ω, latitude)
    return FPlane{FT}(2*Ω*sind(latitude))
end


"""
    BetaPlane{FT} <: AbstractRotation

A parameter object for meridionally increasing Coriolis
parameter (`f = f₀ + βy`).
"""
struct BetaPlane{FT} <: AbstractRotation
    f₀ :: FT
    β  :: FT
end

"""
    BetaPlane([FT=Float64;] f₀, β)

A parameter object for meridionally increasing Coriolis
parameter (`f = f₀ + βy`).
"""
function BetaPlane(FT=Float64; f₀, β)
    return BetaPlane{FT}(f₀, β)
end

"""
    BetaPlane([FT=Float64;] Ω, latitude, R)

Returns a parameter object for meridionally increasing rotation at the
angular frequency `Ωsin(latitude), and therefore with Coriolis parameter
 `f₀ = 2Ωsin(latitude), around a vertical axis.
The Coriolis parameter increases meridionally as `f = f₀ + βy`, where
`β = 2Ωcos(latitude)/R`, and `R` is the radius of the planet.
"""
function BetaPlane(FT=Float64; Ω, latitude, R)
    f₀ = 2*Ω*sind(latitude)
    β = 2*Ω*cosd(latitude)/R
    return BetaPlane{FT}(f₀, β)
end

@inline fv(i, j, k, grid::RegularCartesianGrid{FT}, f, v) where FT =
    FT(0.5) * f * (avgy_f2c(grid, v, i-1,  j, k) + avgy_f2c(grid, v, i, j, k))

@inline fu(i, j, k, grid::RegularCartesianGrid{FT}, f, u) where FT =
    FT(0.5) * f * (avgx_f2c(grid, u, i,  j-1, k) + avgx_f2c(grid, u, i, j, k))

@inbounds @inline fv(i, j, k, grid::RegularCartesianGrid{FT}, f₀, β, v) where FT =
     FT(0.5) * (f₀ + β * grid.yC[j]) * (avgy_f2c(grid, v, i-1,  j, k) + avgy_f2c(grid, v, i, j, k))

@inbounds @inline fu(i, j, k, grid::RegularCartesianGrid{FT}, f₀, β, u) where FT =
     FT(0.5) * (f₀ + β * grid.yF[j]) * (avgx_f2c(grid, u, i,  j-1, k) + avgx_f2c(grid, u, i, j, k))

@inline x_f_cross_U(i, j, k, grid, rotation::FPlane, U) = -fv(i, j, k, grid, rotation.f, U.v)
@inline y_f_cross_U(i, j, k, grid, rotation::FPlane, U) =  fu(i, j, k, grid, rotation.f, U.u)

@inline x_f_cross_U(i, j, k, grid, rotation::BetaPlane, U) = -fv(i, j, k, grid, rotation.f₀, rotation.β, U.v)
@inline y_f_cross_U(i, j, k, grid, rotation::BetaPlane, U) =  fu(i, j, k, grid, rotation.f₀, rotation.β, U.u)

@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Union{FPlane, BetaPlane}, U) where FT = zero(FT)
