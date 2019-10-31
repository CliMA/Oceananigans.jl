module SurfaceWaves

export
    ∂t_uˢ,
    ∂t_vˢ,
    ∂t_wˢ,
    x_curl_Uˢ_cross_U,
    y_curl_Uˢ_cross_U,
    z_curl_Uˢ_cross_U

using Oceananigans: AbstractGrid, Face, Cell, xnode, ynode, znode

"""
    abstract type AbstractStokesDrift end

Parent type for parameter structs for Stokes drift fields
associated with surface waves
"""
abstract type AbstractStokesDrift end

#####
##### Functions for "no surface waves"
#####

@inline ∂t_uˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, time) where FT = zero(FT)
@inline ∂t_vˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, time) where FT = zero(FT)
@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, time) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, time) where FT = zero(FT)
@inline y_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, time) where FT = zero(FT)
@inline z_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, time) where FT = zero(FT)

#####
##### Uniform surface waves
#####

"""
    UniformStokesDrift{UZ, VZ, UT, VT} <: AbstractStokesDrift

Parameter struct for Stokes drift fields associated with surface waves.
"""
struct UniformStokesDrift{UZ, VZ, UT, VT} <: AbstractStokesDrift
    ∂z_uˢ :: UZ
    ∂z_vˢ :: VZ
    ∂t_uˢ :: UT
    ∂t_vˢ :: VT
end

addzero(args...) = 0

"""
    UniformStokesDrift(; ∂z_uˢ=addzero, ∂z_vˢ=addzero, 
                                    ∂t_uˢ=addzero, ∂t_vˢ=addzero)

Construct a set of functions that describes the Stokes drift field beneath
a uniform surface gravity wave field. 
"""
UniformStokesDrift(; ∂z_uˢ=addzero, ∂z_vˢ=addzero, ∂t_uˢ=addzero, ∂t_vˢ=addzero) =
    UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ)

const USD = UniformStokesDrift

@inline ∂t_uˢ(i, j, k, grid, sw::USD, time) = sw.∂t_uˢ(znode(Cell, k, grid), time)
@inline ∂t_vˢ(i, j, k, grid, sw::USD, time) = sw.∂t_vˢ(znode(Cell, k, grid), time)

@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, sw::USD, time) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = @inbounds U.w[i, j, k] * sw.∂z_uˢ(znode(Cell, k, grid), time)
@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = @inbounds U.w[i, j, k] * sw.∂z_vˢ(znode(Cell, k, grid), time)

@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = @inbounds begin (
    - U.u[i, j, k] * sw.∂z_uˢ(znode(Face, k, grid), time)
    - U.v[i, j, k] * sw.∂z_vˢ(znode(Face, k, grid), time) )
end

end # module
