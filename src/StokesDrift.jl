module StokesDrift

export
    UniformStokesDrift,
    ∂t_uˢ,
    ∂t_vˢ,
    ∂t_wˢ,
    x_curl_Uˢ_cross_U,
    y_curl_Uˢ_cross_U,
    z_curl_Uˢ_cross_U

using Oceananigans.Grids: AbstractGrid

using Oceananigans.Fields
using Oceananigans.Operators

"""
    abstract type AbstractStokesDrift end

Parent type for parameter structs for Stokes drift fields
associated with surface waves.
"""
abstract type AbstractStokesDrift end

#####
##### Functions for "no surface waves"
#####

@inline ∂t_uˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)
@inline ∂t_vˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)
@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, args...) where FT = zero(FT)
@inline y_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, args...) where FT = zero(FT)
@inline z_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, args...) where FT = zero(FT)

#####
##### Uniform surface waves
#####

"""
    UniformStokesDrift{P, UZ, VZ, UT, VT} <: AbstractStokesDrift

Parameter struct for Stokes drift fields associated with surface waves.
"""
struct UniformStokesDrift{UZ, VZ, UT, VT, P} <: AbstractStokesDrift
    ∂z_uˢ :: UZ
    ∂z_vˢ :: VZ
    ∂t_uˢ :: UT
    ∂t_vˢ :: VT
    parameters :: P
end

addzero(args...) = 0

"""
    UniformStokesDrift(; ∂z_uˢ=addzero, ∂z_vˢ=addzero, ∂t_uˢ=addzero, ∂t_vˢ=addzero, parameters=nothing)

Construct a set of functions that describes the Stokes drift field beneath
a uniform surface gravity wave field.
"""
UniformStokesDrift(; ∂z_uˢ=addzero, ∂z_vˢ=addzero, ∂t_uˢ=addzero, ∂t_vˢ=addzero, parameters=nothing) =
    UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ, parameters)

const USD = UniformStokesDrift

@inline ∂t_uˢ(i, j, k, grid, sw::USD, args...) = sw.∂t_uˢ(znode(k, grid, Center()), args...)
@inline ∂t_vˢ(i, j, k, grid, sw::USD, args...) = sw.∂t_vˢ(znode(k, grid, Center()), args...)
@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, sw::USD, args...) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, args...) = @inbounds    ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂z_uˢ(znode(k, grid, Center()), args...)
@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, args...) = @inbounds    ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂z_vˢ(znode(k, grid, Center()), args...)
@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, args...) = @inbounds (- ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂z_uˢ(znode(k, grid, Face()), args...)
                                                                           - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sw.∂z_vˢ(znode(k, grid, Face()), args...) )

stokes_tendecy_functions = (:∂t_uˢ, :∂t_vˢ, :∂t_wˢ)
for func in stokes_tendecy_functions
    @eval $func(i, j, k, grid::AbstractGrid{FT}, sw::USD{<:NamedTuple}, time) where FT = $func(i, j, k, grid, sw, time, sw.parameters)
end

stokes_curl_functions = (:x_curl_Uˢ_cross_U, :y_curl_Uˢ_cross_U, :z_curl_Uˢ_cross_U)
for func in stokes_curl_functions
    @eval $func(i, j, k, grid, sw::USD{<:NamedTuple}, U, time) = $func(i, j, k, grid, sw, U, time, sw.parameters)
end

end # module
