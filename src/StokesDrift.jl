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
    UniformStokesDrift{P, UZ, VZ, UT, VT} <: AbstractStokesDrift

Parameter struct for Stokes drift fields associated with surface waves.
"""
struct UniformStokesDrift{P, UZ, VZ, UT, VT} <: AbstractStokesDrift
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
a horizontally-uniform surface gravity wave field.
"""
UniformStokesDrift(; ∂z_uˢ=addzero, ∂z_vˢ=addzero, ∂t_uˢ=addzero, ∂t_vˢ=addzero, parameters=nothing) =
    UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ, parameters)

const USD = UniformStokesDrift
const USDnoP = UniformStokesDrift{<:Nothing}

@inline ∂t_uˢ(i, j, k, grid, sw::USD, time) = sw.∂t_uˢ(znode(k, grid, Center()), time, sw.parameters)
@inline ∂t_vˢ(i, j, k, grid, sw::USD, time) = sw.∂t_vˢ(znode(k, grid, Center()), time, sw.parameters)
@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, sw::USD, time) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = @inbounds    ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂z_uˢ(znode(k, grid, Center()), time, sw.parameters)
@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = @inbounds    ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂z_vˢ(znode(k, grid, Center()), time, sw.parameters)
@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = @inbounds (- ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂z_uˢ(znode(k, grid, Face()), time, sw.parameters)
                                                                        - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sw.∂z_vˢ(znode(k, grid, Face()), time, sw.parameters) )

# Methods for when `parameters == nothing`
@inline ∂t_uˢ(i, j, k, grid, sw::USDnoP, time) = sw.∂t_uˢ(znode(k, grid, Center()), time)
@inline ∂t_vˢ(i, j, k, grid, sw::USDnoP, time) = sw.∂t_vˢ(znode(k, grid, Center()), time)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::USDnoP, U, time) = @inbounds    ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂z_uˢ(znode(k, grid, Center()), time)
@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::USDnoP, U, time) = @inbounds    ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂z_vˢ(znode(k, grid, Center()), time)
@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::USDnoP, U, time) = @inbounds (- ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂z_uˢ(znode(k, grid, Face()), time)
                                                                           - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sw.∂z_vˢ(znode(k, grid, Face()), time) )

end # module
