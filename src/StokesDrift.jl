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
    UniformStokesDrift(; ∂z_uˢ=addzero, ∂z_vˢ=addzero, ∂t_uˢ=addzero, ∂t_vˢ=addzero)

Construct a set of functions that describes the Stokes drift field beneath
a uniform surface gravity wave field.
"""
UniformStokesDrift(; ∂z_uˢ=addzero, ∂z_vˢ=addzero, ∂t_uˢ=addzero, ∂t_vˢ=addzero) =
    UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ)

# TODO: add docstring 
"""
    UniformStokesDrift(grid)

Construct a set of `Field`s that describe the Stokes drift shear
and tendency beneath a uniform surface gravity wave field.
"""
function UniformStokesDrift(grid::AbstractGrid)
                            
    ∂z_uˢ = Field{Nothing, Nothing, Face}(grid)
    ∂z_vˢ = Field{Nothing, Nothing, Face}(grid)
    ∂t_uˢ = Field{Nothing, Nothing, Center}(grid)
    ∂t_vˢ = Field{Nothing, Nothing, Center}(grid)

    return UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ)
end

const USD = UniformStokesDrift

@inline get_stokes_shearᶜ(i, j, k, grid, ∂z_Uˢ, time) = ∂z_Uˢ(znode(Center(), k, grid), time)
@inline get_stokes_shearᶠ(i, j, k, grid, ∂z_Uˢ, time) = ∂z_Uˢ(znode(Face(), k, grid), time)

@inline get_stokes_shearᶜ(i, j, k, grid, ∂z_Uˢ::AbstractArray, time) = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_Uˢ)
@inline get_stokes_shearᶠ(i, j, k, grid, ∂z_Uˢ::AbstractArray, time) = @inbounds ∂z_Uˢ[i, j, k]

@inline get_stokes_tendencyᶜ(i, j, k, grid, time, ∂t_Uˢ) = ∂t_Uˢ(znode(Center(), k, grid), time)
@inline get_stokes_tendencyᶜ(i, j, k, grid, time, ∂t_Uˢ::AbstractArray) = @inbounds ∂t_Uˢ[i, j, k]

@inline ∂t_uˢ(i, j, k, grid, sw::USD, time) = get_stokes_tendencyᶜ(i, j, k, grid, sw.∂t_uˢ, time)
@inline ∂t_vˢ(i, j, k, grid, sw::USD, time) = get_stokes_tendencyᶜ(i, j, k, grid, sw.∂t_vˢ, time)
@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, sw::USD, time) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) =
    ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * get_stokes_shearᶜ(i, j, k, grid, sw.∂z_uˢ, time)

@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) =
    ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * get_stokes_shearᶜ(i, j, k, grid, sw.∂z_vˢ, time)

@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = (
    - ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * get_stokes_shearᶠ(i, j, k, grid, sw.∂z_uˢ, time)
    - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * get_stokes_shearᶠ(i, j, k, grid, sw.∂z_vˢ, time))

end # module
