module StokesDrifts

export
    UniformStokesDrift,
    StokesDrift,
    ∂t_uˢ,
    ∂t_vˢ,
    ∂t_wˢ,
    x_curl_Uˢ_cross_U,
    y_curl_Uˢ_cross_U,
    z_curl_Uˢ_cross_U

using Oceananigans.Grids: AbstractGrid, node

using Oceananigans.Fields
using Oceananigans.Operators

#####
##### Functions for "no surface waves"
#####

@inline ∂t_uˢ(i, j, k, grid, ::Nothing, time) = zero(grid)
@inline ∂t_vˢ(i, j, k, grid, ::Nothing, time) = zero(grid)
@inline ∂t_wˢ(i, j, k, grid, ::Nothing, time) = zero(grid)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, ::Nothing, U, time) = zero(grid)
@inline y_curl_Uˢ_cross_U(i, j, k, grid, ::Nothing, U, time) = zero(grid)
@inline z_curl_Uˢ_cross_U(i, j, k, grid, ::Nothing, U, time) = zero(grid)

#####
##### Uniform surface waves
#####

struct UniformStokesDrift{P, UZ, VZ, UT, VT}
    ∂z_uˢ :: UZ
    ∂z_vˢ :: VZ
    ∂t_uˢ :: UT
    ∂t_vˢ :: VT
    parameters :: P
end

@inline addzero(args...) = 0

"""
    UniformStokesDrift(; ∂z_uˢ=addzero, ∂z_vˢ=addzero, ∂t_uˢ=addzero, ∂t_vˢ=addzero, parameters=nothing)

Construct a set of functions of `(z, t)` that describes the Stokes drift field beneath
a _horizontally-uniform_ surface gravity wave field.
"""
UniformStokesDrift(; ∂z_uˢ=addzero, ∂z_vˢ=addzero, ∂t_uˢ=addzero, ∂t_vˢ=addzero, parameters=nothing) =
    UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ, parameters)

const USD = UniformStokesDrift
const USDnoP = UniformStokesDrift{<:Nothing}
const f = Face()
const c = Center()

@inline ∂t_uˢ(i, j, k, grid, sw::USD, time) = sw.∂t_uˢ(znode(k, grid, c), time, sw.parameters)
@inline ∂t_vˢ(i, j, k, grid, sw::USD, time) = sw.∂t_vˢ(znode(k, grid, c), time, sw.parameters)
@inline ∂t_wˢ(i, j, k, grid, sw::USD, time) = zero(grid)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = @inbounds    ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂z_uˢ(znode(k, grid, c), time, sw.parameters)
@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = @inbounds    ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂z_vˢ(znode(k, grid, c), time, sw.parameters)

@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = @inbounds (- ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂z_uˢ(znode(k, grid, f), time, sw.parameters)
                                                                        - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sw.∂z_vˢ(znode(k, grid, f), time, sw.parameters) )

# Methods for when `parameters == nothing`
@inline ∂t_uˢ(i, j, k, grid, sw::USDnoP, time) = sw.∂t_uˢ(znode(k, grid, c), time)
@inline ∂t_vˢ(i, j, k, grid, sw::USDnoP, time) = sw.∂t_vˢ(znode(k, grid, c), time)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::USDnoP, U, time) = @inbounds    ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂z_uˢ(znode(k, grid, c), time)
@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::USDnoP, U, time) = @inbounds    ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂z_vˢ(znode(k, grid, c), time)
@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::USDnoP, U, time) = @inbounds (- ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂z_uˢ(znode(k, grid, f), time)
                                                                           - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sw.∂z_vˢ(znode(k, grid, f), time))

struct StokesDrift{P, UX, VX, WX, UY, VY, WY, UZ, VZ, WZ, UT, VT, WT}
    ∂x_uˢ :: UX
    ∂x_vˢ :: VX
    ∂x_wˢ :: WX
    ∂y_uˢ :: UY
    ∂y_vˢ :: VY
    ∂y_wˢ :: WY
    ∂z_uˢ :: UZ
    ∂z_vˢ :: VZ
    ∂z_wˢ :: WZ
    ∂t_uˢ :: UT
    ∂t_vˢ :: VT
    ∂t_wˢ :: WT
    parameters :: P
end

function StokesDrift(; ∂x_uˢ = addzero,
                       ∂x_vˢ = addzero,
                       ∂x_wˢ = addzero,
                       ∂y_uˢ = addzero,
                       ∂y_vˢ = addzero,
                       ∂y_wˢ = addzero,
                       ∂z_uˢ = addzero,
                       ∂z_vˢ = addzero,
                       ∂z_wˢ = addzero,
                       ∂t_uˢ = addzero,
                       ∂t_vˢ = addzero,
                       ∂t_wˢ = addzero,
                       parameters = nothing)

    return StokesDrift(∂x_uˢ, ∂x_vˢ, ∂x_wˢ, ∂y_uˢ, ∂y_vˢ, ∂y_wˢ, ∂z_uˢ, ∂z_vˢ, ∂z_wˢ, ∂t_uˢ, ∂t_vˢ, ∂t_wˢ, parameters)
end

const SD = StokesDrift
const SDnoP = StokesDrift{<:Nothing}

@inline ∂t_uˢ(i, j, k, grid, sw::SD, time) = sw.∂t_uˢ(node(i, j, k, grid, f, c, c)..., time, sw.parameters)
@inline ∂t_vˢ(i, j, k, grid, sw::SD, time) = sw.∂t_vˢ(node(i, j, k, grid, c, f, c)..., time, sw.parameters)
@inline ∂t_wˢ(i, j, k, grid, sw::SD, time) = sw.∂t_wˢ(node(i, j, k, grid, c, c, f)..., time, sw.parameters)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::SD, U, time) = (  ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂z_uˢ(node(i, j, k, grid, f, c, c)..., time, sw.parameters)
                                                             + ℑxyᶠᶜᵃ(i, j, k, grid, U.v) * sw.∂y_uˢ(node(i, j, k, grid, f, c, c)..., time, sw.parameters)
                                                             - ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂x_wˢ(node(i, j, k, grid, f, c, c)..., time, sw.parameters)
                                                             - ℑxyᶠᶜᵃ(i, j, k, grid, U.v) * sw.∂x_vˢ(node(i, j, k, grid, f, c, c)..., time, sw.parameters))

@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::SD, U, time) = (  ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂z_vˢ(node(i, j, k, grid, c, f, c)..., time, sw.parameters)
                                                             + ℑxyᶜᶠᵃ(i, j, k, grid, U.u) * sw.∂x_vˢ(node(i, j, k, grid, c, f, c)..., time, sw.parameters)
                                                             - ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂y_wˢ(node(i, j, k, grid, c, f, c)..., time, sw.parameters)
                                                             - ℑxyᶜᶠᵃ(i, j, k, grid, U.u) * sw.∂y_uˢ(node(i, j, k, grid, c, f, c)..., time, sw.parameters))

@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::SD, U, time) = (  ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂x_wˢ(node(i, j, k, grid, c, c, f)..., time, sw.parameters)
                                                             + ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sw.∂y_wˢ(node(i, j, k, grid, c, c, f)..., time, sw.parameters)
                                                             - ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂z_uˢ(node(i, j, k, grid, c, c, f)..., time, sw.parameters)
                                                             - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sw.∂z_vˢ(node(i, j, k, grid, c, c, f)..., time, sw.parameters))


@inline ∂t_uˢ(i, j, k, grid, sw::SDnoP, time) = sw.∂t_uˢ(node(i, j, k, grid, f, c, c)..., time)
@inline ∂t_vˢ(i, j, k, grid, sw::SDnoP, time) = sw.∂t_vˢ(node(i, j, k, grid, c, f, c)..., time)
@inline ∂t_wˢ(i, j, k, grid, sw::SDnoP, time) = sw.∂t_wˢ(node(i, j, k, grid, c, c, f)..., time)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::SDnoP, U, time) = (  ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂z_uˢ(node(i, j, k, grid, f, c, c)..., time)
                                                                + ℑxyᶠᶜᵃ(i, j, k, grid, U.v) * sw.∂y_uˢ(node(i, j, k, grid, f, c, c)..., time)
                                                                - ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂x_wˢ(node(i, j, k, grid, f, c, c)..., time)
                                                                - ℑxyᶠᶜᵃ(i, j, k, grid, U.v) * sw.∂x_vˢ(node(i, j, k, grid, f, c, c)..., time))

@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::SDnoP, U, time) = (  ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂z_vˢ(node(i, j, k, grid, c, f, c)..., time)
                                                                + ℑxyᶜᶠᵃ(i, j, k, grid, U.u) * sw.∂x_vˢ(node(i, j, k, grid, c, f, c)..., time)
                                                                - ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂y_wˢ(node(i, j, k, grid, c, f, c)..., time)
                                                                - ℑxyᶜᶠᵃ(i, j, k, grid, U.u) * sw.∂y_uˢ(node(i, j, k, grid, c, f, c)..., time))

@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::SDnoP, U, time) = (  ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂x_wˢ(node(i, j, k, grid, c, c, f)..., time)
                                                                + ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sw.∂y_wˢ(node(i, j, k, grid, c, c, f)..., time)
                                                                - ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂z_uˢ(node(i, j, k, grid, c, c, f)..., time)
                                                                - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sw.∂z_vˢ(node(i, j, k, grid, c, c, f)..., time))

end # module

