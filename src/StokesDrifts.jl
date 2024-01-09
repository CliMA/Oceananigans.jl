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

Construct a set of functions for a Stokes drift velocity field
corresponding to a horizontally-uniform surface gravity wave field, with optional `parameters`.

If `parameters=nothing`, then the functions `∂z_uˢ`, `∂z_vˢ`, `∂t_uˢ`, `∂t_vˢ` must be callable
with signature `(z, t)`. If `!isnothing(parameters)`, then functions must be callable with
the signature `(z, t, parameters)`.

To resolve the evolution of the Lagrangian-mean momentum, we require vertical-derivatives
and time-derivatives of the horizontal components of the Stokes drift, `uˢ` and `vˢ`.

Example
=======

Exponentially decaying Stokes drift corresponding to a surface Stokes drift of
`uˢ(z=0) = 0.1` and decay scale `h = 20 m`:

```jldoctest
using Oceananigans

@inline uniform_stokes_shear(z, t) = 0.005 * exp(z / 20)

stokes_drift = UniformStokes(∂z_uˢ=uniform_stokes_shear)
```

Exponentially-decaying Stokes drift corresponding to a surface Stokes drift of
`uˢ₀ = 0.1` and decay scale `h = 20 m`, using parameters:

```jldoctest
using Oceananigans

@inline uniform_stokes_shear(z, t, p) = p.uˢ₀ * exp(z / p.h)

stokes_drift_parameters = (uˢ₀ = 0.005, h = 20)
stokes_drift = UniformStokes(∂z_uˢ=uniform_stokes_shear, parameters=stokes_drift_parameters)
```

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

struct StokesDrift{P, VX, WX, UY, WY, UZ, VZ, UT, VT, WT}
    ∂x_vˢ :: VX
    ∂x_wˢ :: WX
    ∂y_uˢ :: UY
    ∂y_wˢ :: WY
    ∂z_uˢ :: UZ
    ∂z_vˢ :: VZ
    ∂t_uˢ :: UT
    ∂t_vˢ :: VT
    ∂t_wˢ :: WT
    parameters :: P
end

"""
    StokesDrift(; ∂z_uˢ=addzero, ∂y_uˢ=addzero, ∂t_uˢ=addzero, 
                  ∂z_vˢ=addzero, ∂x_vˢ=addzero, ∂t_vˢ=addzero, 
                  ∂x_wˢ=addzero, ∂y_wˢ=addzero, ∂t_wˢ=addzero, parameters=nothing)

Construct a set of functions of space and time for a Stokes drift velocity field
corresponding to a surface gravity wave field with an envelope that (potentially) varies
in the horizontal directions.

To resolve the evolution of the Lagrangian-mean momentum, we require all the components
of the "psuedovorticity",

```math
∇ × uˢ = x̂ (∂_y wˢ - ∂_z vˢ) + ŷ (∂_z uˢ - ∂_x wˢ) + ẑ (∂_x vˢ - ∂_y uˢ)
```

as well as time-derivatives of ``uˢ``, ``vˢ``, and ``wˢ``.

Note that each function (e.g., `∂z_uˢ`) is a function of horizontal coordinates and time.
Thus, the correct function signature depends on the grid, since `Flat` horizontal directions
are omitted.

For example, on a grid with `topology = (Periodic, Flat, Bounded)` (and `parameters=nothing`),
then `∂z_uˢ` (for example) should be callable via `∂z_uˢ(x, z, t)`.
When `!isnothing(parameters)`, then in this case `∂z_uˢ` should be callable via `∂z_uˢ(x, z, t, parameters)`.

Similarly, on a grid with `topology = (Periodic, Periodic, Bounded)` and `parameters=nothing`,
`∂z_uˢ` should be callable via `∂z_uˢ(x, y, z, t)`.
"""
function StokesDrift(; ∂x_vˢ = addzero,
                       ∂x_wˢ = addzero,
                       ∂y_uˢ = addzero,
                       ∂y_wˢ = addzero,
                       ∂z_uˢ = addzero,
                       ∂z_vˢ = addzero,
                       ∂t_uˢ = addzero,
                       ∂t_vˢ = addzero,
                       ∂t_wˢ = addzero,
                       parameters = nothing)

    return StokesDrift(∂x_vˢ, ∂x_wˢ, ∂y_uˢ, ∂y_wˢ, ∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ, ∂t_wˢ, parameters)
end

const SD = StokesDrift
const SDnoP = StokesDrift{<:Nothing}

@inline ∂t_uˢ(i, j, k, grid, sw::SD, time) = sw.∂t_uˢ(node(i, j, k, grid, f, c, c)..., time, sw.parameters)
@inline ∂t_vˢ(i, j, k, grid, sw::SD, time) = sw.∂t_vˢ(node(i, j, k, grid, c, f, c)..., time, sw.parameters)
@inline ∂t_wˢ(i, j, k, grid, sw::SD, time) = sw.∂t_wˢ(node(i, j, k, grid, c, c, f)..., time, sw.parameters)

@inline ∂t_uˢ(i, j, k, grid, sw::SDnoP, time) = sw.∂t_uˢ(node(i, j, k, grid, f, c, c)..., time)
@inline ∂t_vˢ(i, j, k, grid, sw::SDnoP, time) = sw.∂t_vˢ(node(i, j, k, grid, c, f, c)..., time)
@inline ∂t_wˢ(i, j, k, grid, sw::SDnoP, time) = sw.∂t_wˢ(node(i, j, k, grid, c, c, f)..., time)

@inline parameters_tuple(sw::SDnoP) = tuple()
@inline parameters_tuple(sw::SD) = tuple(sw.parameters)

@inline function x_curl_Uˢ_cross_U(i, j, k, grid, sw::SD, U, time)
    wᶠᶜᶜ = ℑxzᶠᵃᶜ(i, j, k, grid, U.w) 
    vᶠᶜᶜ = ℑxyᶠᶜᵃ(i, j, k, grid, U.v) 

    pt = parameters_tuple(sw)
    X = node(i, j, k, grid, f, c, c)
    ∂z_uˢ = sw.∂z_uˢ(X..., time, pt...)
    ∂x_wˢ = sw.∂x_wˢ(X..., time, pt...)
    ∂y_uˢ = sw.∂y_uˢ(X..., time, pt...)
    ∂x_vˢ = sw.∂x_vˢ(X..., time, pt...)

    return wᶠᶜᶜ * (∂z_uˢ - ∂x_wˢ) - vᶠᶜᶜ * (∂x_vˢ - ∂y_uˢ)
end


@inline function y_curl_Uˢ_cross_U(i, j, k, grid, sw::SD, U, time)
    wᶜᶠᶜ = ℑyzᵃᶠᶜ(i, j, k, grid, U.w)
    uᶜᶠᶜ = ℑxyᶜᶠᵃ(i, j, k, grid, U.u)

    pt = parameters_tuple(sw)
    X = node(i, j, k, grid, c, f, c)
    ∂z_vˢ = sw.∂z_vˢ(X..., time, pt...)
    ∂y_wˢ = sw.∂y_wˢ(X..., time, pt...)
    ∂x_vˢ = sw.∂x_vˢ(X..., time, pt...)
    ∂y_uˢ = sw.∂y_uˢ(X..., time, pt...)

    return uᶜᶠᶜ * (∂x_vˢ - ∂y_uˢ) - wᶜᶠᶜ * (∂y_wˢ - ∂z_vˢ)
end

@inline function z_curl_Uˢ_cross_U(i, j, k, grid, sw::SD, U, time)
    uᶜᶜᶠ = ℑxzᶜᵃᶠ(i, j, k, grid, U.u)
    vᶜᶜᶠ = ℑyzᵃᶜᶠ(i, j, k, grid, U.v)

    pt = parameters_tuple(sw)
    X = node(i, j, k, grid, c, c, f)
    ∂x_wˢ = sw.∂x_wˢ(X..., time, pt...)
    ∂z_uˢ = sw.∂z_uˢ(X..., time, pt...)
    ∂y_wˢ = sw.∂y_wˢ(X..., time, pt...)
    ∂z_vˢ = sw.∂z_vˢ(X..., time, pt...)

    return vᶜᶜᶠ * (∂y_wˢ - ∂z_vˢ) - uᶜᶜᶠ * (∂z_uˢ - ∂x_wˢ)
end

end # module

