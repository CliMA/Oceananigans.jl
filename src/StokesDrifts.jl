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

using Adapt: adapt

using Oceananigans.Fields
using Oceananigans.Operators

using Oceananigans.Grids: AbstractGrid, node
using Oceananigans.Utils: prettysummary

import Adapt: adapt_structure

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

adapt_structure(to, sd::UniformStokesDrift) = UniformStokesDrift(adapt(to, sd.∂z_uˢ),
                                                                 adapt(to, sd.∂z_vˢ),
                                                                 adapt(to, sd.∂t_uˢ),
                                                                 adapt(to, sd.∂t_vˢ),
                                                                 adapt(to, sd.parameters))

Base.summary(::UniformStokesDrift{Nothing}) = "UniformStokesDrift{Nothing}"

function Base.summary(usd::UniformStokesDrift)
    p_str = prettysummary(usd.parameters)
    return "UniformStokesDrift with parameters $p_str"
end

function Base.show(io::IO, usd::UniformStokesDrift)
    print(io, summary(usd), ':', '\n')
    print(io, "├── ∂z_uˢ: ", prettysummary(usd.∂z_uˢ, false), '\n')
    print(io, "├── ∂z_vˢ: ", prettysummary(usd.∂z_vˢ, false), '\n')
    print(io, "├── ∂t_uˢ: ", prettysummary(usd.∂t_uˢ, false), '\n')
    print(io, "└── ∂t_vˢ: ", prettysummary(usd.∂t_vˢ, false))
end

@inline zerofunction(args...) = 0

"""
    UniformStokesDrift(; ∂z_uˢ=zerofunction, ∂z_vˢ=zerofunction, ∂t_uˢ=zerofunction, ∂t_vˢ=zerofunction, parameters=nothing)

Construct a set of functions for a Stokes drift velocity field
corresponding to a horizontally-uniform surface gravity wave field, with optional `parameters`.

If `parameters=nothing`, then the functions `∂z_uˢ`, `∂z_vˢ`, `∂t_uˢ`, `∂t_vˢ` must be callable
with signature `(z, t)`. If `!isnothing(parameters)`, then functions must be callable with
the signature `(z, t, parameters)`.

To resolve the evolution of the Lagrangian-mean momentum, we require vertical-derivatives
and time-derivatives of the horizontal components of the Stokes drift, `uˢ` and `vˢ`.

Examples
========

Exponentially decaying Stokes drift corresponding to a surface Stokes drift of
`uˢ(z=0) = 0.005` and decay scale `h = 20`:

```jldoctest
using Oceananigans

@inline uniform_stokes_shear(z, t) = 0.005 * exp(z / 20)

stokes_drift = UniformStokesDrift(∂z_uˢ=uniform_stokes_shear)

# output

UniformStokesDrift{Nothing}:
├── ∂z_uˢ: uniform_stokes_shear
├── ∂z_vˢ: zerofunction
├── ∂t_uˢ: zerofunction
└── ∂t_vˢ: zerofunction
```

Exponentially-decaying Stokes drift corresponding to a surface Stokes drift of
`uˢ = 0.005` and decay scale `h = 20`, using parameters:

```jldoctest
using Oceananigans

@inline uniform_stokes_shear(z, t, p) = p.uˢ * exp(z / p.h)

stokes_drift_parameters = (uˢ = 0.005, h = 20)
stokes_drift = UniformStokesDrift(∂z_uˢ=uniform_stokes_shear, parameters=stokes_drift_parameters)

# output

UniformStokesDrift with parameters (uˢ=0.005, h=20):
├── ∂z_uˢ: uniform_stokes_shear
├── ∂z_vˢ: zerofunction
├── ∂t_uˢ: zerofunction
└── ∂t_vˢ: zerofunction
```
"""
UniformStokesDrift(; ∂z_uˢ=zerofunction, ∂z_vˢ=zerofunction, ∂t_uˢ=zerofunction, ∂t_vˢ=zerofunction, parameters=nothing) =
    UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ, parameters)

const USD = UniformStokesDrift
const USDnoP = UniformStokesDrift{<:Nothing}
const f = Face()
const c = Center()

@inline ∂t_uˢ(i, j, k, grid, sw::USD, time) = sw.∂t_uˢ(znode(k, grid, c), time, sw.parameters)
@inline ∂t_vˢ(i, j, k, grid, sw::USD, time) = sw.∂t_vˢ(znode(k, grid, c), time, sw.parameters)
@inline ∂t_wˢ(i, j, k, grid, sw::USD, time) = zero(grid)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) =    ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂z_uˢ(znode(k, grid, c), time, sw.parameters)
@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) =    ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂z_vˢ(znode(k, grid, c), time, sw.parameters)
@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = (- ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂z_uˢ(znode(k, grid, f), time, sw.parameters)
                                                              - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sw.∂z_vˢ(znode(k, grid, f), time, sw.parameters))

# Methods for when `parameters == nothing`
@inline ∂t_uˢ(i, j, k, grid, sw::USDnoP, time) = sw.∂t_uˢ(znode(k, grid, c), time)
@inline ∂t_vˢ(i, j, k, grid, sw::USDnoP, time) = sw.∂t_vˢ(znode(k, grid, c), time)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::USDnoP, U, time) =    ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂z_uˢ(znode(k, grid, c), time)
@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::USDnoP, U, time) =    ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂z_vˢ(znode(k, grid, c), time)
@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::USDnoP, U, time) = (- ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂z_uˢ(znode(k, grid, f), time)
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

adapt_structure(to, sd::StokesDrift) = StokesDrift(adapt(to, sd.∂x_vˢ),
                                                   adapt(to, sd.∂x_wˢ),
                                                   adapt(to, sd.∂y_uˢ),
                                                   adapt(to, sd.∂y_wˢ),
                                                   adapt(to, sd.∂z_uˢ),
                                                   adapt(to, sd.∂z_vˢ),
                                                   adapt(to, sd.∂t_uˢ),
                                                   adapt(to, sd.∂t_vˢ),
                                                   adapt(to, sd.∂t_wˢ),
                                                   adapt(to, sd.parameters))

Base.summary(::StokesDrift{Nothing}) = "StokesDrift{Nothing}"

function Base.summary(sd::StokesDrift)
    p_str = prettysummary(sd.parameters)
    return "StokesDrift with parameters $p_str"
end

function Base.show(io::IO, sd::StokesDrift)
    print(io, summary(sd), ':', '\n')
    print(io, "├── ∂x_vˢ: ", prettysummary(sd.∂x_vˢ, false), '\n')
    print(io, "├── ∂x_wˢ: ", prettysummary(sd.∂x_wˢ, false), '\n')
    print(io, "├── ∂y_uˢ: ", prettysummary(sd.∂y_uˢ, false), '\n')
    print(io, "├── ∂y_wˢ: ", prettysummary(sd.∂y_wˢ, false), '\n')
    print(io, "├── ∂z_uˢ: ", prettysummary(sd.∂z_uˢ, false), '\n')
    print(io, "├── ∂z_vˢ: ", prettysummary(sd.∂z_vˢ, false), '\n')
    print(io, "├── ∂t_uˢ: ", prettysummary(sd.∂t_uˢ, false), '\n')
    print(io, "├── ∂t_vˢ: ", prettysummary(sd.∂t_vˢ, false), '\n')
    print(io, "└── ∂t_wˢ: ", prettysummary(sd.∂t_wˢ, false))
end

"""
    StokesDrift(; ∂z_uˢ=zerofunction, ∂y_uˢ=zerofunction, ∂t_uˢ=zerofunction, 
                  ∂z_vˢ=zerofunction, ∂x_vˢ=zerofunction, ∂t_vˢ=zerofunction, 
                  ∂x_wˢ=zerofunction, ∂y_wˢ=zerofunction, ∂t_wˢ=zerofunction, parameters=nothing)

Construct a set of functions of space and time for a Stokes drift velocity field
corresponding to a surface gravity wave field with an envelope that (potentially) varies
in the horizontal directions.

To resolve the evolution of the Lagrangian-mean momentum, we require all the components
of the "psuedovorticity",

```math
𝛁 × 𝐮ˢ = \\hat{\\boldsymbol{x}} (∂_y wˢ - ∂_z vˢ) + \\hat{\\boldsymbol{y}} (∂_z uˢ - ∂_x wˢ) + \\hat{\\boldsymbol{z}} (∂_x vˢ - ∂_y uˢ)
```

as well as the time-derivatives of ``uˢ``, ``vˢ``, and ``wˢ``.

Note that each function (e.g., `∂z_uˢ`) is generally a function of depth, horizontal coordinates,
and time.Thus, the correct function signature depends on the grid, since `Flat` horizontal directions
are omitted.

For example, on a grid with `topology = (Periodic, Flat, Bounded)` (and `parameters=nothing`),
then, e.g., `∂z_uˢ` is callable via `∂z_uˢ(x, z, t)`. When `!isnothing(parameters)`, then
`∂z_uˢ` is callable via `∂z_uˢ(x, z, t, parameters)`. Similarly, on a grid with
`topology = (Periodic, Periodic, Bounded)` and `parameters=nothing`, `∂z_uˢ` is called
via `∂z_uˢ(x, y, z, t)`.

Example
=======

A wavepacket moving with the group velocity. We write the Stokes drift as:

```math
uˢ(x, y, z, t) = A(x - cᵍ t, y) * ûˢ(z)
```

with ``A(ξ, η) = \\exp{-(ξ^2 + η^2) / 2δ^2}``. If ``uˢ`` represents the solenoidal component
of the Stokes drift, then ``∂_z wˢ = - ∂_x uˢ = - (∂_ξ A) ûˢ`` and therefore, under
the assumption that ``wˢ`` tends to zero at large depths, we get ``wˢ = - (∂_ξ A / 2k) ûˢ``.

```jldoctest
using Oceananigans
using Oceananigans.Units

g = 9.81 # gravitational acceleration

ϵ = 0.1
λ = 100meters  # horizontal wavelength
const k = 2π / λ  # horizontal wavenumber
c = sqrt(g / k)  # phase speed
const δ = 400kilometers  # wavepacket spread
const cᵍ = c / 2  # group speed
const Uˢ = ϵ^2 * c

@inline A(ξ, η) = exp(- (ξ^2 + η^2) / 2δ^2)

@inline ∂ξ_A(ξ, η) = - ξ / δ^2 * A(ξ, η)
@inline ∂η_A(ξ, η) = - η / δ^2 * A(ξ, η)
@inline ∂η_∂ξ_A(ξ, η) = η * ξ / δ^4 * A(ξ, η)
@inline ∂²ξ_A(ξ, η) = (ξ^2 / δ^2 - 1) * A(ξ, η) / δ^2

@inline ûˢ(z) = Uˢ * exp(2k * z)
@inline uˢ(x, y, z, t) = A(x - cᵍ * t, y) * ûˢ(z)

@inline ∂z_uˢ(x, y, z, t) = 2k * A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂y_uˢ(x, y, z, t) = ∂η_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂t_uˢ(x, y, z, t) = - cᵍ * ∂ξ_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂x_wˢ(x, y, z, t) = - 1 / 2k * ∂²ξ_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂y_wˢ(x, y, z, t) = - 1 / 2k * ∂η_∂ξ_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂t_wˢ(x, y, z, t) = + cᵍ / 2k * ∂²ξ_A(x - cᵍ * t, y) * ûˢ(z)

stokes_drift = StokesDrift(; ∂z_uˢ, ∂t_uˢ, ∂y_uˢ, ∂t_wˢ, ∂x_wˢ, ∂y_wˢ)

# output

StokesDrift{Nothing}:
├── ∂x_vˢ: zerofunction
├── ∂x_wˢ: ∂x_wˢ
├── ∂y_uˢ: ∂y_uˢ
├── ∂y_wˢ: ∂y_wˢ
├── ∂z_uˢ: ∂z_uˢ
├── ∂z_vˢ: zerofunction
├── ∂t_uˢ: ∂t_uˢ
├── ∂t_vˢ: zerofunction
└── ∂t_wˢ: ∂t_wˢ
```
"""
function StokesDrift(; ∂x_vˢ = zerofunction,
                       ∂x_wˢ = zerofunction,
                       ∂y_uˢ = zerofunction,
                       ∂y_wˢ = zerofunction,
                       ∂z_uˢ = zerofunction,
                       ∂z_vˢ = zerofunction,
                       ∂t_uˢ = zerofunction,
                       ∂t_vˢ = zerofunction,
                       ∂t_wˢ = zerofunction,
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
