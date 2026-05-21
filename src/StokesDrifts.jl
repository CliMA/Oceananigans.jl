module StokesDrifts

export
    AbstractStokesDrift,
    UniformStokesDrift,
    StokesDrift,
    FieldStokesDrift,
    ∂t_uˢ,
    ∂t_vˢ,
    ∂t_wˢ,
    x_curl_Uˢ_cross_U,
    y_curl_Uˢ_cross_U,
    z_curl_Uˢ_cross_U,
    compute_stokes_drift!

using Adapt: adapt

using KernelAbstractions: @kernel, @index, synchronize

using Oceananigans.Fields
using Oceananigans.Fields: AbstractField
using Oceananigans.Operators
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, Az⁻¹ᶜᶜᶜ
using Oceananigans.Grids: AbstractGrid, node
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: prettysummary, launch!, KernelParameters

import Adapt: adapt_structure

#####
##### Abstract Stokes-drift type
#####

"""
    AbstractStokesDrift

Supertype of all Stokes-drift representations Oceananigans understands. Three
concrete subtypes ship with Oceananigans:

  - [`UniformStokesDrift`](@ref): analytic, horizontally-uniform Stokes drift
    given as functions of depth and time;
  - [`StokesDrift`](@ref): analytic Stokes drift with horizontal structure,
    given as functions of `(x, y, z, t)`;
  - [`FieldStokesDrift`](@ref): Stokes drift carried as Oceananigans `Field`s
    at the staggered velocity locations, suitable for couplings where
    `uˢ, vˢ` are supplied by an external wave model.

`compute_stokes_drift!(stokes_drift, grid)` is a no-op for the analytic
subtypes and refreshes the diagnostic Fields (`wˢ, ∂t_wˢ`) for
`FieldStokesDrift`.
"""
abstract type AbstractStokesDrift end

#####
##### Functions for "no surface waves"
#####

@inline ∂t_uˢ(i, j, k, grid, ::Nothing, time) = zero(grid)
@inline ∂t_vˢ(i, j, k, grid, ::Nothing, time) = zero(grid)
@inline ∂t_wˢ(i, j, k, grid, ::Nothing, time) = zero(grid)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, ::Nothing, U, time) = zero(grid)
@inline y_curl_Uˢ_cross_U(i, j, k, grid, ::Nothing, U, time) = zero(grid)
@inline z_curl_Uˢ_cross_U(i, j, k, grid, ::Nothing, U, time) = zero(grid)

compute_stokes_drift!(stokes_drift, grid) = nothing
compute_stokes_drift!(::Nothing, grid) = nothing

#####
##### Uniform surface waves
#####

struct UniformStokesDrift{P, UZ, VZ, UT, VT} <: AbstractStokesDrift
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

function UniformStokesDrift(grid::AbstractGrid;
                            ∂z_uˢ = Field{Nothing, Nothing, Face}(grid),
                            ∂z_vˢ = Field{Nothing, Nothing, Face}(grid),
                            ∂t_uˢ = Field{Nothing, Nothing, Center}(grid),
                            ∂t_vˢ = Field{Nothing, Nothing, Center}(grid),
                            parameters = nothing)

    return UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ, parameters)
end

const USD = UniformStokesDrift
const USDnoP = UniformStokesDrift{<:Nothing}
const c = Center()
const f = Face()

# Some helpers for three cases: Nothing, AbstractArray, or fallback (function)
@inline ∂z_Uᵃᵃᶜ(i, j, k, grid, sd::USDnoP, ∂z_Uˢ, time)                = ∂z_Uˢ(znode(k, grid, c), time)
@inline ∂z_Uᵃᵃᶜ(i, j, k, grid, sd::USD, ∂z_Uˢ, time)                   = ∂z_Uˢ(znode(k, grid, c), time, sd.parameters)
@inline ∂z_Uᵃᵃᶜ(i, j, k, grid, sd::USD, ∂z_Uˢ::AbstractArray, time)    = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_Uˢ)
@inline ∂z_Uᵃᵃᶜ(i, j, k, grid, sd::USDnoP, ∂z_Uˢ::AbstractArray, time) = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_Uˢ)
@inline ∂z_Uᵃᵃᶜ(i, j, k, grid, sd::USD, ::Nothing, time)               = zero(grid)
@inline ∂z_Uᵃᵃᶜ(i, j, k, grid, sd::USDnoP, ::Nothing, time)            = zero(grid)

@inline ∂z_Uᵃᵃᶠ(i, j, k, grid, sd::USDnoP, ∂z_Uˢ, time)                = ∂z_Uˢ(znode(k, grid, f), time)
@inline ∂z_Uᵃᵃᶠ(i, j, k, grid, sd::USD, ∂z_Uˢ, time)                   = ∂z_Uˢ(znode(k, grid, f), time, sd.parameters)
@inline ∂z_Uᵃᵃᶠ(i, j, k, grid, sd::USD, ∂z_Uˢ::AbstractArray, time)    = @inbounds ∂z_Uˢ[i, j, k]
@inline ∂z_Uᵃᵃᶠ(i, j, k, grid, sd::USDnoP, ∂z_Uˢ::AbstractArray, time) = @inbounds ∂z_Uˢ[i, j, k]
@inline ∂z_Uᵃᵃᶠ(i, j, k, grid, sd::USD, ::Nothing, time)               = zero(grid)
@inline ∂z_Uᵃᵃᶠ(i, j, k, grid, sd::USDnoP, ::Nothing, time)            = zero(grid)

@inline ∂t_U(i, j, k, grid, sd::USDnoP, ∂t_Uˢ, time)                = ∂t_Uˢ(znode(k, grid, c), time)
@inline ∂t_U(i, j, k, grid, sd::USD, ∂t_Uˢ, time)                   = ∂t_Uˢ(znode(k, grid, c), time, sd.parameters)
@inline ∂t_U(i, j, k, grid, sd::USD, ∂t_Uˢ::AbstractArray, time)    = @inbounds ∂t_Uˢ[i, j, k]
@inline ∂t_U(i, j, k, grid, sd::USDnoP, ∂t_Uˢ::AbstractArray, time) = @inbounds ∂t_Uˢ[i, j, k]
@inline ∂t_U(i, j, k, grid, sd::USD, ::Nothing, time)               = zero(grid)
@inline ∂t_U(i, j, k, grid, sd::USDnoP, ::Nothing, time)            = zero(grid)

# Kernel functions
@inline ∂t_uˢ(i, j, k, grid, sd::USD, time) = ∂t_U(i, j, k, grid, sd, sd.∂t_uˢ, time)
@inline ∂t_vˢ(i, j, k, grid, sd::USD, time) = ∂t_U(i, j, k, grid, sd, sd.∂t_vˢ, time)
@inline ∂t_wˢ(i, j, k, grid, sd::USD, time) = zero(grid)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sd::USD, U, time) =
    ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * ∂z_Uᵃᵃᶜ(i, j, k, grid, sd, sd.∂z_uˢ, time)

@inline y_curl_Uˢ_cross_U(i, j, k, grid, sd::USD, U, time) =
    ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * ∂z_Uᵃᵃᶜ(i, j, k, grid, sd, sd.∂z_vˢ, time)

@inline z_curl_Uˢ_cross_U(i, j, k, grid, sd::USD, U, time) = (
    - ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * ∂z_Uᵃᵃᶠ(i, j, k, grid, sd, sd.∂z_uˢ, time)
    - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * ∂z_Uᵃᵃᶠ(i, j, k, grid, sd, sd.∂z_vˢ, time))

struct StokesDrift{P, VX, WX, UY, WY, UZ, VZ, UT, VT, WT} <: AbstractStokesDrift
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
𝛁 × 𝐯ˢ = \\hat{\\boldsymbol{x}} (∂_y wˢ - ∂_z vˢ) + \\hat{\\boldsymbol{y}} (∂_z uˢ - ∂_x wˢ) + \\hat{\\boldsymbol{z}} (∂_x vˢ - ∂_y uˢ)
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

A wavepacket moving with the group velocity in the ``x``-direction.
We write the Stokes drift as:

```math
uˢ(x, y, z, t) = A(x - cᵍ \\, t, y) ûˢ(z)
```

with ``A(ξ, η) = \\exp{[-(ξ^2 + η^2) / 2δ^2]}``. We also assume ``vˢ = 0``.
If ``𝐯ˢ`` represents the solenoidal component of the Stokes drift, then
in this system from incompressibility requirement we have that
``∂_z wˢ = - ∂_x uˢ = - (∂_ξ A) ûˢ`` and therefore, under the assumption
that ``wˢ`` tends to zero at large depths, we get ``wˢ = - (∂_ξ A / 2k) ûˢ``.

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

For couplings where the Stokes drift comes from a wave-resolving model as
Oceananigans `Field`s rather than analytic functions, use
[`FieldStokesDrift`](@ref) instead.
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

    return StokesDrift(∂x_vˢ, ∂x_wˢ, ∂y_uˢ, ∂y_wˢ, ∂z_uˢ, ∂z_vˢ,
                       ∂t_uˢ, ∂t_vˢ, ∂t_wˢ, parameters)
end

const SD = StokesDrift
const SDnoP = StokesDrift{<:Nothing}

@inline parameters_tuple(sw::SDnoP) = tuple()
@inline parameters_tuple(sw::SD) = tuple(sw.parameters)

@inline ∂t_uˢ(i, j, k, grid, sw::SD, time) =
    sw.∂t_uˢ(node(i, j, k, grid, f, c, c)..., time, parameters_tuple(sw)...)
@inline ∂t_vˢ(i, j, k, grid, sw::SD, time) =
    sw.∂t_vˢ(node(i, j, k, grid, c, f, c)..., time, parameters_tuple(sw)...)
@inline ∂t_wˢ(i, j, k, grid, sw::SD, time) =
    sw.∂t_wˢ(node(i, j, k, grid, c, c, f)..., time, parameters_tuple(sw)...)

# Function-mode per-derivative helpers. Each `_∂{x,y,z}_{u,v,w}ˢ_<loc>`
# returns the appropriate component of ∇uˢ at node location <loc> by
# evaluating the user-supplied callable at the node.
@inline _∂z_uˢ_fcc(i, j, k, grid, sw::SD, time) =
    sw.∂z_uˢ(node(i, j, k, grid, f, c, c)..., time, parameters_tuple(sw)...)
@inline _∂z_uˢ_ccf(i, j, k, grid, sw::SD, time) =
    sw.∂z_uˢ(node(i, j, k, grid, c, c, f)..., time, parameters_tuple(sw)...)
@inline _∂z_vˢ_cfc(i, j, k, grid, sw::SD, time) =
    sw.∂z_vˢ(node(i, j, k, grid, c, f, c)..., time, parameters_tuple(sw)...)
@inline _∂z_vˢ_ccf(i, j, k, grid, sw::SD, time) =
    sw.∂z_vˢ(node(i, j, k, grid, c, c, f)..., time, parameters_tuple(sw)...)
@inline _∂y_uˢ_fcc(i, j, k, grid, sw::SD, time) =
    sw.∂y_uˢ(node(i, j, k, grid, f, c, c)..., time, parameters_tuple(sw)...)
@inline _∂y_uˢ_cfc(i, j, k, grid, sw::SD, time) =
    sw.∂y_uˢ(node(i, j, k, grid, c, f, c)..., time, parameters_tuple(sw)...)
@inline _∂x_vˢ_fcc(i, j, k, grid, sw::SD, time) =
    sw.∂x_vˢ(node(i, j, k, grid, f, c, c)..., time, parameters_tuple(sw)...)
@inline _∂x_vˢ_cfc(i, j, k, grid, sw::SD, time) =
    sw.∂x_vˢ(node(i, j, k, grid, c, f, c)..., time, parameters_tuple(sw)...)
@inline _∂x_wˢ(i, j, k, grid, sw::SD, X, time) =
    sw.∂x_wˢ(X..., time, parameters_tuple(sw)...)
@inline _∂y_wˢ(i, j, k, grid, sw::SD, X, time) =
    sw.∂y_wˢ(X..., time, parameters_tuple(sw)...)

#####
##### Field-mode Stokes drift
#####

"""
    FieldStokesDrift{P, US, VS, WS, UT, VT, WT} <: AbstractStokesDrift

Stokes-drift representation backed by Oceananigans `Field`s at the C-grid
velocity locations. Used for couplings where the Stokes drift state is
provided by an external wave model (e.g., a spectral or single-band
wave-action solver) rather than as a closed-form analytic expression.

The six prognostic Field slots are:

- `uˢ` at `(Face,   Center, Center)`,
- `vˢ` at `(Center, Face,   Center)`,
- `wˢ` at `(Center, Center, Face  )`,
- `∂t_uˢ` at `(Face,   Center, Center)`,
- `∂t_vˢ` at `(Center, Face,   Center)`,
- `∂t_wˢ` at `(Center, Center, Face  )`.

Spatial derivatives in the vortex-force curl
(`x_curl_Uˢ_cross_U, y_curl_Uˢ_cross_U, z_curl_Uˢ_cross_U`) are computed
inline from `uˢ, vˢ, wˢ` via the staggered finite-difference operators.
Time derivatives are read directly via `getindex` — the user is expected
to refresh `uˢ, vˢ, ∂t_uˢ, ∂t_vˢ` from the wave model before each ocean
step. `wˢ` and `∂t_wˢ` are computed automatically by
[`compute_stokes_drift!`](@ref) at the start of `update_state!` by
vertical integration of incompressibility.
"""
struct FieldStokesDrift{P, US, VS, WS, UT, VT, WT} <: AbstractStokesDrift
    parameters :: P
    uˢ    :: US
    vˢ    :: VS
    wˢ    :: WS
    ∂t_uˢ :: UT
    ∂t_vˢ :: VT
    ∂t_wˢ :: WT
end

adapt_structure(to, sd::FieldStokesDrift) =
    FieldStokesDrift(adapt(to, sd.parameters),
                     adapt(to, sd.uˢ),
                     adapt(to, sd.vˢ),
                     adapt(to, sd.wˢ),
                     adapt(to, sd.∂t_uˢ),
                     adapt(to, sd.∂t_vˢ),
                     adapt(to, sd.∂t_wˢ))

Base.summary(::FieldStokesDrift{Nothing}) = "FieldStokesDrift{Nothing}"

function Base.summary(sd::FieldStokesDrift)
    p_str = prettysummary(sd.parameters)
    return "FieldStokesDrift with parameters $p_str"
end

function Base.show(io::IO, sd::FieldStokesDrift)
    print(io, summary(sd), ':', '\n')
    print(io, "├── uˢ:    ", prettysummary(sd.uˢ,    false), '\n')
    print(io, "├── vˢ:    ", prettysummary(sd.vˢ,    false), '\n')
    print(io, "├── wˢ:    ", prettysummary(sd.wˢ,    false), '\n')
    print(io, "├── ∂t_uˢ: ", prettysummary(sd.∂t_uˢ, false), '\n')
    print(io, "├── ∂t_vˢ: ", prettysummary(sd.∂t_vˢ, false), '\n')
    print(io, "└── ∂t_wˢ: ", prettysummary(sd.∂t_wˢ, false))
end

"""
    FieldStokesDrift(grid; uˢ=…, vˢ=…, wˢ=…, ∂t_uˢ=…, ∂t_vˢ=…, ∂t_wˢ=…, parameters=nothing)

Allocate a `FieldStokesDrift` with C-grid-located `Field`s for each
prognostic slot. Each slot defaults to a freshly-allocated `Field` at the
matching staggered location and can be overridden with a user-supplied
`Field` via the corresponding kwarg (useful when the caller wants to
share a Field with other code).

```julia
sd = FieldStokesDrift(grid)                          # defaults
sd = FieldStokesDrift(grid; uˢ=my_uˢ_field)          # share uˢ, default rest
```
"""
function FieldStokesDrift(grid;
                          uˢ    = Field{Face,   Center, Center}(grid),
                          vˢ    = Field{Center, Face,   Center}(grid),
                          wˢ    = Field{Center, Center, Face  }(grid),
                          ∂t_uˢ = Field{Face,   Center, Center}(grid),
                          ∂t_vˢ = Field{Center, Face,   Center}(grid),
                          ∂t_wˢ = Field{Center, Center, Face  }(grid),
                          parameters = nothing)
    return FieldStokesDrift(parameters, uˢ, vˢ, wˢ, ∂t_uˢ, ∂t_vˢ, ∂t_wˢ)
end

const FSD = FieldStokesDrift

@inline ∂t_uˢ(i, j, k, grid, sw::FSD, time) = @inbounds sw.∂t_uˢ[i, j, k]
@inline ∂t_vˢ(i, j, k, grid, sw::FSD, time) = @inbounds sw.∂t_vˢ[i, j, k]
@inline ∂t_wˢ(i, j, k, grid, sw::FSD, time) = @inbounds sw.∂t_wˢ[i, j, k]

# Field-mode per-derivative helpers. Compute the relevant component of ∇uˢ
# inline from the stored Fields via the staggered FD operators.
@inline _∂z_uˢ_fcc(i, j, k, grid, sw::FSD, time) =
    ℑzᵃᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, sw.uˢ)
@inline _∂z_uˢ_ccf(i, j, k, grid, sw::FSD, time) =
    ℑxᶜᵃᵃ(i, j, k, grid, ∂zᶠᶜᶠ, sw.uˢ)
@inline _∂z_vˢ_cfc(i, j, k, grid, sw::FSD, time) =
    ℑzᵃᵃᶜ(i, j, k, grid, ∂zᶜᶠᶠ, sw.vˢ)
@inline _∂z_vˢ_ccf(i, j, k, grid, sw::FSD, time) =
    ℑyᵃᶜᵃ(i, j, k, grid, ∂zᶜᶠᶠ, sw.vˢ)
@inline _∂y_uˢ_fcc(i, j, k, grid, sw::FSD, time) =
    ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶠᶠᶜ, sw.uˢ)
@inline _∂y_uˢ_cfc(i, j, k, grid, sw::FSD, time) =
    ℑxᶜᵃᵃ(i, j, k, grid, ∂yᶠᶠᶜ, sw.uˢ)
@inline _∂x_vˢ_fcc(i, j, k, grid, sw::FSD, time) =
    ℑyᵃᶜᵃ(i, j, k, grid, ∂xᶠᶠᶜ, sw.vˢ)
@inline _∂x_vˢ_cfc(i, j, k, grid, sw::FSD, time) =
    ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶠᶜ, sw.vˢ)
@inline _∂x_wˢ(i, j, k, grid, sw::FSD, X, time) =
    ℑzᵃᵃᶜ(i, j, k, grid, ∂xᶠᶜᶠ, sw.wˢ)
@inline _∂y_wˢ(i, j, k, grid, sw::FSD, X, time) =
    ℑzᵃᵃᶜ(i, j, k, grid, ∂yᶜᶠᶠ, sw.wˢ)

#####
##### Shared vortex-force evaluation
#####
##### Both `StokesDrift` (function mode) and `FieldStokesDrift` (Field mode)
##### evaluate the vortex-force `(∇×uˢ) × uᴱ` via the same per-derivative
##### helpers; only the helper dispatch differs. The three `*_curl_Uˢ_cross_U`
##### methods live on the union of the two types so the body is shared.
#####

const NonUniformSD = Union{StokesDrift, FieldStokesDrift}

@inline function x_curl_Uˢ_cross_U(i, j, k, grid, sw::NonUniformSD, U, time)
    wᶠᶜᶜ = ℑxzᶠᵃᶜ(i, j, k, grid, U.w)
    vᶠᶜᶜ = ℑxyᶠᶜᵃ(i, j, k, grid, U.v)

    X = node(i, j, k, grid, f, c, c)
    ∂z_uˢ = _∂z_uˢ_fcc(i, j, k, grid, sw, time)
    ∂x_wˢ = _∂x_wˢ(i, j, k, grid, sw, X, time)
    ∂y_uˢ = _∂y_uˢ_fcc(i, j, k, grid, sw, time)
    ∂x_vˢ = _∂x_vˢ_fcc(i, j, k, grid, sw, time)

    return wᶠᶜᶜ * (∂z_uˢ - ∂x_wˢ) - vᶠᶜᶜ * (∂x_vˢ - ∂y_uˢ)
end

@inline function y_curl_Uˢ_cross_U(i, j, k, grid, sw::NonUniformSD, U, time)
    wᶜᶠᶜ = ℑyzᵃᶠᶜ(i, j, k, grid, U.w)
    uᶜᶠᶜ = ℑxyᶜᶠᵃ(i, j, k, grid, U.u)

    X = node(i, j, k, grid, c, f, c)
    ∂z_vˢ = _∂z_vˢ_cfc(i, j, k, grid, sw, time)
    ∂y_wˢ = _∂y_wˢ(i, j, k, grid, sw, X, time)
    ∂x_vˢ = _∂x_vˢ_cfc(i, j, k, grid, sw, time)
    ∂y_uˢ = _∂y_uˢ_cfc(i, j, k, grid, sw, time)

    return uᶜᶠᶜ * (∂x_vˢ - ∂y_uˢ) - wᶜᶠᶜ * (∂y_wˢ - ∂z_vˢ)
end

@inline function z_curl_Uˢ_cross_U(i, j, k, grid, sw::NonUniformSD, U, time)
    uᶜᶜᶠ = ℑxzᶜᵃᶠ(i, j, k, grid, U.u)
    vᶜᶜᶠ = ℑyzᵃᶜᶠ(i, j, k, grid, U.v)

    X = node(i, j, k, grid, c, c, f)
    ∂x_wˢ = _∂x_wˢ(i, j, k, grid, sw, X, time)
    ∂z_uˢ = _∂z_uˢ_ccf(i, j, k, grid, sw, time)
    ∂y_wˢ = _∂y_wˢ(i, j, k, grid, sw, X, time)
    ∂z_vˢ = _∂z_vˢ_ccf(i, j, k, grid, sw, time)

    return vᶜᶜᶠ * (∂y_wˢ - ∂z_vˢ) - uᶜᶜᶠ * (∂z_uˢ - ∂x_wˢ)
end

#####
##### compute_stokes_drift! — fills wˢ and ∂t_wˢ on a FieldStokesDrift by
##### vertical integration of incompressibility from the bottom upward.
##### No-op for the analytic StokesDrift / UniformStokesDrift.
#####

@kernel function _compute_wˢ_from_continuity!(wˢ, uˢ, vˢ, grid)
    i, j = @index(Global, NTuple)
    wᵏ = zero(eltype(wˢ))
    @inbounds wˢ[i, j, 1] = wᵏ
    Nz = size(grid, 3)
    for k in 2:Nz+1
        δ = flux_div_xyᶜᶜᶜ(i, j, k - 1, grid, uˢ, vˢ) *
            Az⁻¹ᶜᶜᶜ(i, j, k - 1, grid)
        wᵏ -= δ
        @inbounds wˢ[i, j, k] = wᵏ
    end
end

function compute_stokes_drift!(sw::FieldStokesDrift, grid)
    arch = architecture(grid)
    Nx, Ny, _ = size(grid)
    params = KernelParameters(1:Nx, 1:Ny)
    launch!(arch, grid, params, _compute_wˢ_from_continuity!,
            sw.wˢ, sw.uˢ, sw.vˢ, grid)
    launch!(arch, grid, params, _compute_wˢ_from_continuity!,
            sw.∂t_wˢ, sw.∂t_uˢ, sw.∂t_vˢ, grid)
    return nothing
end

end # module
