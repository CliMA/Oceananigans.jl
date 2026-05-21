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
using Oceananigans.Fields: AbstractField
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

struct StokesDrift{P, VX, WX, UY, WY, UZ, VZ, UT, VT, WT, US, VS}
    ∂x_vˢ :: VX
    ∂x_wˢ :: WX
    ∂y_uˢ :: UY
    ∂y_wˢ :: WY
    ∂z_uˢ :: UZ
    ∂z_vˢ :: VZ
    ∂t_uˢ :: UT
    ∂t_vˢ :: VT
    ∂t_wˢ :: WT
    uˢ    :: US
    vˢ    :: VS
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
                                                   adapt(to, sd.uˢ),
                                                   adapt(to, sd.vˢ),
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
    print(io, "├── ∂t_wˢ: ", prettysummary(sd.∂t_wˢ, false), '\n')
    print(io, "├── uˢ:    ", prettysummary(sd.uˢ,    false), '\n')
    print(io, "└── vˢ:    ", prettysummary(sd.vˢ,    false))
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
                       uˢ = nothing,
                       vˢ = nothing,
                       parameters = nothing)

    return StokesDrift(∂x_vˢ, ∂x_wˢ, ∂y_uˢ, ∂y_wˢ, ∂z_uˢ, ∂z_vˢ,
                       ∂t_uˢ, ∂t_vˢ, ∂t_wˢ, uˢ, vˢ, parameters)
end

const SD = StokesDrift
const SDnoP = StokesDrift{<:Nothing}

@inline parameters_tuple(sw::SDnoP) = tuple()
@inline parameters_tuple(sw::SD) = tuple(sw.parameters)

# Function-only path: evaluate the callable `sw.∂t_..ˢ` at the node. This
# single method covers both the parametric (`SD`) and non-parametric (`SDnoP`)
# function paths via `parameters_tuple`.
@inline ∂t_uˢ(i, j, k, grid, sw::SD, time) = sw.∂t_uˢ(node(i, j, k, grid, f, c, c)..., time, parameters_tuple(sw)...)
@inline ∂t_vˢ(i, j, k, grid, sw::SD, time) = sw.∂t_vˢ(node(i, j, k, grid, c, f, c)..., time, parameters_tuple(sw)...)
@inline ∂t_wˢ(i, j, k, grid, sw::SD, time) = sw.∂t_wˢ(node(i, j, k, grid, c, c, f)..., time, parameters_tuple(sw)...)

#####
##### Per-derivative dispatch.
#####
##### Each `_∂{x,y,z}_{u,v,w}ˢ_<loc>(i, j, k, grid, sw, time)` returns the
##### appropriate component of ∇·uˢ at node location <loc>. When
##### `sw.uˢ === nothing` and `sw.vˢ === nothing` (the function-only path),
##### the corresponding `sw.∂..._uˢ`/`sw.∂..._vˢ` callable is evaluated at
##### the node. When `sw.uˢ` or `sw.vˢ` is a `Field` (or any object that
##### satisfies the `getindex(uˢ, i, j, k)` contract), the spatial
##### derivative is computed inline via the staggered finite-difference
##### operators. `wˢ` is taken to be zero in the Field path.
#####

const SDFieldUˢ = StokesDrift{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                              <:Any, <:Any, <:Any, <:AbstractField}
const SDFieldVˢ = StokesDrift{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                              <:Any, <:Any, <:Any, <:Any, <:AbstractField}
const SDFieldUVˢ = StokesDrift{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                               <:Any, <:Any, <:Any, <:AbstractField, <:AbstractField}

# `∂t_uˢ`, `∂t_vˢ` may also be supplied as `AbstractField`s (at the
# corresponding velocity location). When a Field is supplied, the value is
# read directly via `getindex` at the node — no time-derivative is taken of
# the Field; the user is expected to refresh the Field with the desired
# time-derivative (e.g., from a wave model's analytic action tendency)
# before each ocean step.
const SDFieldDtUˢ = StokesDrift{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                <:AbstractField}
const SDFieldDtVˢ = StokesDrift{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                <:Any, <:AbstractField}

@inline ∂t_uˢ(i, j, k, grid, sw::SDFieldDtUˢ, time) = @inbounds sw.∂t_uˢ[i, j, k]
@inline ∂t_vˢ(i, j, k, grid, sw::SDFieldDtVˢ, time) = @inbounds sw.∂t_vˢ[i, j, k]

# ∂z uˢ
@inline _∂z_uˢ_fcc(i, j, k, grid, sw, time) =
    sw.∂z_uˢ(node(i, j, k, grid, f, c, c)..., time, parameters_tuple(sw)...)
@inline _∂z_uˢ_fcc(i, j, k, grid, sw::SDFieldUˢ, time) =
    ℑzᵃᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, sw.uˢ)
@inline _∂z_uˢ_ccf(i, j, k, grid, sw, time) =
    sw.∂z_uˢ(node(i, j, k, grid, c, c, f)..., time, parameters_tuple(sw)...)
@inline _∂z_uˢ_ccf(i, j, k, grid, sw::SDFieldUˢ, time) =
    ℑxᶜᵃᵃ(i, j, k, grid, ∂zᶠᶜᶠ, sw.uˢ)

# ∂z vˢ
@inline _∂z_vˢ_cfc(i, j, k, grid, sw, time) =
    sw.∂z_vˢ(node(i, j, k, grid, c, f, c)..., time, parameters_tuple(sw)...)
@inline _∂z_vˢ_cfc(i, j, k, grid, sw::SDFieldVˢ, time) =
    ℑzᵃᵃᶜ(i, j, k, grid, ∂zᶜᶠᶠ, sw.vˢ)
@inline _∂z_vˢ_ccf(i, j, k, grid, sw, time) =
    sw.∂z_vˢ(node(i, j, k, grid, c, c, f)..., time, parameters_tuple(sw)...)
@inline _∂z_vˢ_ccf(i, j, k, grid, sw::SDFieldVˢ, time) =
    ℑyᵃᶜᵃ(i, j, k, grid, ∂zᶜᶠᶠ, sw.vˢ)

# ∂y uˢ
@inline _∂y_uˢ_fcc(i, j, k, grid, sw, time) =
    sw.∂y_uˢ(node(i, j, k, grid, f, c, c)..., time, parameters_tuple(sw)...)
@inline _∂y_uˢ_fcc(i, j, k, grid, sw::SDFieldUˢ, time) =
    ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶠᶠᶜ, sw.uˢ)
@inline _∂y_uˢ_cfc(i, j, k, grid, sw, time) =
    sw.∂y_uˢ(node(i, j, k, grid, c, f, c)..., time, parameters_tuple(sw)...)
@inline _∂y_uˢ_cfc(i, j, k, grid, sw::SDFieldUˢ, time) =
    ℑxᶜᵃᵃ(i, j, k, grid, ∂yᶠᶠᶜ, sw.uˢ)

# ∂x vˢ
@inline _∂x_vˢ_fcc(i, j, k, grid, sw, time) =
    sw.∂x_vˢ(node(i, j, k, grid, f, c, c)..., time, parameters_tuple(sw)...)
@inline _∂x_vˢ_fcc(i, j, k, grid, sw::SDFieldVˢ, time) =
    ℑyᵃᶜᵃ(i, j, k, grid, ∂xᶠᶠᶜ, sw.vˢ)
@inline _∂x_vˢ_cfc(i, j, k, grid, sw, time) =
    sw.∂x_vˢ(node(i, j, k, grid, c, f, c)..., time, parameters_tuple(sw)...)
@inline _∂x_vˢ_cfc(i, j, k, grid, sw::SDFieldVˢ, time) =
    ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶠᶜ, sw.vˢ)

# ∂x wˢ and ∂y wˢ — wˢ has no Field counterpart on `StokesDrift`; the
# function-only path stays callable, the Field path returns zero.
@inline _∂x_wˢ(i, j, k, grid, sw, X, time) =
    sw.∂x_wˢ(X..., time, parameters_tuple(sw)...)
@inline _∂x_wˢ(i, j, k, grid, sw::SDFieldUˢ, X, time) = zero(grid)
@inline _∂x_wˢ(i, j, k, grid, sw::SDFieldVˢ, X, time) = zero(grid)
@inline _∂x_wˢ(i, j, k, grid, sw::SDFieldUVˢ, X, time) = zero(grid)
@inline _∂y_wˢ(i, j, k, grid, sw, X, time) =
    sw.∂y_wˢ(X..., time, parameters_tuple(sw)...)
@inline _∂y_wˢ(i, j, k, grid, sw::SDFieldUˢ, X, time) = zero(grid)
@inline _∂y_wˢ(i, j, k, grid, sw::SDFieldVˢ, X, time) = zero(grid)
@inline _∂y_wˢ(i, j, k, grid, sw::SDFieldUVˢ, X, time) = zero(grid)

@inline function x_curl_Uˢ_cross_U(i, j, k, grid, sw::SD, U, time)
    wᶠᶜᶜ = ℑxzᶠᵃᶜ(i, j, k, grid, U.w)
    vᶠᶜᶜ = ℑxyᶠᶜᵃ(i, j, k, grid, U.v)

    X = node(i, j, k, grid, f, c, c)
    ∂z_uˢ = _∂z_uˢ_fcc(i, j, k, grid, sw, time)
    ∂x_wˢ = _∂x_wˢ(i, j, k, grid, sw, X, time)
    ∂y_uˢ = _∂y_uˢ_fcc(i, j, k, grid, sw, time)
    ∂x_vˢ = _∂x_vˢ_fcc(i, j, k, grid, sw, time)

    return wᶠᶜᶜ * (∂z_uˢ - ∂x_wˢ) - vᶠᶜᶜ * (∂x_vˢ - ∂y_uˢ)
end

@inline function y_curl_Uˢ_cross_U(i, j, k, grid, sw::SD, U, time)
    wᶜᶠᶜ = ℑyzᵃᶠᶜ(i, j, k, grid, U.w)
    uᶜᶠᶜ = ℑxyᶜᶠᵃ(i, j, k, grid, U.u)

    X = node(i, j, k, grid, c, f, c)
    ∂z_vˢ = _∂z_vˢ_cfc(i, j, k, grid, sw, time)
    ∂y_wˢ = _∂y_wˢ(i, j, k, grid, sw, X, time)
    ∂x_vˢ = _∂x_vˢ_cfc(i, j, k, grid, sw, time)
    ∂y_uˢ = _∂y_uˢ_cfc(i, j, k, grid, sw, time)

    return uᶜᶠᶜ * (∂x_vˢ - ∂y_uˢ) - wᶜᶠᶜ * (∂y_wˢ - ∂z_vˢ)
end

@inline function z_curl_Uˢ_cross_U(i, j, k, grid, sw::SD, U, time)
    uᶜᶜᶠ = ℑxzᶜᵃᶠ(i, j, k, grid, U.u)
    vᶜᶜᶠ = ℑyzᵃᶜᶠ(i, j, k, grid, U.v)

    X = node(i, j, k, grid, c, c, f)
    ∂x_wˢ = _∂x_wˢ(i, j, k, grid, sw, X, time)
    ∂z_uˢ = _∂z_uˢ_ccf(i, j, k, grid, sw, time)
    ∂y_wˢ = _∂y_wˢ(i, j, k, grid, sw, X, time)
    ∂z_vˢ = _∂z_vˢ_ccf(i, j, k, grid, sw, time)

    return vᶜᶜᶠ * (∂y_wˢ - ∂z_vˢ) - uᶜᶜᶠ * (∂z_uˢ - ∂x_wˢ)
end


end # module
