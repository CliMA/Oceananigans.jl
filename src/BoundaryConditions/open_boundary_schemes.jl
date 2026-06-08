using Oceananigans: defaults
using Oceananigans.Grids: column_depthᶠᶜᵃ, column_depthᶜᶠᵃ, column_depthᶜᶜᵃ

"""
    Flather(; gravitational_acceleration = defaults.gravitational_acceleration)

Flather (1976) characteristic boundary condition for the shallow water equations.
Prescribes the incoming Riemann invariant while letting the outgoing one radiate freely:

    Uᵇ = Uᵉˣᵗ + √(g H) ⋅ (ηᵇ − ηᵉˣᵗ)

where `Uᵉˣᵗ` and `ηᵉˣᵗ` are external (prescribed) values, `ηᵇ` is the model free
surface at the boundary, and `H` is the water column depth.

`Flather` is used as the `scheme` of a [`NormalFlowBoundaryCondition`](@ref) for the
barotropic transport; see also the convenience constructor [`FlatherBoundaryCondition`](@ref).
The external values are provided as the boundary condition value and must be a 2-tuple
`(U, η)`. Each element can be a number, array, or function:

- Numbers and arrays are evaluated via `getbc` element-wise.
- Functions follow the standard boundary condition conventions (continuous or discrete form).

This condition is applied to barotropic velocity fields at every barotropic substep in
the split-explicit free surface solver. It requires `model_fields` to contain `η` (the
free surface displacement).

References
==========
* Flather, R. A. (1976). "A tidal model of the north-west European continental shelf."
  Memoires de la Societe Royale des Sciences de Liege, 6(10), 141-164.

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: Flather

flather = Flather()
typeof(flather)

# output
Flather{Float64}
```
"""
struct Flather{FT}
    gravitational_acceleration :: FT
end

function Flather(; gravitational_acceleration = defaults.gravitational_acceleration)
    return Flather(gravitational_acceleration)
end

Adapt.adapt_structure(to, f::Flather) =
    Flather(adapt(to, f.gravitational_acceleration))

const FNBCBC = BoundaryCondition{<:NormalFlow{<:Flather}}

"""
    Chapman(; gravitational_acceleration = defaults.gravitational_acceleration)

Chapman (1985) radiation condition for the free surface displacement at an open boundary,
the standard companion of [`Flather`](@ref): the boundary η radiates at the known
barotropic gravity-wave speed,

    ∂η/∂t ± √(g H) ∂η/∂n = 0

discretized implicitly (the form used by ROMS):

    ηᵇⁿ⁺¹ = (ηᵇⁿ + C η₁ⁿ⁺¹) / (1 + C),    C = √(g H) Δt / Δx

where `η₁` is the boundary-adjacent interior value. Letting the boundary η evolve frees
the surface pressure gradient at the boundary, which balanced flows require to cross it.

`Chapman` is used as the `scheme` of a [`ValueBoundaryCondition`](@ref) on the free
surface displacement `η`; see [`ChapmanBoundaryCondition`](@ref). It is applied at every
barotropic substep, like `Flather`.

References
==========
* Chapman, D. C. (1985). "Numerical treatment of cross-shelf open boundaries in a
  barotropic coastal ocean model." Journal of Physical Oceanography, 15(8), 1060-1075.
"""
struct Chapman{FT}
    gravitational_acceleration :: FT
end

function Chapman(; gravitational_acceleration = defaults.gravitational_acceleration)
    return Chapman(gravitational_acceleration)
end

Adapt.adapt_structure(to, c::Chapman) =
    Chapman(adapt(to, c.gravitational_acceleration))

const CHVBC = BoundaryCondition{<:Value{<:Chapman}}

"""
    Radiation(; inflow_timescale = 0, outflow_timescale = Inf)

Orlanski (1976) radiation condition with locally-diagnosed phase speed
and adaptive nudging (Marchesiello et al. 2001):

    ∂φ/∂t + cₙ ⋅ ∂φ/∂n = - (φ - φᵉˣᵗ) / τ

where `cₙ = −(∂φ/∂t) / (∂φ/∂n)` is diagnosed from interior values, clamped to `[0, Δx/Δt]`.
Inflow vs outflow is decided from the boundary-normal velocity (the boundary value itself
for `NormalFlow` fields, `model_fields` at the boundary face for `Value` fields):
`τ = τ_in` and `cₙ = 0` on inflow, `τ = τ_out` on outflow.

`Radiation` is used as the `scheme` of a [`ValueBoundaryCondition`](@ref) for `Center`-located
fields such as tracers — where it updates the halo cell adjacent to the boundary — or of a
[`NormalFlowBoundaryCondition`](@ref) for boundary-normal velocities — where it updates the
boundary face value. Interior cells are never written.

The previous-timestep interior value needed by the Orlanski phase-speed diagnosis is
stored in an array (`φ₁`) allocated automatically during boundary condition regularization.

References
==========
* Orlanski, I. (1976). "A simple boundary condition for unbounded hyperbolic flows."
  Journal of Computational Physics, 21(3), 251-269.
* Marchesiello, P., McWilliams, J. C., & Shchepetkin, A. (2001). "Open boundary conditions
  for long-term integration of regional oceanic models." Ocean Modelling, 3(1-2), 1-20.

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: Radiation

rad = Radiation(outflow_timescale = 360 * 86400, inflow_timescale = 86400)
rad.outflow_timescale

# output
3.1104e7
```
"""
struct Radiation{FT, S}
    outflow_timescale :: FT
    inflow_timescale  :: FT
    φᵇ  :: S  # anchor boundary value (2D array or nothing)
    φ₁  :: S  # anchor interior value (2D array or nothing)
    φ₁ˡ :: S  # latest interior value (2D array or nothing)
end

function Radiation(FT = defaults.FloatType;
                   inflow_timescale = 0,
                   outflow_timescale = Inf)

    outflow_timescale = convert(FT, outflow_timescale)
    inflow_timescale = convert(FT, inflow_timescale)
    return Radiation(outflow_timescale, inflow_timescale, nothing, nothing, nothing)
end

Adapt.adapt_structure(to, r::Radiation) =
    Radiation(adapt(to, r.outflow_timescale),
              adapt(to, r.inflow_timescale),
              adapt(to, r.φᵇ),
              adapt(to, r.φ₁),
              adapt(to, r.φ₁ˡ))

const RVBC  = BoundaryCondition{<:Value{<:Radiation}}
const RNFBC = BoundaryCondition{<:NormalFlow{<:Radiation}}
const RBC   = Union{RVBC, RNFBC}

#####
##### Radiation storage allocation during BC regularization
#####

# Allocate the 2D storage array holding the previous-timestep interior value φ₁ⁿ
# needed by the Orlanski phase-speed diagnosis.
function materialize_radiation_storage(radiation::Radiation, grid, loc, dim)
    FT = eltype(grid)
    Sx, Sy, Sz = size(grid, loc) # loc-aware: Face on Bounded gives N+1, matching the kernel range
    arch = architecture(grid)

    tangential_size = dim == 1 ? (Sy, Sz) :
                      dim == 2 ? (Sx, Sz) :
                                 (Sx, Sy)

    φᵇ  = on_architecture(arch, zeros(FT, tangential_size...))
    φ₁  = on_architecture(arch, zeros(FT, tangential_size...))
    φ₁ˡ = on_architecture(arch, zeros(FT, tangential_size...))

    return Radiation(radiation.outflow_timescale,
                     radiation.inflow_timescale,
                     φᵇ, φ₁, φ₁ˡ)
end

rebuild_classification(::Value, scheme) = Value(scheme)
rebuild_classification(::NormalFlow, scheme) = NormalFlow(scheme)

# Hook into the regularization pipeline to allocate Radiation storage
function regularize_boundary_condition(bc::RBC, grid, loc, dim, args...)
    regularized_condition = regularize_boundary_condition(bc.condition, grid, loc, dim, args...)
    radiation = bc.classification.scheme
    materialized_radiation = materialize_radiation_storage(radiation, grid, loc, dim)
    classification = rebuild_classification(bc.classification, materialized_radiation)
    return BoundaryCondition(classification, regularized_condition)
end

#####
##### Convenience constructors
#####

"""
    FlatherBoundaryCondition(val; gravitational_acceleration = defaults.gravitational_acceleration, kwargs...)

Construct a `NormalFlowBoundaryCondition` with the [`Flather`](@ref) scheme. `val` must be a
2-tuple `(U, η)` or a function returning a 2-tuple, where `U` is the external barotropic
transport and `η` is the external free surface displacement. Each element of the tuple can be
a number, array, or function (evaluated via `getbc`).

Example
=======

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: FlatherBoundaryCondition

bc = FlatherBoundaryCondition((0, 0))
bc isa Oceananigans.BoundaryConditions.BoundaryCondition

# output
true
```
"""
function FlatherBoundaryCondition(val; gravitational_acceleration = defaults.gravitational_acceleration, kwargs...)
    validate_flather_condition(val)
    scheme = Flather(; gravitational_acceleration)
    return NormalFlowBoundaryCondition(val; scheme, kwargs...)
end

FlatherBoundaryCondition(U, η; kwargs...) = FlatherBoundaryCondition((U, η); kwargs...)

"""
    ChapmanBoundaryCondition(; gravitational_acceleration = defaults.gravitational_acceleration)

Construct a `ValueBoundaryCondition` with the [`Chapman`](@ref) scheme for the free
surface displacement `η` at an open boundary. Pair with [`FlatherBoundaryCondition`](@ref)
on the barotropic transport.
"""
ChapmanBoundaryCondition(; gravitational_acceleration = defaults.gravitational_acceleration) =
    ValueBoundaryCondition(0; scheme = Chapman(; gravitational_acceleration))


function validate_flather_condition(val)
    if val isa Union{Tuple, NamedTuple}
        length(val) == 2 || throw(ArgumentError(
            "Flather boundary condition requires a 2-tuple (U, η) for " *
            "external transport and free surface, got a $(length(val))-tuple."))
    elseif !(val isa Function)
        throw(ArgumentError(
            "FlatherBoundaryCondition requires a 2-tuple (U, η) or a function " *
            "returning a 2-tuple, where U is the external barotropic transport " *
            "and η is the external free surface displacement. " *
            "Got an argument of type $(typeof(val))."))
    end
    return nothing
end

#####
##### Flather halo filling
#####

const FAA = Tuple{Face,   Any, Any}
const CAA = Tuple{Center, Any, Any}
const AFA = Tuple{Any, Face,   Any}
const ACA = Tuple{Any, Center, Any}
const AAF = Tuple{Any, Any, Face, }
const AAC = Tuple{Any, Any, Center}

# During initialization (no clock/model_fields available yet), fill halos with zero.
@inline   _fill_east_halo!(j, k, grid, c, bc::FNBCBC, loc) = @inbounds c[grid.Nx + 1, j, k] = zero(grid)
@inline   _fill_west_halo!(j, k, grid, c, bc::FNBCBC, loc) = @inbounds c[1, j, k]           = zero(grid)
@inline  _fill_north_halo!(i, k, grid, c, bc::FNBCBC, loc) = @inbounds c[i, grid.Ny + 1, k] = zero(grid)
@inline  _fill_south_halo!(i, k, grid, c, bc::FNBCBC, loc) = @inbounds c[i, 1, k]           = zero(grid)

# The Flather condition for normal barotropic transport at a boundary:
#
#   East/North (right boundary):  Uᵇ = Uᵉˣᵗ + √(g H) ⋅ (ηᵇ − ηᵉˣᵗ)
#   West/South (left  boundary):  Uᵇ = Uᵉˣᵗ − √(g H) ⋅ (ηᵇ − ηᵉˣᵗ)
#
# The sign convention follows from the characteristic decomposition of the
# shallow water equations: the incoming Riemann invariant is prescribed from
# external data while the outgoing one radiates freely.
#
# The boundary condition value (accessed via getbc) must return a 2-tuple (U, η)
# of external transport and free surface values.
#
# ηᵇ is the face average of the two adjacent cells (ROMS form): under η's default
# mirror fill this equals the interior sample, while a Chapman condition on the
# boundary row couples into the transport through the average.
#
# Requires `model_fields` to contain:
#   - η :: free surface displacement field

@inline function _fill_east_halo!(j, k, grid, c, bc::FNBCBC, ::FAA, clock, model_fields)
    i = grid.Nx + 1
    k_top = grid.Nz + 1
    flather = bc.classification.scheme

    g = flather.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶠᶜᵃ(i, j, k_top, grid, η)

    Uᵉˣᵗ, ηᵉˣᵗ = getbc(bc, j, k, grid, clock, model_fields)
    ηᵇ = ℑxᶠᵃᵃ(i, j, k_top, grid, η)

    @inbounds c[i, j, k] = Uᵉˣᵗ + sqrt(g * max(H, zero(H))) * (ηᵇ - ηᵉˣᵗ)

    return nothing
end

@inline function _fill_west_halo!(j, k, grid, c, bc::FNBCBC, ::FAA, clock, model_fields)
    k_top = grid.Nz + 1
    flather = bc.classification.scheme

    g = flather.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶠᶜᵃ(1, j, k_top, grid, η)

    Uᵉˣᵗ, ηᵉˣᵗ = getbc(bc, j, k, grid, clock, model_fields)
    ηᵇ = ℑxᶠᵃᵃ(1, j, k_top, grid, η)

    @inbounds c[1, j, k] = Uᵉˣᵗ - sqrt(g * max(H, zero(H))) * (ηᵇ - ηᵉˣᵗ)

    return nothing
end

@inline function _fill_north_halo!(i, k, grid, c, bc::FNBCBC, ::AFA, clock, model_fields)
    j = grid.Ny + 1
    k_top = grid.Nz + 1
    flather = bc.classification.scheme

    g = flather.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶜᶠᵃ(i, j, k_top, grid, η)

    Vᵉˣᵗ, ηᵉˣᵗ = getbc(bc, i, k, grid, clock, model_fields)
    ηᵇ = ℑyᵃᶠᵃ(i, j, k_top, grid, η)

    @inbounds c[i, j, k] = Vᵉˣᵗ + sqrt(g * max(H, zero(H))) * (ηᵇ - ηᵉˣᵗ)

    return nothing
end

@inline function _fill_south_halo!(i, k, grid, c, bc::FNBCBC, ::AFA, clock, model_fields)
    k_top = grid.Nz + 1
    flather = bc.classification.scheme

    g = flather.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶜᶠᵃ(i, 1, k_top, grid, η)

    Vᵉˣᵗ, ηᵉˣᵗ = getbc(bc, i, k, grid, clock, model_fields)
    ηᵇ = ℑyᵃᶠᵃ(i, 1, k_top, grid, η)

    @inbounds c[i, 1, k] = Vᵉˣᵗ - sqrt(g * max(H, zero(H))) * (ηᵇ - ηᵉˣᵗ)

    return nothing
end

# A fill without a clock (e.g. during initialization or state reconciliation) behaves
# as a first call: Δt = 0 and zero-gradient initialization of the boundary value.
@inline stage_Δt(clock) = clock.last_stage_Δt
@inline stage_Δt(::Nothing) = Inf

@inline anchored_fill(clock) = clock.stage ≤ 1
@inline anchored_fill(::Nothing) = true

#####
##### Chapman halo filling — implicit gravity-wave radiation of the free surface
#####

@inline function _fill_west_halo!(j, k, grid, c, bc::CHVBC, ::CAA, clock, model_fields)
    anchored_fill(clock) || return nothing
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    g = bc.classification.scheme.gravitational_acceleration
    k_top = grid.Nz + 1

    @inbounds begin
        η₁ = c[1, j, k]
        H  = column_depthᶜᶜᵃ(1, j, k_top, grid, c)
        C  = sqrt(g * H) * Δt / Δxᶠᶜᶜ(1, j, k, grid)
        ηᵇ = ifelse(first_call, η₁, c[0, j, k]) # zero-gradient initialization
        c[0, j, k] = (ηᵇ + C * η₁) / (1 + C)
    end

    return nothing
end

@inline function _fill_east_halo!(j, k, grid, c, bc::CHVBC, ::CAA, clock, model_fields)
    anchored_fill(clock) || return nothing
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    g = bc.classification.scheme.gravitational_acceleration
    i = grid.Nx + 1
    k_top = grid.Nz + 1

    @inbounds begin
        η₁ = c[i-1, j, k]
        H  = column_depthᶜᶜᵃ(i-1, j, k_top, grid, c)
        C  = sqrt(g * H) * Δt / Δxᶠᶜᶜ(i, j, k, grid)
        ηᵇ = ifelse(first_call, η₁, c[i, j, k]) # zero-gradient initialization
        c[i, j, k] = (ηᵇ + C * η₁) / (1 + C)
    end

    return nothing
end

@inline function _fill_south_halo!(i, k, grid, c, bc::CHVBC, ::ACA, clock, model_fields)
    anchored_fill(clock) || return nothing
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    g = bc.classification.scheme.gravitational_acceleration
    k_top = grid.Nz + 1

    @inbounds begin
        η₁ = c[i, 1, k]
        H  = column_depthᶜᶜᵃ(i, 1, k_top, grid, c)
        C  = sqrt(g * H) * Δt / Δyᶜᶠᶜ(i, 1, k, grid)
        ηᵇ = ifelse(first_call, η₁, c[i, 0, k]) # zero-gradient initialization
        c[i, 0, k] = (ηᵇ + C * η₁) / (1 + C)
    end

    return nothing
end

@inline function _fill_north_halo!(i, k, grid, c, bc::CHVBC, ::ACA, clock, model_fields)
    anchored_fill(clock) || return nothing
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    g = bc.classification.scheme.gravitational_acceleration
    j = grid.Ny + 1
    k_top = grid.Nz + 1

    @inbounds begin
        η₁ = c[i, j-1, k]
        H  = column_depthᶜᶜᵃ(i, j-1, k_top, grid, c)
        C  = sqrt(g * H) * Δt / Δyᶜᶠᶜ(i, j, k, grid)
        ηᵇ = ifelse(first_call, η₁, c[i, j, k]) # zero-gradient initialization
        c[i, j, k] = (ηᵇ + C * η₁) / (1 + C)
    end

    return nothing
end

#####
##### Radiation halo filling — Orlanski (1976) with Marchesiello et al. (2001) nudging
#####

# True Orlanski radiation condition with locally-diagnosed phase speed:
#
#   φᵇⁿ⁺¹ = (φᵇⁿ + Cₙ φ₁ⁿ⁺¹ + τ̃ φᵉˣᵗ) / (1 + Cₙ + τ̃)
#
# where Cₙ = cₙ Δt / Δx is the Courant number of the diagnosed phase speed,
# clamped to [0, 1]. The phase speed is diagnosed at the boundary-adjacent
# interior point from time and space derivatives:
#
#   Cₙ = -(φ₁ⁿ⁺¹ - φ₁ⁿ) / (φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹)
#
# where φ₁ is the boundary-adjacent interior value and φ₂ is one point
# deeper into the interior.
#
# The previous interior value φ₁ⁿ is stored in an array inside the Radiation struct;
# the previous boundary value is read from the field itself, so that increments applied
# between fills (e.g. the Flather-consistent barotropic correction at NormalFlow faces)
# are retained.
#
# Adaptive nudging (Marchesiello et al. 2001):
#   - Outflow: τ = outflow_timescale (typically weak or Inf)
#   - Inflow:  τ = inflow_timescale (typically strong)
#
# Inflow vs outflow is decided from the boundary-normal velocity, not from the sign of
# the diagnosed phase speed: the local gradient ∂φ∂ξ vanishes whenever an extremum exits
# through the boundary, where the phase-speed ratio blows up and flips sign, spuriously
# selecting the inflow branch mid-outflow (and slamming the boundary to φᵉˣᵗ when
# inflow_timescale is small). The velocity-based branch is the PerturbationAdvection
# convention; the diagnosed Cₙ, clamped to [0, 1], is kept as the radiation weight so
# that wave signals can still exit faster than the advecting flow.

@inline function orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
    # Diagnose phase speed Courant number (Orlanski 1976)
    ∂t_φ = φ₁ⁿ⁺¹ - φ₁ⁿ
    ∂ξ_φ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹

    # Cₙ = -(∂φ/∂t) / (∂φ/∂ξ) in the outward-normal direction
    # Guard against zero spatial gradient
    Cᶜ = ifelse(∂ξ_φ == 0, zero(∂t_φ), - ∂t_φ / ∂ξ_φ)

    # Radiation-plus-advection
    Cₙ = ifelse(outflow, max(zero(Cᶜ), min(one(Cᶜ), max(Cᶜ, Cᵃ))), zero(Cᶜ))

    τ = ifelse(outflow, radiation.outflow_timescale, radiation.inflow_timescale)
    τ̃ = Δt / τ

    # Implicit Orlanski radiation + nudging
    φᵇⁿ⁺¹ = (φᵇⁿ + Cₙ * φ₁ⁿ⁺¹ + τ̃ * φᵉˣᵗ) / (1 + Cₙ + τ̃)

    return ifelse(τ == 0, φᵉˣᵗ, φᵇⁿ⁺¹)
end

# The radiated point is the boundary face for NormalFlow (Face-located fields) and the first halo cell for Value 
# (Center-located fields): right boundaries coincide at N+1; left boundaries are 1 (Face) and 0 (Center).
# The update is a convex combination of previous boundary, interior, and exterior values, so it is bounded.

@inline function radiate_east_halo!(iᵇ, j, k, grid, c, bc, Uₙ, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, j, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[iᵇ-1, j, k]      # first interior (new time)
        φ₂ⁿ⁺¹ = c[iᵇ-2, j, k]      # second interior (new time)

        φᵇᵃ = ifelse(anchored, c[iᵇ, j, k], radiation.φᵇ[j, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[j, k], radiation.φ₁[j, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δxᶠᶜᶜ(iᵇ, j, k, grid)
        outflow = Uᵃ >= 0

        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)

        c[iᵇ, j, k]         = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[j, k]  = φᵇⁿ   # anchor for later stages
        radiation.φ₁[j, k]  = φ₁ⁿ   # anchor for later stages
        radiation.φ₁ˡ[j, k] = φ₁ⁿ⁺¹ # latest interior, promoted at the next anchored fill
    end

    return nothing
end

@inline function radiate_west_halo!(iᵇ, j, k, grid, c, bc, Uₙ, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, j, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[iᵇ+1, j, k]      # first interior (new time)
        φ₂ⁿ⁺¹ = c[iᵇ+2, j, k]      # second interior (new time)

        φᵇᵃ = ifelse(anchored, c[iᵇ, j, k], radiation.φᵇ[j, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[j, k], radiation.φ₁[j, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δxᶠᶜᶜ(iᵇ + 1, j, k, grid)
        outflow = Uᵃ <= 0

        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)

        c[iᵇ, j, k]         = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[j, k]  = φᵇⁿ   # anchor for later stages
        radiation.φ₁[j, k]  = φ₁ⁿ   # anchor for later stages
        radiation.φ₁ˡ[j, k] = φ₁ⁿ⁺¹ # latest interior, promoted at the next anchored fill
    end

    return nothing
end

@inline function radiate_north_halo!(jᵇ, i, k, grid, c, bc, Uₙ, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, jᵇ-1, k]      # first interior (new time)
        φ₂ⁿ⁺¹ = c[i, jᵇ-2, k]      # second interior (new time)

        φᵇᵃ = ifelse(anchored, c[i, jᵇ, k], radiation.φᵇ[i, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[i, k], radiation.φ₁[i, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δyᶜᶠᶜ(i, jᵇ, k, grid)
        outflow = Uᵃ >= 0

        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)

        c[i, jᵇ, k]         = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[i, k]  = φᵇⁿ   # anchor for later stages
        radiation.φ₁[i, k]  = φ₁ⁿ   # anchor for later stages
        radiation.φ₁ˡ[i, k] = φ₁ⁿ⁺¹ # latest interior, promoted at the next anchored fill
    end

    return nothing
end

@inline function radiate_south_halo!(jᵇ, i, k, grid, c, bc, Uₙ, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, jᵇ+1, k]      # first interior (new time)
        φ₂ⁿ⁺¹ = c[i, jᵇ+2, k]      # second interior (new time)

        φᵇᵃ = ifelse(anchored, c[i, jᵇ, k], radiation.φᵇ[i, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[i, k], radiation.φ₁[i, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δyᶜᶠᶜ(i, jᵇ + 1, k, grid)
        outflow = Uᵃ <= 0

        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)

        c[i, jᵇ, k]         = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[i, k]  = φᵇⁿ   # anchor for later stages
        radiation.φ₁[i, k]  = φ₁ⁿ   # anchor for later stages
        radiation.φ₁ˡ[i, k] = φ₁ⁿ⁺¹ # latest interior, promoted at the next anchored fill
    end

    return nothing
end

@inline function radiate_top_halo!(kᵇ, i, j, grid, c, bc, Uₙ, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, j, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, j, kᵇ-1]      # first interior (new time)
        φ₂ⁿ⁺¹ = c[i, j, kᵇ-2]      # second interior (new time)

        φᵇᵃ = ifelse(anchored, c[i, j, kᵇ], radiation.φᵇ[i, j])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[i, j], radiation.φ₁[i, j])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δzᶜᶜᶠ(i, j, kᵇ, grid)
        outflow = Uᵃ >= 0

        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)

        c[i, j, kᵇ]         = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[i, j]  = φᵇⁿ   # anchor for later stages
        radiation.φ₁[i, j]  = φ₁ⁿ   # anchor for later stages
        radiation.φ₁ˡ[i, j] = φ₁ⁿ⁺¹ # latest interior, promoted at the next anchored fill
    end

    return nothing
end

@inline function radiate_bottom_halo!(kᵇ, i, j, grid, c, bc, Uₙ, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, j, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, j, kᵇ+1]      # first interior (new time)
        φ₂ⁿ⁺¹ = c[i, j, kᵇ+2]      # second interior (new time)

        φᵇᵃ = ifelse(anchored, c[i, j, kᵇ], radiation.φᵇ[i, j])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[i, j], radiation.φ₁[i, j])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δzᶜᶜᶠ(i, j, kᵇ + 1, grid)
        outflow = Uᵃ <= 0

        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)

        c[i, j, kᵇ]         = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[i, j]  = φᵇⁿ   # anchor for later stages
        radiation.φ₁[i, j]  = φ₁ⁿ   # anchor for later stages
        radiation.φ₁ˡ[i, j] = φ₁ⁿ⁺¹ # latest interior, promoted at the next anchored fill
    end

    return nothing
end

# NormalFlow fields radiate with their own boundary value as advecting velocity (Uₙ = nothing);
# Value fields (tracers, tangential velocities) are advected by the boundary-normal velocity
@inline   _fill_east_halo!(j, k, grid, c, bc::RNFBC, ::FAA, clock, model_fields) =   radiate_east_halo!(grid.Nx+1, j, k, grid, c, bc, nothing, clock, model_fields)
@inline   _fill_west_halo!(j, k, grid, c, bc::RNFBC, ::FAA, clock, model_fields) =   radiate_west_halo!(1,         j, k, grid, c, bc, nothing, clock, model_fields)
@inline  _fill_north_halo!(i, k, grid, c, bc::RNFBC, ::AFA, clock, model_fields) =  radiate_north_halo!(grid.Ny+1, i, k, grid, c, bc, nothing, clock, model_fields)
@inline  _fill_south_halo!(i, k, grid, c, bc::RNFBC, ::AFA, clock, model_fields) =  radiate_south_halo!(1,         i, k, grid, c, bc, nothing, clock, model_fields)
@inline    _fill_top_halo!(i, j, grid, c, bc::RNFBC, ::AAF, clock, model_fields) =    radiate_top_halo!(grid.Nz+1, i, j, grid, c, bc, nothing, clock, model_fields)
@inline _fill_bottom_halo!(i, j, grid, c, bc::RNFBC, ::AAF, clock, model_fields) = radiate_bottom_halo!(1,         i, j, grid, c, bc, nothing, clock, model_fields)

@inline   _fill_east_halo!(j, k, grid, c, bc::RVBC,  ::CAA, clock, model_fields) =   radiate_east_halo!(grid.Nx+1, j, k, grid, c, bc, @inbounds(model_fields.u[grid.Nx, j, k]), clock, model_fields)
@inline   _fill_west_halo!(j, k, grid, c, bc::RVBC,  ::CAA, clock, model_fields) =   radiate_west_halo!(0,         j, k, grid, c, bc, @inbounds(model_fields.u[2, j, k]),       clock, model_fields)
@inline  _fill_north_halo!(i, k, grid, c, bc::RVBC,  ::ACA, clock, model_fields) =  radiate_north_halo!(grid.Ny+1, i, k, grid, c, bc, @inbounds(model_fields.v[i, grid.Ny, k]), clock, model_fields)
@inline  _fill_south_halo!(i, k, grid, c, bc::RVBC,  ::ACA, clock, model_fields) =  radiate_south_halo!(0,         i, k, grid, c, bc, @inbounds(model_fields.v[i, 2, k]),       clock, model_fields)
@inline    _fill_top_halo!(i, j, grid, c, bc::RVBC,  ::AAC, clock, model_fields) =    radiate_top_halo!(grid.Nz+1, i, j, grid, c, bc, @inbounds(model_fields.w[i, j, grid.Nz]), clock, model_fields)
@inline _fill_bottom_halo!(i, j, grid, c, bc::RVBC,  ::AAC, clock, model_fields) = radiate_bottom_halo!(0,         i, j, grid, c, bc, @inbounds(model_fields.w[i, j, 2]),       clock, model_fields)
