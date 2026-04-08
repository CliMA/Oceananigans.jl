using Oceananigans: defaults
using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ
using Oceananigans.Grids: column_depthᶠᶜᵃ, column_depthᶜᶠᵃ

"""
    Flather(; external_values, gravitational_acceleration = defaults.gravitational_acceleration)

Flather (1976) characteristic boundary condition for the shallow water equations.
Prescribes the incoming Riemann invariant while letting the outgoing one radiate freely:

    Uᵇ = Uᵉˣᵗ + √(g H) ⋅ (ηᵇ − ηᵉˣᵗ)

where `Uᵉˣᵗ` and `ηᵉˣᵗ` are external (prescribed) values, `ηᵇ` is the model free
surface at the boundary, and `H` is the water column depth.

The `external_values` should be a `NamedTuple` with fields `η` and `U`, each of which
can be a number, array, or function of `(ξ, η, t)`.

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

flather = Flather(external_values = (η = 0, U = 0))
typeof(flather)

# output
Flather{@NamedTuple{η::Int64, U::Int64}, Float64}
```
"""
struct Flather{E, FT}
    external_values :: E
    gravitational_acceleration :: FT
end

function Flather(; external_values,
                   gravitational_acceleration = defaults.gravitational_acceleration)
    return Flather(external_values, gravitational_acceleration)
end

Adapt.adapt_structure(to, f::Flather) =
    Flather(adapt(to, f.external_values), adapt(to, f.gravitational_acceleration))

const FlatherOBC = BoundaryCondition{<:Open{<:Flather}}

"""
    Radiation(; outflow_relaxation_timescale = Inf, inflow_relaxation_timescale = 300)

Orlanski (1976) radiation condition with locally-diagnosed phase speed
and adaptive nudging (Marchesiello et al. 2001):

    ∂φ/∂t + cₙ ⋅ ∂φ/∂n = -(φ - φᵉˣᵗ) / τ

where `cₙ = −(∂φ/∂t) / (∂φ/∂n)` is diagnosed from interior values, clamped to `[0, Δx/Δt]`.
`τ = τ_in` on inflow (`cₙ < 0` pointing inward), `τ = τ_out` on outflow.

The previous-timestep boundary and interior values needed by the Orlanski formula are
stored in separate arrays (`φᵇ` and `φ₁`) rather than in the field's halo region.
These arrays are allocated automatically during boundary condition regularization.

References
==========
* Orlanski, I. (1976). "A simple boundary condition for unbounded hyperbolic flows."
  Journal of Computational Physics, 21(3), 251-269.
* Marchesiello, P., McWilliams, J. C., & Shchepetkin, A. (2001). "Open boundary conditions
  for long-term integration of regional oceanic models." Ocean Modelling, 3(1-2), 1-20.

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: Radiation

rad = Radiation(outflow_relaxation_timescale = 360 * 86400, inflow_relaxation_timescale = 86400)
rad.outflow_relaxation_timescale

# output
3.1104e7
```
"""
struct Radiation{FT, S}
    outflow_relaxation_timescale :: FT
    inflow_relaxation_timescale  :: FT
    φᵇ :: S  # previous boundary value storage (2D array or nothing)
    φ₁ :: S  # previous interior value storage (2D array or nothing)
end

function Radiation(FT = defaults.FloatType;
                   outflow_relaxation_timescale = Inf,
                   inflow_relaxation_timescale = 300)

    outflow_relaxation_timescale = convert(FT, outflow_relaxation_timescale)
    inflow_relaxation_timescale = convert(FT, inflow_relaxation_timescale)
    return Radiation(outflow_relaxation_timescale, inflow_relaxation_timescale, nothing, nothing)
end

Adapt.adapt_structure(to, r::Radiation) =
    Radiation(adapt(to, r.outflow_relaxation_timescale),
              adapt(to, r.inflow_relaxation_timescale),
              adapt(to, r.φᵇ),
              adapt(to, r.φ₁))

const RadiationOBC = BoundaryCondition{<:Open{<:Radiation}}

#####
##### Radiation storage allocation during BC regularization
#####

# Allocate 2D storage arrays for the Orlanski radiation scheme.
# The arrays hold previous-timestep values (φᵇⁿ and φ₁ⁿ) that must
# not be stored in the field's halo (which gets modified by other kernels).

function materialize_radiation_storage(radiation::Radiation, grid, dim)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    arch = architecture(grid)

    if dim == 1      # x-boundary (east/west): indexed by (j, k)
        φᵇ = on_architecture(arch, zeros(FT, Ny, Nz))
        φ₁ = on_architecture(arch, zeros(FT, Ny, Nz))
    elseif dim == 2  # y-boundary (north/south): indexed by (i, k)
        φᵇ = on_architecture(arch, zeros(FT, Nx, Nz))
        φ₁ = on_architecture(arch, zeros(FT, Nx, Nz))
    else             # z-boundary (top/bottom): indexed by (i, j)
        φᵇ = on_architecture(arch, zeros(FT, Nx, Ny))
        φ₁ = on_architecture(arch, zeros(FT, Nx, Ny))
    end

    return Radiation(radiation.outflow_relaxation_timescale,
                     radiation.inflow_relaxation_timescale,
                     φᵇ, φ₁)
end

# Hook into the regularization pipeline to allocate Radiation storage
function regularize_boundary_condition(bc::RadiationOBC, grid, loc, dim, args...)
    regularized_condition = regularize_boundary_condition(bc.condition, grid, loc, dim, args...)
    radiation = bc.classification.scheme
    materialized_radiation = materialize_radiation_storage(radiation, grid, dim)
    classification = Open(materialized_radiation)
    return BoundaryCondition(classification, regularized_condition)
end

#####
##### Helper to evaluate external values (numbers, arrays, or functions)
#####

@inline get_external_value(val::Number, args...) = val
@inline get_external_value(val::AbstractArray, i, grid, clock) = @inbounds val[i]
@inline get_external_value(val::Function, i, grid, clock) = val(i, grid, clock)

#####
##### Flather halo filling
#####

# The Flather condition for normal barotropic transport at a boundary:
#
#   East/North (right boundary):  Uᵇ = Uᵉˣᵗ + √(g H) ⋅ (ηᵇ − ηᵉˣᵗ)
#   West/South (left  boundary):  Uᵇ = Uᵉˣᵗ − √(g H) ⋅ (ηᵇ − ηᵉˣᵗ)
#
# The sign convention follows from the characteristic decomposition of the
# shallow water equations: the incoming Riemann invariant is prescribed from
# external data while the outgoing one radiates freely.
#
# Requires `model_fields` to contain:
#   - η :: free surface displacement field

@inline function _fill_east_halo!(j, k, grid, c, bc::FlatherOBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    i = grid.Nx + 1
    k_top = grid.Nz + 1
    flather = bc.classification.scheme

    g = flather.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶠᶜᵃ(i, j, k_top, grid, η)

    Uᵉˣᵗ = get_external_value(flather.external_values.U, j, grid, clock)
    ηᵉˣᵗ = get_external_value(flather.external_values.η, j, grid, clock)
    ηᵇ   = @inbounds η[grid.Nx, j, k_top]

    @inbounds c[i, j, k] = Uᵉˣᵗ + sqrt(g * H) * (ηᵇ - ηᵉˣᵗ)

    return nothing
end

@inline function _fill_west_halo!(j, k, grid, c, bc::FlatherOBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    k_top = grid.Nz + 1
    flather = bc.classification.scheme

    g = flather.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶠᶜᵃ(1, j, k_top, grid, η)

    Uᵉˣᵗ = get_external_value(flather.external_values.U, j, grid, clock)
    ηᵉˣᵗ = get_external_value(flather.external_values.η, j, grid, clock)
    ηᵇ   = @inbounds η[1, j, k_top]

    @inbounds c[1, j, k] = Uᵉˣᵗ - sqrt(g * H) * (ηᵇ - ηᵉˣᵗ)

    return nothing
end

@inline function _fill_north_halo!(i, k, grid, c, bc::FlatherOBC, ::Tuple{Any, Face, Any}, clock, model_fields)
    j = grid.Ny + 1
    k_top = grid.Nz + 1
    flather = bc.classification.scheme

    g = flather.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶜᶠᵃ(i, j, k_top, grid, η)

    Vᵉˣᵗ = get_external_value(flather.external_values.U, i, grid, clock)
    ηᵉˣᵗ = get_external_value(flather.external_values.η, i, grid, clock)
    ηᵇ   = @inbounds η[i, grid.Ny, k_top]

    @inbounds c[i, j, k] = Vᵉˣᵗ + sqrt(g * H) * (ηᵇ - ηᵉˣᵗ)

    return nothing
end

@inline function _fill_south_halo!(i, k, grid, c, bc::FlatherOBC, ::Tuple{Any, Face, Any}, clock, model_fields)
    k_top = grid.Nz + 1
    flather = bc.classification.scheme

    g = flather.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶜᶠᵃ(i, 1, k_top, grid, η)

    Vᵉˣᵗ = get_external_value(flather.external_values.U, i, grid, clock)
    ηᵉˣᵗ = get_external_value(flather.external_values.η, i, grid, clock)
    ηᵇ   = @inbounds η[i, 1, k_top]

    @inbounds c[i, 1, k] = Vᵉˣᵗ - sqrt(g * H) * (ηᵇ - ηᵉˣᵗ)

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
# Previous-timestep values φᵇⁿ and φ₁ⁿ are stored in separate arrays
# inside the Radiation struct, not in the field's halo, to avoid corruption
# by other kernels (e.g. the barotropic corrector).
#
# Adaptive nudging (Marchesiello et al. 2001):
#   - Outflow (Cₙ > 0): τ = relaxation_timescale (typically weak or Inf)
#   - Inflow  (Cₙ ≤ 0): τ = inflow_relaxation_timescale (typically strong)

@inline function orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation)
    # Diagnose phase speed Courant number (Orlanski 1976)
    ∂φ∂t = φ₁ⁿ⁺¹ - φ₁ⁿ
    ∂φ∂ξ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹

    # Cₙ = -(∂φ/∂t) / (∂φ/∂ξ) in the outward-normal direction
    # Guard against zero spatial gradient
    Cₙ_raw = ifelse(∂φ∂ξ == 0, zero(∂φ∂t), -∂φ∂t / ∂φ∂ξ)

    # Adaptive nudging: strong on inflow, weak on outflow
    τ = ifelse(Cₙ_raw > 0, radiation.outflow_relaxation_timescale, radiation.inflow_relaxation_timescale)
    τ̃ = Δt / τ

    # Clamp Courant number to [0, 1]
    Cₙ = max(zero(Cₙ_raw), min(one(Cₙ_raw), Cₙ_raw))

    # Implicit Orlanski radiation + nudging
    return (φᵇⁿ + Cₙ * φ₁ⁿ⁺¹ + τ̃ * φᵉˣᵗ) / (1 + Cₙ + τ̃)
end

# For "right" boundaries (east, north, top), the boundary index depends on
# the field location:
#   - Face fields: boundary face is at index N+1, interior at N, N-1
#   - Center fields: boundary halo is at index N, interior at N-1, N-2
# "Left" boundaries (west, south, bottom) are at 1 for both locations.

@inline boundary_index_right(::Face, N)   = N + 1
@inline boundary_index_right(::Center, N) = N
@inline boundary_index_left(::Face)   = 1
@inline boundary_index_left(::Center) = 1

@inline function _fill_east_halo!(j, k, grid, c, bc::RadiationOBC, loc, clock, model_fields)
    i = boundary_index_right(loc[1], grid.Nx)
    first_call = isinf(clock.last_stage_Δt)
    Δt = ifelse(first_call, zero(clock.last_stage_Δt), clock.last_stage_Δt)
    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, j, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i-1, j, k]       # first interior (new time)
        φ₂ⁿ⁺¹ = c[i-2, j, k]       # second interior (new time)
        φᵇⁿ   = ifelse(first_call, c[i, j, k], radiation.φᵇ[j, k])
        φ₁ⁿ   = ifelse(first_call, φ₁ⁿ⁺¹,      radiation.φ₁[j, k])

        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation)

        c[i, j, k]         = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[j, k] = φᵇⁿ⁺¹ # store for next time step
        radiation.φ₁[j, k] = φ₁ⁿ⁺¹ # store interior for next time step
    end

    return nothing
end

@inline function _fill_west_halo!(j, k, grid, c, bc::RadiationOBC, loc, clock, model_fields)
    iᵇ = boundary_index_left(loc[1])
    first_call = isinf(clock.last_stage_Δt)
    Δt = ifelse(first_call, zero(clock.last_stage_Δt), clock.last_stage_Δt)

    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, j, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[iᵇ+1, j, k]     # first interior (new time)
        φ₂ⁿ⁺¹ = c[iᵇ+2, j, k]     # second interior (new time)
        φᵇⁿ   = ifelse(first_call, c[iᵇ, j, k], radiation.φᵇ[j, k])
        φ₁ⁿ   = ifelse(first_call, φ₁ⁿ⁺¹,       radiation.φ₁[j, k])

        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation)

        c[iᵇ, j, k]        = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[j, k] = φᵇⁿ⁺¹ # store for next time step
        radiation.φ₁[j, k] = φ₁ⁿ⁺¹ # store interior for next time step
    end

    return nothing
end

@inline function _fill_north_halo!(i, k, grid, c, bc::RadiationOBC, loc, clock, model_fields)
    j = boundary_index_right(loc[2], grid.Ny)
    first_call = isinf(clock.last_stage_Δt)
    Δt = ifelse(first_call, zero(clock.last_stage_Δt), clock.last_stage_Δt)

    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, j-1, k]       # first interior (new time)
        φ₂ⁿ⁺¹ = c[i, j-2, k]       # second interior (new time)
        φᵇⁿ   = ifelse(first_call, c[i, j, k],  radiation.φᵇ[i, k])
        φ₁ⁿ   = ifelse(first_call, φ₁ⁿ⁺¹,       radiation.φ₁[i, k])

        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation)

        c[i, j, k]         = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[i, k] = φᵇⁿ⁺¹ # store for next time step
        radiation.φ₁[i, k] = φ₁ⁿ⁺¹ # store interior for next time step
    end

    return nothing
end

@inline function _fill_south_halo!(i, k, grid, c, bc::RadiationOBC, loc, clock, model_fields)
    jᵇ = boundary_index_left(loc[2])
    first_call = isinf(clock.last_stage_Δt)
    Δt = ifelse(first_call, zero(clock.last_stage_Δt), clock.last_stage_Δt)

    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, jᵇ+1, k]     # first interior (new time)
        φ₂ⁿ⁺¹ = c[i, jᵇ+2, k]     # second interior (new time)
        φᵇⁿ   = ifelse(first_call, c[i, jᵇ, k], radiation.φᵇ[i, k])
        φ₁ⁿ   = ifelse(first_call, φ₁ⁿ⁺¹,       radiation.φ₁[i, k])

        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation)

        c[i, jᵇ, k]        = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[i, k] = φᵇⁿ⁺¹ # store for next time step
        radiation.φ₁[i, k] = φ₁ⁿ⁺¹ # store interior for next time step
    end

    return nothing
end

@inline function _fill_top_halo!(i, j, grid, c, bc::RadiationOBC, loc, clock, model_fields)
    kᵇ = boundary_index_right(loc[3], grid.Nz)
    first_call = isinf(clock.last_stage_Δt)
    Δt = ifelse(first_call, zero(clock.last_stage_Δt), clock.last_stage_Δt)

    radiation = bc.classification.scheme
    
    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, j, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, j, kᵇ-1]       # first interior (new time)
        φ₂ⁿ⁺¹ = c[i, j, kᵇ-2]       # second interior (new time)
        φᵇⁿ   = ifelse(first_call, c[i, j, kᵇ], radiation.φᵇ[i, j])
        φ₁ⁿ   = ifelse(first_call, φ₁ⁿ⁺¹,       radiation.φ₁[i, j])
        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation)

        c[i, j, kᵇ]        = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[i, j] = φᵇⁿ⁺¹ # store for next time step
        radiation.φ₁[i, j] = φ₁ⁿ⁺¹ # store interior for next time step
    end

    return nothing
end

@inline function _fill_bottom_halo!(i, j, grid, c, bc::RadiationOBC, loc, clock, model_fields)
    kᵇ = boundary_index_left(loc[3])
    first_call = isinf(clock.last_stage_Δt)
    Δt = ifelse(first_call, zero(clock.last_stage_Δt), clock.last_stage_Δt)

    @inbounds begin
        radiation = bc.classification.scheme
        φᵉˣᵗ  = getbc(bc, i, j, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, j, kᵇ+1]     # first interior (new time)
        φ₂ⁿ⁺¹ = c[i, j, kᵇ+2]     # second interior (new time)
        φᵇⁿ   = ifelse(first_call, c[i, j, kᵇ], radiation.φᵇ[i, j])
        φ₁ⁿ   = ifelse(first_call, φ₁ⁿ⁺¹,       radiation.φ₁[i, j])
        φᵇⁿ⁺¹ = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation)

        c[i, j, kᵇ]        = φᵇⁿ⁺¹ # set boundary value
        radiation.φᵇ[i, j] = φᵇⁿ⁺¹ # store for next time step
        radiation.φ₁[i, j] = φ₁ⁿ⁺¹ # store interior for next time step
    end

    return nothing
end
