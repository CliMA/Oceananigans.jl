#####
##### GravityWaveRadiation (Flather 1976) and its companion SurfaceWaveRadiation (Chapman 1985)
#####

"""
    GravityWaveRadiation(; gravitational_acceleration = defaults.gravitational_acceleration)

Flather (1976) characteristic boundary condition for the shallow water equations.
Prescribes the incoming Riemann invariant while letting the outgoing one radiate freely:

    Uᵇ = Uᵉˣᵗ + √(g H) ⋅ (ηᵇ − ηᵉˣᵗ)

where `Uᵉˣᵗ` and `ηᵉˣᵗ` are external (prescribed) values, `ηᵇ` is the model free
surface at the boundary, and `H` is the water column depth.

`GravityWaveRadiation` is used as the `scheme` of a [`NormalFlowBoundaryCondition`](@ref) for the
barotropic transport; see also the convenience constructor [`GravityWaveRadiationBoundaryCondition`](@ref).
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
using Oceananigans.BoundaryConditions: GravityWaveRadiation

GravityWaveRadiation()

# output
GravityWaveRadiation{Float64}
└── gravitational_acceleration: 9.80665
```
"""
struct GravityWaveRadiation{FT}
    gravitational_acceleration :: FT
end

function GravityWaveRadiation(; gravitational_acceleration = defaults.gravitational_acceleration)
    return GravityWaveRadiation(gravitational_acceleration)
end

Adapt.adapt_structure(to, f::GravityWaveRadiation) =
    GravityWaveRadiation(adapt(to, f.gravitational_acceleration))

const GWNFBC = BoundaryCondition{<:NormalFlow{<:GravityWaveRadiation}}

Base.summary(::GravityWaveRadiation{FT}) where FT = "GravityWaveRadiation{$FT}"

function Base.show(io::IO, f::GravityWaveRadiation)
    print(io, summary(f), '\n')
    print(io, "└── gravitational_acceleration: ", prettysummary(f.gravitational_acceleration))
end

"""
    SurfaceWaveRadiation(; gravitational_acceleration = defaults.gravitational_acceleration)

Chapman (1985) radiation condition for the free surface displacement at an open boundary,
the standard companion of [`GravityWaveRadiation`](@ref): the boundary η radiates at the known
barotropic gravity-wave speed,

    ∂η/∂t ± √(g H) ∂η/∂n = 0

discretized implicitly (the form used by ROMS):

    ηᵇⁿ⁺¹ = (ηᵇⁿ + C η₁ⁿ⁺¹) / (1 + C),    C = √(g H) Δt / Δx

where `η₁` is the boundary-adjacent interior value. Letting the boundary η evolve frees
the surface pressure gradient at the boundary, which balanced flows require to cross it.

`SurfaceWaveRadiation` is used as the `scheme` of a [`ValueBoundaryCondition`](@ref) on the free
surface displacement `η`; see [`SurfaceWaveRadiationBoundaryCondition`](@ref). It is applied at every
barotropic substep, like `GravityWaveRadiation`.

References
==========
* Chapman, D. C. (1985). "Numerical treatment of cross-shelf open boundaries in a
  barotropic coastal ocean model." Journal of Physical Oceanography, 15(8), 1060-1075.

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: SurfaceWaveRadiation

SurfaceWaveRadiation()

# output
SurfaceWaveRadiation{Float64}
└── gravitational_acceleration: 9.80665
```
"""
struct SurfaceWaveRadiation{FT}
    gravitational_acceleration :: FT
end

function SurfaceWaveRadiation(; gravitational_acceleration = defaults.gravitational_acceleration)
    return SurfaceWaveRadiation(gravitational_acceleration)
end

Adapt.adapt_structure(to, c::SurfaceWaveRadiation) =
    SurfaceWaveRadiation(adapt(to, c.gravitational_acceleration))

const IGWVBC = BoundaryCondition{<:Value{<:SurfaceWaveRadiation}}

Base.summary(::SurfaceWaveRadiation{FT}) where FT = "SurfaceWaveRadiation{$FT}"

function Base.show(io::IO, c::SurfaceWaveRadiation)
    print(io, summary(c), '\n')
    print(io, "└── gravitational_acceleration: ", prettysummary(c.gravitational_acceleration))
end

@inline gravity_wave_boundary_condition(bc::GWNFBC) = true
@inline gravity_wave_boundary_condition(bc)         = false

#####
##### Convenience constructors
#####

"""
    GravityWaveRadiationBoundaryCondition(val; gravitational_acceleration = defaults.gravitational_acceleration, kwargs...)

Construct a `NormalFlowBoundaryCondition` with the [`GravityWaveRadiation`](@ref) scheme. `val` must be a 2-tuple `(U, η)` or a function
returning a 2-tuple, where `U` is the external barotropic transport and `η` is the external free surface displacement. Each
element of the tuple can be a number, array, or function (evaluated via `getbc`).

Example
=======

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: GravityWaveRadiationBoundaryCondition

bc = GravityWaveRadiationBoundaryCondition((0, 0))
bc isa Oceananigans.BoundaryConditions.BoundaryCondition

# output
true
```
"""
function GravityWaveRadiationBoundaryCondition(val; gravitational_acceleration = defaults.gravitational_acceleration, kwargs...)
    validate_gravity_wave_condition(val)
    scheme = GravityWaveRadiation(; gravitational_acceleration)
    return NormalFlowBoundaryCondition(val; scheme, kwargs...)
end

GravityWaveRadiationBoundaryCondition(U, η; kwargs...) = GravityWaveRadiationBoundaryCondition((U, η); kwargs...)

"""
    SurfaceWaveRadiationBoundaryCondition(; gravitational_acceleration = defaults.gravitational_acceleration)

Construct a `ValueBoundaryCondition` with the [`SurfaceWaveRadiation`](@ref) scheme for the free surface displacement `η`
at an open boundary. Pair with [`GravityWaveRadiationBoundaryCondition`](@ref) on the barotropic transport.
"""
SurfaceWaveRadiationBoundaryCondition(; gravitational_acceleration = defaults.gravitational_acceleration) =
    ValueBoundaryCondition(0; scheme = SurfaceWaveRadiation(; gravitational_acceleration))


function validate_gravity_wave_condition(val)
    if val isa Union{Tuple, NamedTuple}
        length(val) == 2 || throw(ArgumentError(
            "GravityWaveRadiation boundary condition requires a 2-tuple (U, η) for " *
            "external transport and free surface, got a $(length(val))-tuple."))
    elseif !(val isa Function)
        throw(ArgumentError(
            "GravityWaveRadiationBoundaryCondition requires a 2-tuple (U, η) or a function " *
            "returning a 2-tuple, where U is the external barotropic transport " *
            "and η is the external free surface displacement. " *
            "Got an argument of type $(typeof(val))."))
    end
    return nothing
end

#####
##### GravityWaveRadiation halo filling
#####

# During initialization (no clock/model_fields available yet), fill halos with zero.
@inline   _fill_east_halo!(j, k, grid, c, bc::GWNFBC, loc) = @inbounds c[grid.Nx + 1, j, k] = zero(grid)
@inline   _fill_west_halo!(j, k, grid, c, bc::GWNFBC, loc) = @inbounds c[1, j, k]           = zero(grid)
@inline  _fill_north_halo!(i, k, grid, c, bc::GWNFBC, loc) = @inbounds c[i, grid.Ny + 1, k] = zero(grid)
@inline  _fill_south_halo!(i, k, grid, c, bc::GWNFBC, loc) = @inbounds c[i, 1, k]           = zero(grid)

# The GravityWaveRadiation condition for normal barotropic transport at a boundary:
#
#   East/North (right boundary):  Uᵇ = Uᵉˣᵗ + √(g H) ⋅ (ηᵇ − ηᵉˣᵗ)
#   West/South (left  boundary):  Uᵇ = Uᵉˣᵗ − √(g H) ⋅ (ηᵇ − ηᵉˣᵗ)
#
# The sign convention follows from the characteristic decomposition of the shallow water equations:
# the incoming Riemann invariant is prescribed from external data while the outgoing one radiates freely.
#
# The boundary condition value (accessed via getbc) must return a 2-tuple (U, η) of external transport
# and free surface values.
#
# ηᵇ is the face average of the two adjacent cells (ROMS form): under η's default mirror fill this equals the
# interior sample, while an SurfaceWaveRadiation condition on the boundary row couples into the transport through the average.
#
# Requires `model_fields` to contain:
#   - η :: free surface displacement field

@inline function _fill_east_halo!(j, k, grid, c, bc::GWNFBC, ::FAA, clock, model_fields)
    i = grid.Nx + 1
    kᴺ⁺¹ = grid.Nz + 1
    gravity_wave = bc.classification.scheme

    g = gravity_wave.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶠᶜᵃ(i, j, kᴺ⁺¹, grid, η)

    Uᵉˣᵗ, ηᵉˣᵗ = getbc(bc, j, k, grid, clock, model_fields)
    ηᵇ = ℑxᶠᵃᵃ(i, j, kᴺ⁺¹, grid, η)

    @inbounds c[i, j, k] = ifelse(H <= zero(H), zero(grid), Uᵉˣᵗ + sqrt(g * max(H, zero(H))) * (ηᵇ - ηᵉˣᵗ))

    return nothing
end

@inline function _fill_west_halo!(j, k, grid, c, bc::GWNFBC, ::FAA, clock, model_fields)
    kᴺ⁺¹ = grid.Nz + 1
    gravity_wave = bc.classification.scheme

    g = gravity_wave.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶠᶜᵃ(1, j, kᴺ⁺¹, grid, η)

    Uᵉˣᵗ, ηᵉˣᵗ = getbc(bc, j, k, grid, clock, model_fields)
    ηᵇ = ℑxᶠᵃᵃ(1, j, kᴺ⁺¹, grid, η)

    @inbounds c[1, j, k] = ifelse(H <= zero(H), zero(grid), Uᵉˣᵗ - sqrt(g * max(H, zero(H))) * (ηᵇ - ηᵉˣᵗ))

    return nothing
end

@inline function _fill_north_halo!(i, k, grid, c, bc::GWNFBC, ::AFA, clock, model_fields)
    j = grid.Ny + 1
    kᴺ⁺¹ = grid.Nz + 1
    gravity_wave = bc.classification.scheme

    g = gravity_wave.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶜᶠᵃ(i, j, kᴺ⁺¹, grid, η)

    Vᵉˣᵗ, ηᵉˣᵗ = getbc(bc, i, k, grid, clock, model_fields)
    ηᵇ = ℑyᵃᶠᵃ(i, j, kᴺ⁺¹, grid, η)

    @inbounds c[i, j, k] = ifelse(H <= zero(H), zero(grid), Vᵉˣᵗ + sqrt(g * max(H, zero(H))) * (ηᵇ - ηᵉˣᵗ))

    return nothing
end

@inline function _fill_south_halo!(i, k, grid, c, bc::GWNFBC, ::AFA, clock, model_fields)
    kᴺ⁺¹ = grid.Nz + 1
    gravity_wave = bc.classification.scheme

    g = gravity_wave.gravitational_acceleration
    η = model_fields.η
    H = column_depthᶜᶠᵃ(i, 1, kᴺ⁺¹, grid, η)

    Vᵉˣᵗ, ηᵉˣᵗ = getbc(bc, i, k, grid, clock, model_fields)
    ηᵇ = ℑyᵃᶠᵃ(i, 1, kᴺ⁺¹, grid, η)

    @inbounds c[i, 1, k] = ifelse(H <= zero(H), zero(grid), Vᵉˣᵗ - sqrt(g * max(H, zero(H))) * (ηᵇ - ηᵉˣᵗ))

    return nothing
end

#####
##### SurfaceWaveRadiation halo filling — implicit gravity-wave radiation of the free surface
#####

@inline function _fill_west_halo!(j, k, grid, η, bc::IGWVBC, ::CAA, clock, model_fields)
    anchored_fill(clock) || return nothing
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    g = bc.classification.scheme.gravitational_acceleration
    kᴺ⁺¹ = grid.Nz + 1

    @inbounds begin
        η₁ = η[1, j, k]
        H  = column_depthᶜᶜᵃ(1, j, kᴺ⁺¹, grid, η)
        C  = sqrt(g * max(H, zero(H))) * Δt / Δxᶠᶜᶜ(1, j, k, grid)
        ηᵇ = ifelse(first_call, η₁, η[0, j, k]) # zero-gradient initialization
        η[0, j, k] = (ηᵇ + C * η₁) / (1 + C)
    end

    return nothing
end

@inline function _fill_east_halo!(j, k, grid, η, bc::IGWVBC, ::CAA, clock, model_fields)
    anchored_fill(clock) || return nothing
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    g = bc.classification.scheme.gravitational_acceleration
    i = grid.Nx + 1
    kᴺ⁺¹ = grid.Nz + 1

    @inbounds begin
        η₁ = η[i-1, j, k]
        H  = column_depthᶜᶜᵃ(i-1, j, kᴺ⁺¹, grid, η)
        C  = sqrt(g * max(H, zero(H))) * Δt / Δxᶠᶜᶜ(i, j, k, grid)
        ηᵇ = ifelse(first_call, η₁, η[i, j, k]) # zero-gradient initialization
        η[i, j, k] = (ηᵇ + C * η₁) / (1 + C)
    end

    return nothing
end

@inline function _fill_south_halo!(i, k, grid, η, bc::IGWVBC, ::ACA, clock, model_fields)
    anchored_fill(clock) || return nothing
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    g = bc.classification.scheme.gravitational_acceleration
    kᴺ⁺¹ = grid.Nz + 1

    @inbounds begin
        η₁ = η[i, 1, k]
        H  = column_depthᶜᶜᵃ(i, 1, kᴺ⁺¹, grid, η)
        C  = sqrt(g * max(H, zero(H))) * Δt / Δyᶜᶠᶜ(i, 1, k, grid)
        ηᵇ = ifelse(first_call, η₁, η[i, 0, k]) # zero-gradient initialization
        η[i, 0, k] = (ηᵇ + C * η₁) / (1 + C)
    end

    return nothing
end

@inline function _fill_north_halo!(i, k, grid, η, bc::IGWVBC, ::ACA, clock, model_fields)
    anchored_fill(clock) || return nothing
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    g = bc.classification.scheme.gravitational_acceleration
    j = grid.Ny + 1
    kᴺ⁺¹ = grid.Nz + 1

    @inbounds begin
        η₁ = η[i, j-1, k]
        H  = column_depthᶜᶜᵃ(i, j-1, kᴺ⁺¹, grid, η)
        C  = sqrt(g * max(H, zero(H))) * Δt / Δyᶜᶠᶜ(i, j, k, grid)
        ηᵇ = ifelse(first_call, η₁, η[i, j, k]) # zero-gradient initialization
        η[i, j, k] = (ηᵇ + C * η₁) / (1 + C)
    end

    return nothing
end
