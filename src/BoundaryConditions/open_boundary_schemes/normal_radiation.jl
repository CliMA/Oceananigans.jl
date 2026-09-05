#####
##### NormalRadiation (based on Orlanski 1976) open boundary scheme
#####

"""
    NormalRadiation(; inflow_timescale = 0, outflow_timescale = Inf, use_boundary_velocity = false)

Orlanski (1976) radiation condition with locally-diagnosed phase speed
and adaptive nudging (Marchesiello et al. 2001):

    ∂φ/∂t + cₙ ⋅ ∂φ/∂n = - (φ - φᵉˣᵗ) / τ

where `cₙ = −(∂φ/∂t) / (∂φ/∂n)` is diagnosed from interior values, clamped to `[0, Δx/Δt]`.
Inflow vs outflow is decided from the boundary-normal velocity: for `NormalFlow` fields the boundary
value itself, for `Value` fields (tracers, tangential velocities) a `model_fields` velocity taken one
cell into the interior, or at the boundary face when `use_boundary_velocity = true` (the interior value
is a prognostic velocity, the boundary value is itself radiation-extrapolated; see Orlanski 1976 and the
MITgcm `obcs` implementation). `τ = τ_in` and `cₙ = 0` on inflow, `τ = τ_out` on outflow.

`use_boundary_velocity` only affects `Value` boundary conditions (tracers, tangential velocities). It is
ignored for a `NormalFlowBoundaryCondition`, which radiates the normal velocity using its own boundary value.

`NormalRadiation` is used as the `scheme` of a [`ValueBoundaryCondition`](@ref) for `Center`-located
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
using Oceananigans.BoundaryConditions: NormalRadiation

NormalRadiation(outflow_timescale = 360 * 86400, inflow_timescale = 86400)

# output
NormalRadiation{Float64}
├── inflow_timescale: 86400.0
├── outflow_timescale: 3.1104e7
└── use_boundary_velocity: false
```
"""
struct NormalRadiation{FT, S}
    outflow_timescale :: FT
    inflow_timescale  :: FT
    use_boundary_velocity :: Bool # advect Value fields with the boundary-face velocity instead of one cell in
    φᵇ  :: S  # anchor boundary value (2D array or nothing)
    φ₁  :: S  # anchor interior value (2D array or nothing)
    φ₁ˡ :: S  # latest interior value (2D array or nothing)
end

function NormalRadiation(FT = defaults.FloatType;
                   inflow_timescale = 0,
                   outflow_timescale = Inf,
                   use_boundary_velocity = false)

    outflow_timescale = convert(FT, outflow_timescale)
    inflow_timescale = convert(FT, inflow_timescale)
    return NormalRadiation(outflow_timescale, inflow_timescale, use_boundary_velocity, nothing, nothing, nothing)
end

Adapt.adapt_structure(to, r::NormalRadiation) =
    NormalRadiation(adapt(to, r.outflow_timescale),
              adapt(to, r.inflow_timescale),
              r.use_boundary_velocity,
              adapt(to, r.φᵇ),
              adapt(to, r.φ₁),
              adapt(to, r.φ₁ˡ))

Base.summary(::NormalRadiation{FT}) where FT = "NormalRadiation{$FT}"

function Base.show(io::IO, r::NormalRadiation)
    print(io, summary(r), '\n')
    print(io, "├── inflow_timescale: ",  prettysummary(r.inflow_timescale), '\n')
    print(io, "├── outflow_timescale: ", prettysummary(r.outflow_timescale), '\n')
    print(io, "└── use_boundary_velocity: ", r.use_boundary_velocity)
end

const RVBC  = BoundaryCondition{<:Value{<:NormalRadiation}}
const RNFBC = BoundaryCondition{<:NormalFlow{<:NormalRadiation}}
const RBC   = Union{RVBC, RNFBC}

#####
##### NormalRadiation storage allocation during BC regularization
#####

# Allocate the 2D storage array holding the previous-timestep interior value φ₁ⁿ
# needed by the Orlanski phase-speed diagnosis.
function materialize_radiation_storage(radiation::NormalRadiation, grid, loc, dim)
    FT = eltype(grid)
    Sx, Sy, Sz = size(grid, loc) # loc-aware: Face on Bounded gives N+1, matching the kernel range
    arch = architecture(grid)

    tangential_size = dim == 1 ? (Sy, Sz) :
                      dim == 2 ? (Sx, Sz) :
                                 (Sx, Sy)

    φᵇ  = on_architecture(arch, zeros(FT, tangential_size...))
    φ₁  = on_architecture(arch, zeros(FT, tangential_size...))
    φ₁ˡ = on_architecture(arch, zeros(FT, tangential_size...))

    return NormalRadiation(radiation.outflow_timescale,
                     radiation.inflow_timescale,
                     radiation.use_boundary_velocity,
                     φᵇ, φ₁, φ₁ˡ)
end

rebuild_classification(::Value, scheme) = Value(scheme)
rebuild_classification(::NormalFlow, scheme) = NormalFlow(scheme)

# Hook into the regularization pipeline to allocate NormalRadiation storage
function regularize_boundary_condition(bc::RBC, grid, loc, dim, args...)
    regularized_condition = regularize_boundary_condition(bc.condition, grid, loc, dim, args...)
    radiation = bc.classification.scheme
    materialized_radiation = materialize_radiation_storage(radiation, grid, loc, dim)
    classification = rebuild_classification(bc.classification, materialized_radiation)
    return BoundaryCondition(classification, regularized_condition)
end

#####
##### NormalRadiation halo filling — Orlanski (1976) with Marchesiello et al. (2001) nudging
#####

# Inflow/outflow from boundary-normal velocity, not phase-speed sign: avoids extremum-exit blowup where ∂φ/∂ξ → 0.

@inline function orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
    ∂t_φ = φ₁ⁿ⁺¹ - φ₁ⁿ
    ∂ξ_φ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹

    # guard ∂ξ_φ == 0 (extremum at boundary face)
    Cᶜ = ifelse(∂ξ_φ == 0, zero(∂t_φ), - ∂t_φ / ∂ξ_φ)

    Cₙ = ifelse(outflow, max(zero(Cᶜ), min(one(Cᶜ), max(Cᶜ, Cᵃ))), zero(Cᶜ))

    τ = ifelse(outflow, radiation.outflow_timescale, radiation.inflow_timescale)
    τ̃ = Δt / τ

    φᵇⁿ⁺¹ = (φᵇⁿ + Cₙ * φ₁ⁿ⁺¹ + τ̃ * φᵉˣᵗ) / (1 + Cₙ + τ̃)

    return ifelse(τ == 0, φᵉˣᵗ, φᵇⁿ⁺¹)
end

# NormalFlow: radiate boundary face; Value: radiate first halo cell.

@inline function radiate_east_halo!(iᵇ, j, k, grid, c, bc, Uₙ, loc, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme
    ℓx, ℓy, ℓz = loc

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, j, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[iᵇ-1, j, k]
        φ₂ⁿ⁺¹ = c[iᵇ-2, j, k]

        φᵇᵃ = ifelse(anchored, c[iᵇ, j, k], radiation.φᵇ[j, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[j, k], radiation.φ₁[j, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δxᶠᶜᶜ(iᵇ, j, k, grid)
        outflow = Uᵃ >= 0

        φᵇⁿ⁺¹  = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
        closed = immersed_peripheral_node(grid.Nx, j, k, grid, Center(), ℓy, ℓz)
        c[iᵇ, j, k]         = ifelse(closed, zero(grid), φᵇⁿ⁺¹)
        radiation.φᵇ[j, k]  = φᵇⁿ   # anchor old values for sub-stages
        radiation.φ₁[j, k]  = φ₁ⁿ
        radiation.φ₁ˡ[j, k] = φ₁ⁿ⁺¹ # latest interior, promoted at next anchored fill
    end

    return nothing
end

@inline function radiate_west_halo!(iᵇ, j, k, grid, c, bc, Uₙ, loc, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme
    ℓx, ℓy, ℓz = loc

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, j, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[iᵇ+1, j, k]
        φ₂ⁿ⁺¹ = c[iᵇ+2, j, k]

        φᵇᵃ = ifelse(anchored, c[iᵇ, j, k], radiation.φᵇ[j, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[j, k], radiation.φ₁[j, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δxᶠᶜᶜ(iᵇ + 1, j, k, grid)
        outflow = Uᵃ <= 0

        φᵇⁿ⁺¹  = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
        closed = immersed_peripheral_node(1, j, k, grid, Center(), ℓy, ℓz)
        c[iᵇ, j, k]         = ifelse(closed, zero(grid), φᵇⁿ⁺¹)
        radiation.φᵇ[j, k]  = φᵇⁿ   # anchor old values for sub-stages
        radiation.φ₁[j, k]  = φ₁ⁿ
        radiation.φ₁ˡ[j, k] = φ₁ⁿ⁺¹ # latest interior, promoted at next anchored fill
    end

    return nothing
end

@inline function radiate_north_halo!(jᵇ, i, k, grid, c, bc, Uₙ, loc, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme
    ℓx, ℓy, ℓz = loc

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, jᵇ-1, k]
        φ₂ⁿ⁺¹ = c[i, jᵇ-2, k]

        φᵇᵃ = ifelse(anchored, c[i, jᵇ, k], radiation.φᵇ[i, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[i, k], radiation.φ₁[i, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δyᶜᶠᶜ(i, jᵇ, k, grid)
        outflow = Uᵃ >= 0

        φᵇⁿ⁺¹  = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
        closed = immersed_peripheral_node(i, grid.Ny, k, grid, ℓx, Center(), ℓz)
        c[i, jᵇ, k]         = ifelse(closed, zero(grid), φᵇⁿ⁺¹)
        radiation.φᵇ[i, k]  = φᵇⁿ   # anchor old values for sub-stages
        radiation.φ₁[i, k]  = φ₁ⁿ
        radiation.φ₁ˡ[i, k] = φ₁ⁿ⁺¹ # latest interior, promoted at next anchored fill
    end

    return nothing
end

@inline function radiate_south_halo!(jᵇ, i, k, grid, c, bc, Uₙ, loc, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme
    ℓx, ℓy, ℓz = loc

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, jᵇ+1, k]
        φ₂ⁿ⁺¹ = c[i, jᵇ+2, k]

        φᵇᵃ = ifelse(anchored, c[i, jᵇ, k], radiation.φᵇ[i, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[i, k], radiation.φ₁[i, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δyᶜᶠᶜ(i, jᵇ + 1, k, grid)
        outflow = Uᵃ <= 0

        φᵇⁿ⁺¹  = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
        closed = immersed_peripheral_node(i, 1, k, grid, ℓx, Center(), ℓz)
        c[i, jᵇ, k]         = ifelse(closed, zero(grid), φᵇⁿ⁺¹)
        radiation.φᵇ[i, k]  = φᵇⁿ   # anchor old values for sub-stages
        radiation.φ₁[i, k]  = φ₁ⁿ
        radiation.φ₁ˡ[i, k] = φ₁ⁿ⁺¹ # latest interior, promoted at next anchored fill
    end

    return nothing
end

@inline function radiate_top_halo!(kᵇ, i, j, grid, c, bc, Uₙ, loc, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme
    ℓx, ℓy, ℓz = loc

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, j, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, j, kᵇ-1]
        φ₂ⁿ⁺¹ = c[i, j, kᵇ-2]

        φᵇᵃ = ifelse(anchored, c[i, j, kᵇ], radiation.φᵇ[i, j])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[i, j], radiation.φ₁[i, j])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δzᶜᶜᶠ(i, j, kᵇ, grid)
        outflow = Uᵃ >= 0

        φᵇⁿ⁺¹  = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
        closed = immersed_peripheral_node(i, j, grid.Nz, grid, ℓx, ℓy, Center())
        c[i, j, kᵇ]         = ifelse(closed, zero(grid), φᵇⁿ⁺¹)
        radiation.φᵇ[i, j]  = φᵇⁿ   # anchor old values for sub-stages
        radiation.φ₁[i, j]  = φ₁ⁿ
        radiation.φ₁ˡ[i, j] = φ₁ⁿ⁺¹ # latest interior, promoted at next anchored fill
    end

    return nothing
end

@inline function radiate_bottom_halo!(kᵇ, i, j, grid, c, bc, Uₙ, loc, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme
    ℓx, ℓy, ℓz = loc

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, j, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, j, kᵇ+1]
        φ₂ⁿ⁺¹ = c[i, j, kᵇ+2]

        φᵇᵃ = ifelse(anchored, c[i, j, kᵇ], radiation.φᵇ[i, j])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[i, j], radiation.φ₁[i, j])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)
        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δzᶜᶜᶠ(i, j, kᵇ + 1, grid)
        outflow = Uᵃ <= 0

        φᵇⁿ⁺¹  = orlanski_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
        closed = immersed_peripheral_node(i, j, 1, grid, ℓx, ℓy, Center())
        c[i, j, kᵇ]         = ifelse(closed, zero(grid), φᵇⁿ⁺¹)
        radiation.φᵇ[i, j]  = φᵇⁿ   # anchor old values for sub-stages
        radiation.φ₁[i, j]  = φ₁ⁿ
        radiation.φ₁ˡ[i, j] = φ₁ⁿ⁺¹ # latest interior, promoted at next anchored fill
    end

    return nothing
end

# NormalFlow fields radiate with their own boundary value as advecting velocity (Uₙ = nothing);
# Value fields (tracers, tangential velocities) are advected by the boundary-normal velocity
@inline   _fill_east_halo!(j, k, grid, c, bc::RNFBC, loc::FAA, clock, model_fields) =   radiate_east_halo!(grid.Nx+1, j, k, grid, c, bc, nothing, loc, clock, model_fields)
@inline   _fill_west_halo!(j, k, grid, c, bc::RNFBC, loc::FAA, clock, model_fields) =   radiate_west_halo!(1,         j, k, grid, c, bc, nothing, loc, clock, model_fields)
@inline  _fill_north_halo!(i, k, grid, c, bc::RNFBC, loc::AFA, clock, model_fields) =  radiate_north_halo!(grid.Ny+1, i, k, grid, c, bc, nothing, loc, clock, model_fields)
@inline  _fill_south_halo!(i, k, grid, c, bc::RNFBC, loc::AFA, clock, model_fields) =  radiate_south_halo!(1,         i, k, grid, c, bc, nothing, loc, clock, model_fields)
@inline    _fill_top_halo!(i, j, grid, c, bc::RNFBC, loc::AAF, clock, model_fields) =    radiate_top_halo!(grid.Nz+1, i, j, grid, c, bc, nothing, loc, clock, model_fields)
@inline _fill_bottom_halo!(i, j, grid, c, bc::RNFBC, loc::AAF, clock, model_fields) = radiate_bottom_halo!(1,         i, j, grid, c, bc, nothing, loc, clock, model_fields)

# Clockless fallback: more specific than the NFBC catch-all, so getbc is never called without clock on GPU.
@inline   _fill_east_halo!(j, k, grid, c, bc::RNFBC, loc, args...) = nothing
@inline   _fill_west_halo!(j, k, grid, c, bc::RNFBC, loc, args...) = nothing
@inline  _fill_north_halo!(i, k, grid, c, bc::RNFBC, loc, args...) = nothing
@inline  _fill_south_halo!(i, k, grid, c, bc::RNFBC, loc, args...) = nothing
@inline    _fill_top_halo!(i, j, grid, c, bc::RNFBC, loc, args...) = nothing
@inline _fill_bottom_halo!(i, j, grid, c, bc::RNFBC, loc, args...) = nothing

# Advect Value fields with the boundary-face velocity (`use_boundary_velocity`) or the one-cell-interior
# velocity (default): the interior value is prognostic, the boundary value is radiation-extrapolated.
@inline radiation_velocity_index(bc, boundary_index, interior_index) =
    ifelse(bc.classification.scheme.use_boundary_velocity, boundary_index, interior_index)

@inline _fill_east_halo!(j, k, grid, c, bc::RVBC, loc::CAA, clock, model_fields) =
    radiate_east_halo!(grid.Nx+1, j, k, grid, c, bc, @inbounds(model_fields.u[radiation_velocity_index(bc, grid.Nx+1, grid.Nx), j, k]), loc, clock, model_fields)
@inline _fill_west_halo!(j, k, grid, c, bc::RVBC, loc::CAA, clock, model_fields) =
    radiate_west_halo!(0, j, k, grid, c, bc, @inbounds(model_fields.u[radiation_velocity_index(bc, 1, 2), j, k]), loc, clock, model_fields)
@inline _fill_north_halo!(i, k, grid, c, bc::RVBC, loc::ACA, clock, model_fields) =
    radiate_north_halo!(grid.Ny+1, i, k, grid, c, bc, @inbounds(model_fields.v[i, radiation_velocity_index(bc, grid.Ny+1, grid.Ny), k]), loc, clock, model_fields)
@inline _fill_south_halo!(i, k, grid, c, bc::RVBC, loc::ACA, clock, model_fields) =
    radiate_south_halo!(0, i, k, grid, c, bc, @inbounds(model_fields.v[i, radiation_velocity_index(bc, 1, 2), k]), loc, clock, model_fields)
@inline _fill_top_halo!(i, j, grid, c, bc::RVBC, loc::AAC, clock, model_fields) =
    radiate_top_halo!(grid.Nz+1, i, j, grid, c, bc, @inbounds(model_fields.w[i, j, radiation_velocity_index(bc, grid.Nz+1, grid.Nz)]), loc, clock, model_fields)
@inline _fill_bottom_halo!(i, j, grid, c, bc::RVBC, loc::AAC, clock, model_fields) =
    radiate_bottom_halo!(0, i, j, grid, c, bc, @inbounds(model_fields.w[i, j, radiation_velocity_index(bc, 1, 2)]), loc, clock, model_fields)
