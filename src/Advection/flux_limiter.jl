using Oceananigans.BoundaryConditions: fill_halo_regions!

#####
##### Flux limiting for bounds-preserving WENO
#####
#
# The rescaling of Zhang and Shu (2010) keeps a forward Euler update within `bounds` only while the
# Courant number stays below ω̂₁. A larger time step, or a Wicker–Skamarock stage that updates the
# step-start state with fluxes built from a later stage, can remove more from a cell than it holds.
# The flux limiter scales each cell's outgoing fluxes by the fraction `r = min(1, available / removed)`
# of them the coming update can afford: a conservative, zeroth-order flux-corrected transport limiter
# that keeps the tracer non-negative for any time step, at first-order accuracy where it acts.

"""
    FluxLimiter(fraction, provisional_fraction)

The `flux_limiter` of a [`BoundsPreservation`](@ref), which keeps the tracer non-negative by limiting each
cell's outgoing fluxes: the fraction ``r ∈ [0, 1]`` of them the coming update applies (`fraction`), and the
fractions of the previous limiter pass (`provisional_fraction`), which scale the incoming fluxes credited to
a cell. The fields are `nothing` until the scheme is materialized on a grid.

Each face flux is scaled by the fraction of the cell it leaves, so the limited fluxes stay conservative.
The fractions are computed by [`update_flux_limiter!`](@ref) before the fluxes are evaluated.
"""
struct FluxLimiter{R}
    fraction :: R              # r, the fraction of each cell's outgoing fluxes the coming update applies
    provisional_fraction :: R  # r from the previous pass, which scales the incoming fluxes credited to a cell
end

Adapt.adapt_structure(to, limiter::FluxLimiter) =
    FluxLimiter(Adapt.adapt(to, limiter.fraction), Adapt.adapt(to, limiter.provisional_fraction))

Architectures.on_architecture(to, limiter::FluxLimiter) =
    FluxLimiter(on_architecture(to, limiter.fraction), on_architecture(to, limiter.provisional_fraction))

const FluxLimitedBounds = BoundsPreservation{<:Any, <:Any, <:FluxLimiter}
const FluxLimitedWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:FluxLimitedBounds}

Base.show(io::IO, bounds::FluxLimitedBounds) =
    print(io, "(", bounds.minimum_value, ", ", bounds.maximum_value, "), flux_limiter=true")

#####
##### Limited face fluxes and their divergence
#####

# Each face flux is scaled by the fraction of its upwind cell.
@inline function upwind_flux_fraction_x(i, j, k, advection, u)
    r = advection.bounds.flux_limiter.fraction
    return @inbounds ifelse(u[i, j, k] ≥ 0, r[i-1, j, k], r[i, j, k])
end

@inline function upwind_flux_fraction_y(i, j, k, advection, v)
    r = advection.bounds.flux_limiter.fraction
    return @inbounds ifelse(v[i, j, k] ≥ 0, r[i, j-1, k], r[i, j, k])
end

@inline function upwind_flux_fraction_z(i, j, k, advection, w)
    r = advection.bounds.flux_limiter.fraction
    return @inbounds ifelse(w[i, j, k] ≥ 0, r[i, j, k-1], r[i, j, k])
end

# Support for Flat directions, which have no neighbors to index
@inline limited_face_flux_x(i, j, k, grid::XFlatGrid, advection, args...) = zero(grid)
@inline limited_face_flux_y(i, j, k, grid::YFlatGrid, advection, args...) = zero(grid)
@inline limited_face_flux_z(i, j, k, grid::ZFlatGrid, advection, args...) = zero(grid)

@inline limited_face_flux_x(i, j, k, grid, advection, ρ, u, c) =
    upwind_flux_fraction_x(i, j, k, advection, u) * bounded_face_flux_x(i, j, k, grid, advection, ρ, u, c)

@inline limited_face_flux_y(i, j, k, grid, advection, ρ, v, c) =
    upwind_flux_fraction_y(i, j, k, advection, v) * bounded_face_flux_y(i, j, k, grid, advection, ρ, v, c)

@inline limited_face_flux_z(i, j, k, grid, advection, ρ, w, c) =
    upwind_flux_fraction_z(i, j, k, advection, w) * bounded_face_flux_z(i, j, k, grid, advection, ρ, w, c)

@inline limited_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, u, c) =
    limited_face_flux_x(i+1, j, k, grid, advection, ρ, u, c) - limited_face_flux_x(i, j, k, grid, advection, ρ, u, c)

@inline limited_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, v, c) =
    limited_face_flux_y(i, j+1, k, grid, advection, ρ, v, c) - limited_face_flux_y(i, j, k, grid, advection, ρ, v, c)

@inline limited_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, w, c) =
    limited_face_flux_z(i, j, k+1, grid, advection, ρ, w, c) - limited_face_flux_z(i, j, k, grid, advection, ρ, w, c)

@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::FluxLimitedWENO, ::ZeroU, ::ZeroField) = zero(grid)

# With every fraction equal to one this is the bounds-preserving divergence.
@inline function div_Uc(i, j, k, grid, advection::FluxLimitedWENO, U, c)
    div_x = limited_tracer_flux_divergence_x(i, j, k, grid, advection, 1, U.u, c)
    div_y = limited_tracer_flux_divergence_y(i, j, k, grid, advection, 1, U.v, c)
    div_z = limited_tracer_flux_divergence_z(i, j, k, grid, advection, 1, U.w, c)

    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

#####
##### Flux fractions
#####

# Tracer the unlimited outgoing fluxes remove from cell (i, j, k), per unit volume and time.
@inline function outgoing_tracer_flux(i, j, k, grid, advection, ρ, U, c)
    Fx⁻ = bounded_face_flux_x(i,   j, k, grid, advection, ρ, U.u, c)
    Fx⁺ = bounded_face_flux_x(i+1, j, k, grid, advection, ρ, U.u, c)
    Fy⁻ = bounded_face_flux_y(i, j,   k, grid, advection, ρ, U.v, c)
    Fy⁺ = bounded_face_flux_y(i, j+1, k, grid, advection, ρ, U.v, c)
    Fz⁻ = bounded_face_flux_z(i, j, k,   grid, advection, ρ, U.w, c)
    Fz⁺ = bounded_face_flux_z(i, j, k+1, grid, advection, ρ, U.w, c)
    outgoing = max(0, Fx⁺) - min(0, Fx⁻) + max(0, Fy⁺) - min(0, Fy⁻) + max(0, Fz⁺) - min(0, Fz⁻)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * outgoing
end

# Tracer the incoming fluxes bring into cell (i, j, k) through its two faces in one direction, each scaled
# by the fraction `r` of the cell it leaves. A Flat direction has no neighbors (and no halo to index).
@inline limited_incoming_flux_x(i, j, k, grid::XFlatGrid, advection, args...) = zero(grid)
@inline limited_incoming_flux_y(i, j, k, grid::YFlatGrid, advection, args...) = zero(grid)
@inline limited_incoming_flux_z(i, j, k, grid::ZFlatGrid, advection, args...) = zero(grid)

@inline function limited_incoming_flux_x(i, j, k, grid, advection, ρ, u, c, r)
    F⁻ = bounded_face_flux_x(i,   j, k, grid, advection, ρ, u, c)
    F⁺ = bounded_face_flux_x(i+1, j, k, grid, advection, ρ, u, c)
    return @inbounds max(0, F⁻) * r[i-1, j, k] - min(0, F⁺) * r[i+1, j, k]
end

@inline function limited_incoming_flux_y(i, j, k, grid, advection, ρ, v, c, r)
    F⁻ = bounded_face_flux_y(i, j,   k, grid, advection, ρ, v, c)
    F⁺ = bounded_face_flux_y(i, j+1, k, grid, advection, ρ, v, c)
    return @inbounds max(0, F⁻) * r[i, j-1, k] - min(0, F⁺) * r[i, j+1, k]
end

@inline function limited_incoming_flux_z(i, j, k, grid, advection, ρ, w, c, r)
    F⁻ = bounded_face_flux_z(i, j, k,   grid, advection, ρ, w, c)
    F⁺ = bounded_face_flux_z(i, j, k+1, grid, advection, ρ, w, c)
    return @inbounds max(0, F⁻) * r[i, j, k-1] - min(0, F⁺) * r[i, j, k+1]
end

# Tracer the limited incoming fluxes bring into cell (i, j, k), per unit volume and time.
@inline function limited_incoming_tracer_flux(i, j, k, grid, advection, ρ, U, c, r)
    incoming = limited_incoming_flux_x(i, j, k, grid, advection, ρ, U.u, c, r) +
               limited_incoming_flux_y(i, j, k, grid, advection, ρ, U.v, c, r) +
               limited_incoming_flux_z(i, j, k, grid, advection, ρ, U.w, c, r)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * incoming
end

@inline credited_incoming_tracer_flux(::Val{false}, i, j, k, grid, advection, ρ, U, c, r₀) = zero(grid)
@inline credited_incoming_tracer_flux(::Val{true},  i, j, k, grid, advection, ρ, U, c, r₀) =
    limited_incoming_tracer_flux(i, j, k, grid, advection, ρ, U, c, r₀)

# r = min(1, available / removed), where `available` is the content at the start of the update plus, when
# credited, the incoming fluxes at the previous pass's fractions `r₀`. Those never exceed the final fractions,
# so the credited incoming fluxes never exceed the actual ones and every pass keeps the tracer non-negative.
@kernel function _compute_flux_limiter!(r, r₀, grid, advection, ρ, U, c, c⁰, Δt, credit_incoming)
    i, j, k = @index(Global, NTuple)
    removed = Δt * outgoing_tracer_flux(i, j, k, grid, advection, ρ, U, c)
    content = @inbounds max(0, c⁰[i, j, k])
    incoming = Δt * credited_incoming_tracer_flux(credit_incoming, i, j, k, grid, advection, ρ, U, c, r₀)
    available = content + ifelse(isfinite(incoming), incoming, zero(incoming))
    @inbounds r[i, j, k] = ifelse(isfinite(removed) & (removed > available), available / removed, one(available))
end

"""
$(TYPEDSIGNATURES)

Refresh the fractions of a flux-limited `scheme` (see [`FluxLimiter`](@ref)) for the coming update
`c⁰ - Δt * div_Uc(c)` of a tracer: `c` is the field the fluxes are reconstructed from, `c⁰` the state
the update adds the tendency to (`c` itself for a forward Euler update, the step-start tracer for a stage
of a `SplitRungeKuttaTimeStepper`), `U` the velocities the fluxes transport with, and `ρ` the density
weighting the fluxes, which `div_Uc` takes as `1`. `c⁰` is measured in the units of `ρ * c`.

A no-op for every other scheme.
"""
update_flux_limiter!(scheme, grid, c, c⁰, U, Δt, ρ=1) = nothing

function update_flux_limiter!(scheme::FluxLimitedWENO, grid, c, c⁰, U, Δt, ρ=1)
    r = scheme.bounds.flux_limiter.fraction
    r₀ = scheme.bounds.flux_limiter.provisional_fraction
    arch = architecture(grid)

    # The first pass limits by the content alone; each later pass credits the incoming fluxes at the
    # previous pass's fractions, which grow from pass to pass. The last pass writes `r`.
    launch!(arch, grid, :xyz, _compute_flux_limiter!, r₀, r₀, grid, scheme, ρ, U, c, c⁰, Δt, Val(false))
    fill_halo_regions!(r₀)

    for (rⁿ, rᵖ) in ((r, r₀), (r₀, r), (r, r₀))
        launch!(arch, grid, :xyz, _compute_flux_limiter!, rⁿ, rᵖ, grid, scheme, ρ, U, c, c⁰, Δt, Val(true))
        fill_halo_regions!(rⁿ)
    end

    return nothing
end

#####
##### Model interface
#####

"""
$(TYPEDSIGNATURES)

The velocities that advect the tracer `name` of `model`: its `velocities` unless the model extends this,
for instance with drift velocities or advective forcing.
"""
tracer_transport_velocities(model, ::Val{name}) where name = model.velocities

# The state the coming update adds the tendency to: between the stages of a `SplitRungeKuttaTimeStepper`
# the step-start fields, which the stepper caches before its first stage; otherwise the current tracer.
# (`clock.stage` is the stage that just finished.)
stage_start_tracer(timestepper, clock, c, name) = c

stage_start_tracer(timestepper::SplitRungeKuttaTimeStepper, clock, c, name) =
    clock.stage < timestepper.Nstages ? timestepper.Ψ⁻[name] : c

update_flux_limiter!(scheme, model, val_name) = nothing

# `adaptive_advection_timestep` is the Δt of the coming stage.
function update_flux_limiter!(scheme::FluxLimitedWENO, model, val_name::Val{name}) where name
    c = model.tracers[name]
    c⁰ = stage_start_tracer(model.timestepper, model.clock, c, name)
    U = tracer_transport_velocities(model, val_name)
    Δt = adaptive_advection_timestep(model.timestepper, model.clock)
    return update_flux_limiter!(scheme, model.grid, c, c⁰, U, Δt)
end
