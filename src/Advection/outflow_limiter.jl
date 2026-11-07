using Oceananigans.BoundaryConditions: fill_halo_regions!

#####
##### Outflow limiting for bounds-preserving WENO
#####
#
# The rescaling of Zhang and Shu (2010) keeps a forward Euler update within `bounds` only while the
# Courant number stays below ω̂₁. A larger time step, or a Wicker–Skamarock stage that updates the
# step-start state with fluxes built from a later stage, can remove more from a cell than it holds.
# The outflow limiter scales each cell's outgoing fluxes by the fraction `r = min(1, available / removed)`
# of them the coming update can afford: a conservative, zeroth-order flux-corrected transport limiter
# that keeps the tracer non-negative for any time step, at first-order accuracy where it acts.

"""
    OutflowLimiter(rescaling, outflow_fraction, provisional_fraction)

The `limiter` of a [`BoundsPreservation`](@ref) that also keeps the tracer non-negative by limiting each
cell's outflow: the cell-wise reconstruction `rescaling` factor ``θ``, the fraction ``r ∈ [0, 1]`` of each
cell's outgoing fluxes the coming update applies (`outflow_fraction`), and the fractions of the previous
limiter pass (`provisional_fraction`), which scale the inflow credited to a cell. The fields are `nothing`
until the scheme is materialized on a grid.

Each face flux is scaled by the fraction of the cell it leaves, so the limited fluxes stay conservative.
The fractions are computed by [`update_outflow_limiter!`](@ref) before the fluxes are evaluated.
"""
struct OutflowLimiter{Θ, R}
    rescaling :: Θ             # the cell-wise rescaling factor θ of the reconstructions
    outflow_fraction :: R      # r, the fraction of each cell's outflow the coming update applies
    provisional_fraction :: R  # r from the previous pass, which scales the inflow credited to a cell
end

Adapt.adapt_structure(to, limiter::OutflowLimiter) =
    OutflowLimiter(Adapt.adapt(to, limiter.rescaling),
                   Adapt.adapt(to, limiter.outflow_fraction),
                   Adapt.adapt(to, limiter.provisional_fraction))

Architectures.on_architecture(to, limiter::OutflowLimiter) =
    OutflowLimiter(on_architecture(to, limiter.rescaling),
                   on_architecture(to, limiter.outflow_fraction),
                   on_architecture(to, limiter.provisional_fraction))

const OutflowLimitedBounds = BoundsPreservation{<:Any, <:OutflowLimiter}
const OutflowLimitedWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:OutflowLimitedBounds}

Base.show(io::IO, bounds::OutflowLimitedBounds) =
    print(io, "(", bounds.minimum_value, ", ", bounds.maximum_value, "), outflow_limiter=true")

@inline rescaling_factor(scheme::OutflowLimitedWENO) = scheme.bounds.limiter.rescaling

#####
##### Limited face fluxes and their divergence
#####

# Each face flux is scaled by the outflow fraction of its upwind cell.
@inline function upwind_outflow_fraction_x(i, j, k, advection, u)
    r = advection.bounds.limiter.outflow_fraction
    return @inbounds ifelse(u[i, j, k] ≥ 0, r[i-1, j, k], r[i, j, k])
end

@inline function upwind_outflow_fraction_y(i, j, k, advection, v)
    r = advection.bounds.limiter.outflow_fraction
    return @inbounds ifelse(v[i, j, k] ≥ 0, r[i, j-1, k], r[i, j, k])
end

@inline function upwind_outflow_fraction_z(i, j, k, advection, w)
    r = advection.bounds.limiter.outflow_fraction
    return @inbounds ifelse(w[i, j, k] ≥ 0, r[i, j, k-1], r[i, j, k])
end

# Support for Flat directions, which have no neighbors to index
@inline limited_face_flux_x(i, j, k, grid::XFlatGrid, advection, args...) = zero(grid)
@inline limited_face_flux_y(i, j, k, grid::YFlatGrid, advection, args...) = zero(grid)
@inline limited_face_flux_z(i, j, k, grid::ZFlatGrid, advection, args...) = zero(grid)

@inline limited_face_flux_x(i, j, k, grid, advection, ρ, u, c) =
    upwind_outflow_fraction_x(i, j, k, advection, u) * bounded_face_flux_x(i, j, k, grid, advection, ρ, u, c)

@inline limited_face_flux_y(i, j, k, grid, advection, ρ, v, c) =
    upwind_outflow_fraction_y(i, j, k, advection, v) * bounded_face_flux_y(i, j, k, grid, advection, ρ, v, c)

@inline limited_face_flux_z(i, j, k, grid, advection, ρ, w, c) =
    upwind_outflow_fraction_z(i, j, k, advection, w) * bounded_face_flux_z(i, j, k, grid, advection, ρ, w, c)

@inline limited_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, u, c) =
    limited_face_flux_x(i+1, j, k, grid, advection, ρ, u, c) - limited_face_flux_x(i, j, k, grid, advection, ρ, u, c)

@inline limited_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, v, c) =
    limited_face_flux_y(i, j+1, k, grid, advection, ρ, v, c) - limited_face_flux_y(i, j, k, grid, advection, ρ, v, c)

@inline limited_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, w, c) =
    limited_face_flux_z(i, j, k+1, grid, advection, ρ, w, c) - limited_face_flux_z(i, j, k, grid, advection, ρ, w, c)

@inline div_Uc(i, j, k, grid, advection::OutflowLimitedWENO, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::OutflowLimitedWENO, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection::OutflowLimitedWENO, ::ZeroU, ::ZeroField) = zero(grid)

# With every outflow fraction equal to one this is the bounds-preserving divergence.
@inline function div_Uc(i, j, k, grid, advection::OutflowLimitedWENO, U, c)
    div_x = limited_tracer_flux_divergence_x(i, j, k, grid, advection, 1, U.u, c)
    div_y = limited_tracer_flux_divergence_y(i, j, k, grid, advection, 1, U.v, c)
    div_z = limited_tracer_flux_divergence_z(i, j, k, grid, advection, 1, U.w, c)

    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

#####
##### Outflow fractions
#####

# Tracer the unlimited fluxes remove from cell (i, j, k), per unit volume and time.
@inline function explicit_outflow(i, j, k, grid, advection, ρ, U, c)
    Fx⁻ = bounded_face_flux_x(i,   j, k, grid, advection, ρ, U.u, c)
    Fx⁺ = bounded_face_flux_x(i+1, j, k, grid, advection, ρ, U.u, c)
    Fy⁻ = bounded_face_flux_y(i, j,   k, grid, advection, ρ, U.v, c)
    Fy⁺ = bounded_face_flux_y(i, j+1, k, grid, advection, ρ, U.v, c)
    Fz⁻ = bounded_face_flux_z(i, j, k,   grid, advection, ρ, U.w, c)
    Fz⁺ = bounded_face_flux_z(i, j, k+1, grid, advection, ρ, U.w, c)
    outflow = max(0, Fx⁺) - min(0, Fx⁻) + max(0, Fy⁺) - min(0, Fy⁻) + max(0, Fz⁺) - min(0, Fz⁻)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * outflow
end

# Tracer the fluxes bring into cell (i, j, k) through its two faces in one direction, each scaled
# by the fraction `r` of the cell it leaves. A Flat direction has no neighbors (and no halo to index).
@inline limited_inflow_x(i, j, k, grid::XFlatGrid, advection, args...) = zero(grid)
@inline limited_inflow_y(i, j, k, grid::YFlatGrid, advection, args...) = zero(grid)
@inline limited_inflow_z(i, j, k, grid::ZFlatGrid, advection, args...) = zero(grid)

@inline function limited_inflow_x(i, j, k, grid, advection, ρ, u, c, r)
    F⁻ = bounded_face_flux_x(i,   j, k, grid, advection, ρ, u, c)
    F⁺ = bounded_face_flux_x(i+1, j, k, grid, advection, ρ, u, c)
    return @inbounds max(0, F⁻) * r[i-1, j, k] - min(0, F⁺) * r[i+1, j, k]
end

@inline function limited_inflow_y(i, j, k, grid, advection, ρ, v, c, r)
    F⁻ = bounded_face_flux_y(i, j,   k, grid, advection, ρ, v, c)
    F⁺ = bounded_face_flux_y(i, j+1, k, grid, advection, ρ, v, c)
    return @inbounds max(0, F⁻) * r[i, j-1, k] - min(0, F⁺) * r[i, j+1, k]
end

@inline function limited_inflow_z(i, j, k, grid, advection, ρ, w, c, r)
    F⁻ = bounded_face_flux_z(i, j, k,   grid, advection, ρ, w, c)
    F⁺ = bounded_face_flux_z(i, j, k+1, grid, advection, ρ, w, c)
    return @inbounds max(0, F⁻) * r[i, j, k-1] - min(0, F⁺) * r[i, j, k+1]
end

# Tracer the limited fluxes bring into cell (i, j, k), per unit volume and time.
@inline function limited_inflow(i, j, k, grid, advection, ρ, U, c, r)
    inflow = limited_inflow_x(i, j, k, grid, advection, ρ, U.u, c, r) +
             limited_inflow_y(i, j, k, grid, advection, ρ, U.v, c, r) +
             limited_inflow_z(i, j, k, grid, advection, ρ, U.w, c, r)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * inflow
end

@inline credited_inflow(::Val{false}, i, j, k, grid, advection, ρ, U, c, r₀) = zero(grid)
@inline credited_inflow(::Val{true},  i, j, k, grid, advection, ρ, U, c, r₀) = limited_inflow(i, j, k, grid, advection, ρ, U, c, r₀)

# r = min(1, available / removed), where `available` is the content at the start of the update plus,
# when credited, the inflow at the previous pass's fractions `r₀`. Those never exceed the final fractions,
# so the credited inflow never exceeds the actual one and every pass keeps the tracer non-negative.
@kernel function _compute_outflow_fraction!(r, r₀, grid, advection, ρ, U, c, c⁰, Δt, credit_inflow)
    i, j, k = @index(Global, NTuple)
    removed = Δt * explicit_outflow(i, j, k, grid, advection, ρ, U, c)
    content = @inbounds max(0, c⁰[i, j, k])
    inflow = Δt * credited_inflow(credit_inflow, i, j, k, grid, advection, ρ, U, c, r₀)
    available = content + ifelse(isfinite(inflow), inflow, zero(inflow))
    @inbounds r[i, j, k] = ifelse(isfinite(removed) & (removed > available), available / removed, one(available))
end

"""
$(TYPEDSIGNATURES)

Refresh the outflow fractions of an outflow-limited `scheme` (see [`OutflowLimiter`](@ref)) for the coming
update `c⁰ - Δt * div_Uc(c)` of a tracer: `c` is the field the fluxes are reconstructed from, `c⁰` the state
the update adds the tendency to (`c` itself for a forward Euler update, the step-start tracer for a stage
of a `SplitRungeKuttaTimeStepper`), `U` the velocities the fluxes transport with, and `ρ` the density
weighting the fluxes, which `div_Uc` takes as `1`. `c⁰` is measured in the units of `ρ * c`.

A no-op for every other scheme.
"""
update_outflow_limiter!(scheme, grid, c, c⁰, U, Δt, ρ=1) = nothing

function update_outflow_limiter!(scheme::OutflowLimitedWENO, grid, c, c⁰, U, Δt, ρ=1)
    r = scheme.bounds.limiter.outflow_fraction
    r₀ = scheme.bounds.limiter.provisional_fraction
    arch = architecture(grid)

    # The first pass limits by the content alone; each later pass credits the inflow at the previous
    # pass's fractions, which grow from pass to pass. The last pass writes `r`.
    launch!(arch, grid, :xyz, _compute_outflow_fraction!, r₀, r₀, grid, scheme, ρ, U, c, c⁰, Δt, Val(false))
    fill_halo_regions!(r₀)

    for (rⁿ, rᵖ) in ((r, r₀), (r₀, r), (r, r₀))
        launch!(arch, grid, :xyz, _compute_outflow_fraction!, rⁿ, rᵖ, grid, scheme, ρ, U, c, c⁰, Δt, Val(true))
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

update_outflow_limiter!(scheme, model, val_name) = nothing

# `adaptive_advection_timestep` is the Δt of the coming stage.
function update_outflow_limiter!(scheme::OutflowLimitedWENO, model, val_name::Val{name}) where name
    c = model.tracers[name]
    c⁰ = stage_start_tracer(model.timestepper, model.clock, c, name)
    U = tracer_transport_velocities(model, val_name)
    Δt = adaptive_advection_timestep(model.timestepper, model.clock)
    return update_outflow_limiter!(scheme, model.grid, c, c⁰, U, Δt)
end
