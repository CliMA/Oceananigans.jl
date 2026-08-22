using Oceananigans.TurbulenceClosures: implicit_step!

import Oceananigans.TimeSteppers: ssp_substep!

using Oceananigans.TimeSteppers: accumulate_ssp_slow_forcing!, install_ssp_slow_forcing!,
                                 _ssp_substep_field!, SSPRungeKuttaTimeStepper

#####
##### Free-surface formulations the strong-stability-preserving stages support
#####
##### `ImplicitFreeSurface` is the one exclusion. Its solve takes the divergence of a *predictor* velocity,
##### so the Shu-Osher blend would have to be deferred until after the pressure correction, on a state that
##### has already overwritten `Ψᵐ⁻¹`. The low-storage path sidesteps this by restarting every stage from
##### `Ψⁿ`, which the Shu-Osher form does not do.
#####

validate_timestepper_free_surface(timestepper, free_surface) = nothing

function validate_timestepper_free_surface(::SSPRungeKuttaTimeStepper, ::ImplicitFreeSurface)
    msg = string("SSPRungeKuttaTimeStepper does not support ImplicitFreeSurface.", '\n',
                 "Use SplitExplicitFreeSurface or ExplicitFreeSurface, or time-step with ",
                 "timestepper = :SplitRungeKutta3.")
    throw(ArgumentError(msg))
end

function validate_timestepper_free_surface(timestepper::MultiStageTimeStepper, free_surface::SplitExplicitFreeSurface)
    slow_forcing = free_surface.slow_forcing
    slow_forcing isa SplitExplicitFreeSurfaces.ReconstructedSlowForcing || return nothing

    nodes = SplitExplicitFreeSurfaces.stage_sample_times(timestepper)
    name  = nameof(typeof(slow_forcing))

    if length(slow_forcing.nodes) != timestepper.Nstages
        msg = string(name, " was built for ", length(slow_forcing.nodes), " stages, but the timestepper has ",
                     timestepper.Nstages, ". Build it from the same timestepper.")
        throw(ArgumentError(msg))
    end

    if !all(isapprox.(slow_forcing.nodes, nodes; rtol=1e-12))
        msg = string(name, " was built on the stage sample times ", slow_forcing.nodes,
                     ", but the timestepper samples at ", nodes, ". Build it from the same timestepper.")
        throw(ArgumentError(msg))
    end

    return nothing
end

"""
$(TYPEDSIGNATURES)

Perform one strong-stability-preserving Runge-Kutta stage for `HydrostaticFreeSurfaceModel`.

The baroclinic velocities, tracers and grid are advanced by a full `Δt` and blended with the cached state
by the Shu-Osher pair `(a, b)`. The barotropic mode is *not* blended: the sub-cycles run on the first
`Nstages - 1` stages are predictors, supplying the transport that advects tracers at their stage, and the
barotropic velocity at `n+1` is set by a single corrector sub-cycle after the last stage, forced by the
stage-weighted slow forcing.

That asymmetry is not an optimisation, it is what makes the scheme second order. The Shu-Osher weights
are derived for forward-Euler stages; carrying a sub-cycled barotropic solve through them over-integrates
it and costs an order.
"""
ssp_substep!(model::HydrostaticFreeSurfaceModel, Δt, a, b, callbacks) =
    ssp_substep!(model, model.free_surface, model.grid, Δt, a, b, callbacks)

@inline function ssp_substep!(model, free_surface, grid, Δt, a, b, callbacks)
    timestepper = model.timestepper
    final_stage = model.clock.stage == timestepper.Nstages

    @apply_regionally compute_momentum_flux_bcs!(model)

    # This stage's depth-integrated slow forcing, accumulated into the corrector's combination.
    compute_free_surface_tendency!(grid, model, free_surface)
    accumulate_ssp_slow_forcing!(timestepper, model.clock.stage)

    # On the final stage the accumulation is complete, so the barotropic solve becomes the corrector: it
    # is forced by the stage-weighted combination and sets the barotropic velocity and free surface at
    # n+1. On the earlier stages it is a predictor, discarded except for the transport it supplies.
    #
    # It must run BEFORE `compute_transport_velocities!`. Tracer constancy requires the transport that
    # advects them to be the same flux that advanced the free surface; stepping the free surface
    # afterwards would advect with the previous stage's transport and break it (measured: 3e-5 against
    # 2e-15 on a z-star seamount grid).
    final_stage && install_ssp_slow_forcing!(timestepper)
    cache_previous_stage_free_surface!(free_surface, timestepper)
    step_free_surface!(free_surface, model, timestepper, Δt)

    # The free surface is part of the blended state, exactly as the modal analysis has it: the stage
    # advance is `Ψᵐ = a Ψⁿ + b (Ψᵐ⁻¹ + Δt Gᵐ)` and `Ψ` includes the layer thicknesses.
    blend_free_surface!(free_surface, timestepper, a, b)

    @apply_regionally begin
        compute_transport_velocities!(model, free_surface)
        ssp_substep_velocities!(model.velocities, model, Δt, a, b)
        mask_immersed_horizontal_velocities!(model.velocities)
    end

    u, v, _ = model.velocities
    fill_halo_regions!((u, v), model.clock, fields(model); async=true)

    @apply_regionally begin
        compute_tracer_tendencies!(model)
        rk_substep_grid!(grid, model, model.vertical_coordinate, Δt)

        # Reconcile the depth average of the three-dimensional velocity with the barotropic solve at
        # EVERY stage, exactly as the low-storage path does. Deferring it to the final stage leaves the
        # transport that advects tracers inconsistent with the free surface at stages 1 and 2 and breaks
        # constancy (measured: 3e-5 against 2e-15 on a z-star seamount grid).
        correct_barotropic_mode!(model, Δt)
        ssp_substep_tracers!(model.tracers, model, Δt, a, b)
    end

    return nothing
end

#####
##### Velocities
#####

function ssp_substep_velocities!(velocities, model, Δt, a, b)
    ssp_substep_velocity!(velocities, model, Δt, a, b, Val(:u))
    ssp_substep_velocity!(velocities, model, Δt, a, b, Val(:v))
    return nothing
end

@inline function ssp_substep_velocity!(velocities, model, Δt, a, b, ::Val{name}) where name
    grid = model.grid
    FT = eltype(grid)

    Gⁿ = model.timestepper.Gⁿ[name]
    Ψ⁻ = model.timestepper.Ψ⁻[name]
    velocity_field = velocities[name]

    launch!(architecture(grid), grid, :xyz,
            _ssp_substep_field!, velocity_field, convert(FT, Δt), Gⁿ, Ψ⁻,
            convert(FT, a), convert(FT, b); exclude_periphery=true)

    implicit_step!(velocity_field,
                   model.timestepper.implicit_solver,
                   model.closure,
                   model.closure_fields,
                   nothing,
                   model.clock,
                   fields(model),
                   Δt,
                   model.advection.momentum,
                   model.velocities)
    return nothing
end

#####
##### Tracers
#####

ssp_substep_tracers!(::EmptyNamedTuple, model, Δt, a, b) = nothing

function ssp_substep_tracers!(tracers, model, Δt, a, b)
    ssp_substep_tracers!(model, Δt, a, b, Val(1), Val(propertynames(tracers)))
    return nothing
end

@inline ssp_substep_tracers!(model, Δt, a, b, ::Val, ::Val{()}) = nothing

@inline function ssp_substep_tracers!(model, Δt, a, b, ::Val{tracer_index}, ::Val{names}) where {tracer_index, names}
    ssp_substep_tracer!(model, Δt, a, b, Val(tracer_index), Val(first(names)))
    ssp_substep_tracers!(model, Δt, a, b, Val(tracer_index + 1), Val(Base.tail(names)))
    return nothing
end

@inline function ssp_substep_tracer!(model, Δt, a, b, ::Val{tracer_index}, ::Val{tracer_name}) where {tracer_index, tracer_name}
    closure = model.closure
    (hasclosure(closure, FlavorOfCATKE) && tracer_name == :e) && return nothing

    grid = model.grid
    FT = eltype(grid)

    Gⁿ = model.timestepper.Gⁿ[tracer_name]
    Ψ⁻ = model.timestepper.Ψ⁻[tracer_name]
    c  = model.tracers[tracer_name]

    launch!(architecture(grid), grid, :xyz,
            _ssp_substep_tracer_field!, c, grid, convert(FT, Δt), Gⁿ, Ψ⁻, convert(FT, a), convert(FT, b))

    @inbounds c_advection = model.advection[tracer_name]
    implicit_step!(c,
                   model.timestepper.implicit_solver,
                   closure,
                   model.closure_fields,
                   Val(tracer_index),
                   model.clock,
                   fields(model),
                   Δt,
                   c_advection,
                   model.transport_velocities)
    return nothing
end

# σc is the evolved quantity, so the Shu-Osher blend applies to the *thickness-weighted* tracer:
#
#     (σc)ᵐ = a (σc)ⁿ + b [(σc)ᵐ⁻¹ + Δt Gᵐ] ,   cᵐ = (σc)ᵐ / σᵐ
#
# The previous-stage term must be weighted by σᵐ⁻¹, not by σᵐ. `rk_substep_grid!` copies σⁿ into σ⁻
# before advancing it, so by the time this runs σ⁻ holds σᵐ⁻¹ and σⁿ holds σᵐ.
#
# Setting c ≡ 1 gives σᵐ = a σⁿ + b (σᵐ⁻¹ - Δt ∇·Ũᵐ), which is what `blend_free_surface!` produces --
# that pair is what makes constancy exact, and either one alone does not.
@kernel function _ssp_substep_tracer_field!(c, grid, Δt, Gⁿ, σc⁻, a, b)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    σᶜᶜ⁻ = σ⁻(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (a * σc⁻[i, j, k] + b * (σᶜᶜ⁻ * c[i, j, k] + Δt * Gⁿ[i, j, k])) / σᶜᶜⁿ
end


#####
##### Blending the free surface into the Shu-Osher combination
#####

"""
$(TYPEDSIGNATURES)

Blend the free-surface displacement into the Shu-Osher combination, `η ← a ηⁿ + b η̂`, where `η̂` is what
the barotropic sub-cycle just produced.

The thickness follows η, so this is what keeps `σ` consistent with the transport that advects the
tracers: the free-surface increment is `b (η̂ - ηⁿ) = -b Δt ∇·Ũ`, carrying the same factor `b` that the
tracer update carries.
"""
@inline function cache_previous_stage_free_surface!(free_surface, timestepper)
    isnothing(timestepper.G★) && return nothing
    parent(timestepper.G★.η) .= parent(free_surface.displacement)
    return nothing
end

@inline function blend_free_surface!(free_surface, timestepper, a, b)
    isnothing(timestepper.G★) && return nothing
    η    = free_surface.displacement
    ηⁿ   = timestepper.Ψ⁻.η
    ηᵐ⁻¹ = timestepper.G★.η

    # The sub-cycle restarts from ηⁿ, so its output is η̂ = ηⁿ + Δη. The blend needs that increment
    # applied to the previous stage: ηᵐ = a ηⁿ + b (ηᵐ⁻¹ + Δη), which is what the modal analysis does
    # when it advances the layer thicknesses from `yᵐ⁻¹` rather than from `y⁰`.
    parent(η) .= a .* parent(ηⁿ) .+ b .* (parent(ηᵐ⁻¹) .+ parent(η) .- parent(ηⁿ))
    return nothing
end
