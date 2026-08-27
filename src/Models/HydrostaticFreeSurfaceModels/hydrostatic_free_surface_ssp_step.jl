using Oceananigans.TurbulenceClosures: implicit_step!

import Oceananigans.TimeSteppers: ssp_substep!

using Oceananigans.TimeSteppers: accumulate_ssp_slow_forcing!, install_ssp_slow_forcing!,
                                 _ssp_euler_substep_field!, _ssp_blend_field!, SSPRungeKuttaTimeStepper

# `ImplicitFreeSurface` solves on a predictor velocity, so the Shu-Osher blend would have to be deferred
# until after a pressure correction that has already overwritten Ψᵐ⁻¹.
validate_timestepper_free_surface(timestepper, free_surface) = nothing

validate_timestepper_free_surface(::SSPRungeKuttaTimeStepper, ::ImplicitFreeSurface) =
    throw(ArgumentError("SSPRungeKuttaTimeStepper does not support ImplicitFreeSurface"))

function validate_timestepper_free_surface(timestepper::MultiStageTimeStepper, free_surface::SplitExplicitFreeSurface)
    slow_forcing = free_surface.slow_forcing
    slow_forcing isa SplitExplicitFreeSurfaces.ReconstructedSlowForcing || return nothing

    τ = SplitExplicitFreeSurfaces.stage_sample_times(timestepper)
    nodes = slow_forcing.nodes
    length(nodes) == length(τ) && all(isapprox.(nodes, τ; rtol=1e-12)) && return nothing

    throw(ArgumentError("$(nameof(typeof(slow_forcing))) samples the slow forcing at $nodes, but the timestepper samples at $τ"))
end

"""
$(TYPEDSIGNATURES)

Perform one strong-stability-preserving Runge-Kutta stage for `HydrostaticFreeSurfaceModel`.

The baroclinic velocities, tracers and grid are advanced by a full `Δt` and blended with the cached state
by the Shu-Osher pair `(a, b)`. Each stage is an implicit-explicit forward-Euler step: the explicit tendency
and the vertical solve are applied to `Ψᵐ⁻¹`, and the blend follows.

The barotropic mode is *not* blended: the sub-cycles run on the first `Nstages - 1` stages are predictors,
supplying the transport that advects tracers at their stage, and the barotropic velocity at `n+1` is set by a
single corrector sub-cycle after the last stage, forced by the stage-weighted slow forcing. The Shu-Osher
weights are derived for forward-Euler stages, and carrying a sub-cycled barotropic solve through them
over-integrates it and costs an order.
"""
ssp_substep!(model::HydrostaticFreeSurfaceModel, Δt, a, b, callbacks) =
    ssp_substep!(model, model.free_surface, model.grid, Δt, a, b, callbacks)

@inline function ssp_substep!(model, free_surface, grid, Δt, a, b, callbacks)
    timestepper = model.timestepper
    final_stage = model.clock.stage == timestepper.Nstages

    @apply_regionally begin
        update_transport_velocities!(model.transport_velocities, model.velocities, free_surface)
        compute_momentum_flux_bcs!(model)
        ssp_euler_substep_velocities!(model.velocities, model, Δt)
    end

    compute_free_surface_tendency!(grid, model, free_surface, Δt)
    accumulate_ssp_slow_forcing!(timestepper, model.clock.stage)

    # Must run before `compute_transport_velocities!`: tracer constancy needs the transport that advects
    # them to be the same flux that advanced the free surface.
    final_stage && install_ssp_slow_forcing!(timestepper)
    cache_previous_stage_free_surface!(free_surface, timestepper)
    step_free_surface!(free_surface, model, timestepper, Δt)

    blend_free_surface!(free_surface, timestepper, a, b)

    @apply_regionally begin
        ssp_blend_velocities!(model.velocities, model, a, b)
        mask_immersed_horizontal_velocities!(model.velocities)
        compute_transport_velocities!(model, free_surface)
    end

    u, v, _ = model.velocities
    fill_halo_regions!((u, v), model.clock, fields(model); async=true)

    @apply_regionally begin
        compute_tracer_tendencies!(model)
        rk_substep_grid!(grid, model, model.vertical_coordinate, Δt)

        # Reconciled at every stage: deferring it to the final stage breaks tracer constancy.
        correct_barotropic_mode!(model, Δt)
        ssp_substep_tracers!(model.tracers, model, Δt, a, b)
    end

    return nothing
end

#####
##### Velocities
#####

function ssp_euler_substep_velocities!(velocities, model, Δt)
    ssp_euler_substep_velocity!(velocities, model, Δt, Val(:u))
    ssp_euler_substep_velocity!(velocities, model, Δt, Val(:v))
    return nothing
end

@inline function ssp_euler_substep_velocity!(velocities, model, Δt, ::Val{name}) where name
    grid = model.grid
    FT = eltype(grid)

    Gⁿ = model.timestepper.Gⁿ[name]
    velocity_field = velocities[name]

    launch!(architecture(grid), grid, :xyz, _ssp_euler_substep_field!, velocity_field, convert(FT, Δt), Gⁿ; exclude_periphery=true)

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

function ssp_blend_velocities!(velocities, model, a, b)
    grid = model.grid
    FT = eltype(grid)
    Ψ⁻ = model.timestepper.Ψ⁻
    a, b = convert(FT, a), convert(FT, b)

    launch!(architecture(grid), grid, :xyz, _ssp_blend_field!, velocities.u, Ψ⁻.u, a, b; exclude_periphery=true)
    launch!(architecture(grid), grid, :xyz, _ssp_blend_field!, velocities.v, Ψ⁻.v, a, b; exclude_periphery=true)

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

    launch!(architecture(grid), grid, :xyz, _ssp_euler_substep_tracer_field!, c, grid, convert(FT, Δt), Gⁿ)

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

    launch!(architecture(grid), grid, :xyz, _ssp_blend_tracer_field!, c, grid, Ψ⁻, convert(FT, a), convert(FT, b))

    return nothing
end

# The Euler stage advances the thickness-weighted tracer, (σĉ) = (σc)ᵐ⁻¹ + Δt Gᵐ, so the previous-stage
# term carries σᵐ⁻¹ rather than σᵐ.
@kernel function _ssp_euler_substep_tracer_field!(c, grid, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    σᶜᶜ⁻ = σ⁻(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (σᶜᶜ⁻ * c[i, j, k] + Δt * Gⁿ[i, j, k]) / σᶜᶜⁿ
end

# The blend applies to the thickness-weighted tracer, (σc)ᵐ = a (σc)ⁿ + b (σĉ), on the stage thickness σᵐ.
@kernel function _ssp_blend_tracer_field!(c, grid, σc⁻, a, b)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = a * σc⁻[i, j, k] / σᶜᶜⁿ + b * c[i, j, k]
end

#####
##### Blending the free surface into the Shu-Osher combination
#####

@inline function cache_previous_stage_free_surface!(free_surface, timestepper)
    isnothing(timestepper.G★) && return nothing
    parent(timestepper.G★.η) .= parent(free_surface.displacement)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Blend the free-surface displacement into the Shu-Osher combination, `η ← a ηⁿ + b η̂`, where `η̂` is what
the barotropic sub-cycle just produced. The thickness follows η, so the increment `b (η̂ - ηⁿ) = -b Δt ∇·Ũ`
carries the same factor `b` as the tracer update, which is what keeps `σ` consistent with the transport.
"""
@inline function blend_free_surface!(free_surface, timestepper, a, b)
    isnothing(timestepper.G★) && return nothing
    η    = free_surface.displacement
    ηⁿ   = timestepper.Ψ⁻.η
    ηᵐ⁻¹ = timestepper.G★.η

    # The sub-cycle restarts from ηⁿ, so its output is η̂ = ηⁿ + Δη and the blend applies Δη to ηᵐ⁻¹.
    parent(η) .= a .* parent(ηⁿ) .+ b .* (parent(ηᵐ⁻¹) .+ parent(η) .- parent(ηⁿ))
    return nothing
end
