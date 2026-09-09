using Oceananigans.TurbulenceClosures: implicit_step!

import Oceananigans.TimeSteppers: ssp_substep!

using Oceananigans.TimeSteppers: accumulate_ssp_slow_forcing!, install_ssp_slow_forcing!,
                                 _ssp_euler_substep_field!, _ssp_blend_field!, SSPRungeKuttaTimeStepper,
                                 SSPBarotropicForcing

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

The barotropic pair is blended by *increment* rather than by state: every sub-cycle restarts from
`(ηⁿ, Uⁿ, Vⁿ)` and spans the full `Δt`, and the blend applies the increment it produced to the previous
stage, `Ψᵐ = a Ψⁿ + b (Ψᵐ⁻¹ + Ψ⋆ - Ψⁿ)`. Because the Shu-Osher weights sum to one, the free barotropic
propagator is applied exactly once per step, so the over-integration that blending the barotropic *state*
would incur -- the Shu-Osher weights are derived for forward-Euler stages, not for near-exact advances --
does not arise, while the barotropic stage values are the ones the Shu-Osher recursion requires. The last
stage is driven by the stage-weighted slow forcing, so its sub-cycle is simultaneously the predictor that
supplies that stage's transport and the corrector that sets the barotropic pair at `n+1`.
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
    cache_previous_stage_barotropic_state!(free_surface, timestepper)
    step_free_surface!(free_surface, model, timestepper, Δt)

    blend_barotropic_state!(free_surface, timestepper, a, b)

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
##### Blending the barotropic state into the Shu-Osher combination
#####
##### Every sub-cycle restarts from (ηⁿ, Uⁿ, Vⁿ) and spans the full Δt, so its output is Ψ⋆ = Ψⁿ + ΔΨ and
##### the blend applies ΔΨ to the previous stage. Unrolled, Ψⁿ⁺¹ = Ψⁿ + Σₘ βₘ ΔΨᵐ with Σₘ βₘ = 1: the free
##### barotropic propagator acts once per step, and only the responses to the stage forcings are combined.
##### Blending the barotropic *state* instead would compose three near-exact advances and over-integrate.

@inline cache_previous_stage_barotropic_state!(free_surface, timestepper::SSPRungeKuttaTimeStepper) =
    cache_previous_stage_barotropic_state!(free_surface, timestepper.G★)

@inline cache_previous_stage_barotropic_state!(free_surface, ::Nothing) = nothing

@inline function cache_previous_stage_barotropic_state!(free_surface, G★)
    parent(G★.η) .= parent(free_surface.displacement)
    return nothing
end

@inline function cache_previous_stage_barotropic_state!(free_surface, G★::SSPBarotropicForcing)
    U, V = free_surface.barotropic_velocities
    parent(G★.η)  .= parent(free_surface.displacement)
    parent(G★.Uᵐ) .= parent(U)
    parent(G★.Vᵐ) .= parent(V)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Blend the barotropic state into the Shu-Osher combination, `Ψ ← a Ψⁿ + b (Ψᵐ⁻¹ + Ψ⋆ - Ψⁿ)`, where `Ψ⋆` is
what the sub-cycle just produced from `Ψⁿ`.

The free surface carries the identity: the thickness follows `η`, so its increment
`b (η⋆ - ηⁿ) = -b Δt ∇·Ũ` must carry the same factor `b` as the tracer update, which is what keeps `σ`
consistent with the transport that advected the tracers. The barotropic velocity carries the order: with
the same blend, `Uᵐ` is the Shu-Osher stage value the quadrature `β` expects, so a slow forcing that
depends on the velocity -- Coriolis, drag, momentum advection of the barotropic flow -- is integrated at
the order of the composition rather than at first order.
"""
@inline blend_barotropic_state!(free_surface, timestepper::SSPRungeKuttaTimeStepper, a, b) =
    blend_barotropic_state!(free_surface, timestepper.G★, timestepper.Ψ⁻, a, b)

@inline blend_barotropic_state!(free_surface, ::Nothing, Ψ⁻, a, b) = nothing

@inline function blend_barotropic_state!(free_surface, G★, Ψ⁻, a, b)
    blend_increment!(free_surface.displacement, Ψ⁻.η, G★.η, a, b)
    return nothing
end

@inline function blend_barotropic_state!(free_surface, G★::SSPBarotropicForcing, Ψ⁻, a, b)
    U, V = free_surface.barotropic_velocities
    blend_increment!(free_surface.displacement, Ψ⁻.η, G★.η,  a, b)
    blend_increment!(U,                         Ψ⁻.U, G★.Uᵐ, a, b)
    blend_increment!(V,                         Ψ⁻.V, G★.Vᵐ, a, b)
    return nothing
end

# ψ holds the sub-cycle output Ψ⋆; ψⁿ the state at n; ψᵐ⁻¹ the previous stage.
@inline function blend_increment!(ψ, ψⁿ, ψᵐ⁻¹, a, b)
    parent(ψ) .= a .* parent(ψⁿ) .+ b .* (parent(ψᵐ⁻¹) .+ parent(ψ) .- parent(ψⁿ))
    return nothing
end
