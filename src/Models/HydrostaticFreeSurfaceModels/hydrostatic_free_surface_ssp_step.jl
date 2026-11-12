using Oceananigans.TimeSteppers: _ssp_euler_substep_field!, _ssp_blend_field!, _ssp_blend_tracer_field!,
                                 SSPPreviousStageState, implicit_step!, ssp_stage_fractions, ssp_substep!

# `ImplicitFreeSurface` solves on a predictor velocity, so the Shu-Osher blend would have to be deferred
# until after a pressure correction that has already overwritten Ψᵐ⁻¹.
validate_timestepper_free_surface(timestepper, free_surface) = nothing

validate_timestepper_free_surface(::SSPRungeKuttaTimeStepper, ::ImplicitFreeSurface) =
    throw(ArgumentError("SSPRungeKuttaTimeStepper does not support ImplicitFreeSurface"))

function validate_timestepper_free_surface(timestepper::MultiStageTimeStepper, free_surface::SplitExplicitFreeSurface)
    slow_forcing = free_surface.slow_forcing
    slow_forcing isa SplitExplicitFreeSurfaces.ProgressiveSlowForcing || return nothing

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
does not arise, while the barotropic stage values are the ones the Shu-Osher recursion requires. Every
sub-cycle is driven by the slow forcing of the preceding stage, and the increment blend gives that forcing
the Shu-Osher weights `(1/6, 1/6, 2/3)`, the same weights the blend gives the baroclinic tendencies.
"""
TimeSteppers.ssp_substep!(model::HydrostaticFreeSurfaceModel, Δt, a, b, callbacks) =
    ssp_substep!(model, model.free_surface, model.grid, Δt, a, b, callbacks)

@inline function TimeSteppers.ssp_substep!(model, free_surface, grid, Δt, a, b, callbacks)
    timestepper = model.timestepper

    # As in split Runge-Kutta, exterior states step from their value at tⁿ to the time the stage reaches
    stage_Δt = convert(eltype(grid), ssp_stage_fractions(timestepper.coefficients)[model.clock.stage] * Δt)

    @apply_regionally begin
        update_transport_velocities!(model.transport_velocities, model.velocities, free_surface)
        compute_momentum_flux_bcs!(model)
        ssp_euler_substep_velocities!(model.velocities, model, Δt)
    end

    compute_free_surface_tendency!(grid, model, free_surface, Δt)

    foreach((ψᵐ⁻¹, ψ) -> ψᵐ⁻¹ .= ψ, previous_stage_fields(timestepper.G★), free_surface_fields(free_surface))
    step_free_surface!(free_surface, model, timestepper, Δt)

    @apply_regionally begin
        ssp_blend_velocities!(model.velocities, model, a, b, stage_Δt)
        mask_immersed_horizontal_velocities!(model.velocities)
        compute_transport_velocities!(model, free_surface)
    end

    u, v, _ = model.velocities
    fill_halo_regions!((u, v), model.clock, fields(model); async=true)

    @apply_regionally compute_tracer_tendencies!(model)
    execute_tracer_tendency_callbacks!(model, callbacks)

    @apply_regionally begin
        rk_substep_grid!(grid, model, model.vertical_coordinate, Δt)

        # Reconciled at every stage: deferring it to the final stage breaks tracer constancy.
        correct_barotropic_mode!(model, Δt)
        ssp_substep_tracers!(model.tracers, model, Δt, a, b)
        step_boundary_conditions!(model.tracers, model.clock, fields(model), stage_Δt)
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

function ssp_blend_velocities!(velocities, model, a, b, stage_Δt)
    grid = model.grid
    FT = eltype(grid)
    Ψ⁻ = model.timestepper.Ψ⁻
    a, b = convert(FT, a), convert(FT, b)

    launch!(architecture(grid), grid, :xyz, _ssp_blend_field!, velocities.u, Ψ⁻.u, a, b; exclude_periphery=true)
    launch!(architecture(grid), grid, :xyz, _ssp_blend_field!, velocities.v, Ψ⁻.v, a, b; exclude_periphery=true)
    step_boundary_conditions!(velocities, model.clock, fields(model), stage_Δt)

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

    # The adaptive implicit advection must see the same total velocity as the explicit flux, drift included
    @inbounds c_advection = model.advection[tracer_name]
    c_velocities = tracer_advecting_velocities(model.transport_velocities, model.biogeochemistry, closure,
                                               model.closure_fields, model.forcing[tracer_name], Val(tracer_name))

    implicit_step!(c,
                   model.timestepper.implicit_solver,
                   closure,
                   model.closure_fields,
                   Val(tracer_index),
                   model.clock,
                   fields(model),
                   Δt,
                   c_advection,
                   c_velocities)

    launch!(architecture(grid), grid, :xyz, _ssp_blend_tracer_field!, c, grid, Ψ⁻, convert(FT, a), convert(FT, b))

    return nothing
end

# The Euler stage advances the thickness-weighted tracer, (σĉ) = (σc)ᵐ⁻¹ + Δt Gᵐ, so the previous-stage
# term carries σᵐ⁻¹ rather than σᵐ.
@kernel function _ssp_euler_substep_tracer_field!(c, grid, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)
    σ⁻¹ᶜᶜⁿ = σ⁻¹ⁿ(i, j, k, grid, Center(), Center(), Center())
    σᶜᶜ⁻ = σ⁻(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (σᶜᶜ⁻ * c[i, j, k] + Δt * Gⁿ[i, j, k]) * σ⁻¹ᶜᶜⁿ
end

#####
##### Blending the barotropic state into the Shu-Osher combination
#####
##### Every sub-cycle restarts from (ηⁿ, Uⁿ, Vⁿ) and spans the full Δt, so its output is Ψ⋆ = Ψⁿ + ΔΨ and
##### the blend applies ΔΨ to the previous stage. Unrolled, Ψⁿ⁺¹ = Ψⁿ + Σₘ βₘ ΔΨᵐ with Σₘ βₘ = 1: the free
##### barotropic propagator acts once per step, and only the responses to the stage forcings are combined.
##### Blending the barotropic *state* instead would compose three near-exact advances and over-integrate.

@inline previous_stage_fields(::Nothing) = NamedTuple()
@inline previous_stage_fields(G★::SSPPreviousStageState) = (η = G★.η, U = G★.Uᵐ, V = G★.Vᵐ)

blend_free_surface!(free_surface, timestepper, stage) = nothing

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
function blend_free_surface!(free_surface, timestepper::SSPRungeKuttaTimeStepper, stage)
    grid = free_surface.displacement.grid
    FT = eltype(grid)
    a, b = timestepper.coefficients[stage]
    ψ = free_surface_fields(free_surface)
    launch!(architecture(grid), grid, :xy, _blend_free_surface!, ψ, timestepper.Ψ⁻[keys(ψ)],
            previous_stage_fields(timestepper.G★), convert(FT, a), convert(FT, b), grid.Nz + 1)
    return nothing
end

# ψ holds the output of the solve that restarted from ψⁿ.
@kernel function _blend_free_surface!(ψ, ψⁿ, ψᵐ⁻¹, a, b, k_top)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ψ.η[i, j, k_top] = a * ψⁿ.η[i, j, k_top] + b * (ψᵐ⁻¹.η[i, j, k_top] + ψ.η[i, j, k_top] - ψⁿ.η[i, j, k_top])
        ψ.U[i, j, 1]     = a * ψⁿ.U[i, j, 1]     + b * (ψᵐ⁻¹.U[i, j, 1]     + ψ.U[i, j, 1]     - ψⁿ.U[i, j, 1])
        ψ.V[i, j, 1]     = a * ψⁿ.V[i, j, 1]     + b * (ψᵐ⁻¹.V[i, j, 1]     + ψ.V[i, j, 1]     - ψⁿ.V[i, j, 1])
    end
end
