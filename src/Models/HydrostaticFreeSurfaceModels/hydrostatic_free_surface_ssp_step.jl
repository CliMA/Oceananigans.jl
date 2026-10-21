using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.TimeSteppers: SSPRungeKutta3TimeStepper, ssp_quadrature_weights, _ssp_euler_substep_field!, _ssp_blend_field!

import Oceananigans.TimeSteppers: ssp_substep!

step_free_surface!(::ImplicitFreeSurface, model, ::SSPRungeKutta3TimeStepper, Δt) =
    throw(ArgumentError("SSPRungeKutta3TimeStepper does not support ImplicitFreeSurface"))

"""
$(TYPEDSIGNATURES)

Perform one strong-stability-preserving Runge-Kutta stage for `HydrostaticFreeSurfaceModel`.

Velocities and thickness-weighted tracers take an implicit-explicit forward-Euler step of `Δt` from the previous
stage and are blended with the state at `tⁿ` by the Shu-Osher pair `(a, b)`. The barotropic solve restarts from
`(ηⁿ, Uⁿ)` and spans `Δt`; its increment is blended, `Ψᵐ = a Ψⁿ + b (Ψᵐ⁻¹ + Ψ⋆ - Ψⁿ)`. The last solve is driven by
the stage-weighted slow forcing `Σₘ βₘ Ĝᵐ` [Lan et al. (2022)](@cite Lan2022).
"""
function ssp_substep!(model::HydrostaticFreeSurfaceModel, Δt, a, b, callbacks)
    grid = model.grid
    free_surface = model.free_surface
    timestepper = model.timestepper
    stage = model.clock.stage

    @apply_regionally begin
        update_transport_velocities!(model.transport_velocities, model.velocities, free_surface)
        compute_momentum_flux_bcs!(model)
        ssp_euler_substep_velocities!(model.velocities, model, Δt)
    end

    compute_free_surface_tendency!(grid, model, free_surface, Δt)

    β = ssp_quadrature_weights(timestepper.coefficients)[stage]
    map((Ĝ, Gⁿ) -> weight_slow_forcing!(parent(Ĝ), parent(Gⁿ), β, stage, timestepper.Nstages), timestepper.Ĝ, timestepper.Gⁿ[keys(timestepper.Ĝ)])

    ψ = free_surface_fields(free_surface)
    map((ψᵐ⁻¹, ψ) -> parent(ψᵐ⁻¹) .= parent(ψ), timestepper.Ψᵐ⁻¹, ψ)
    step_free_surface!(free_surface, model, timestepper, Δt)
    map((ψ, ψⁿ, ψᵐ⁻¹) -> blend_increment!(parent(ψ), parent(ψⁿ), parent(ψᵐ⁻¹), a, b), ψ, timestepper.Ψ⁻[keys(ψ)], timestepper.Ψᵐ⁻¹)

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
        correct_barotropic_mode!(model, Δt)
        ssp_substep_tracers!(model.tracers, model, Δt, a, b)
    end

    return nothing
end

#####
##### Barotropic mode
#####

# Ĝ accumulates βₘ Ĝᵐ over the stages, and the last stage drives its solve with the full sum.
@inline function weight_slow_forcing!(Ĝ, Gⁿ, β, stage, Nstages)
    if stage == 1
        Ĝ .= β .* Gⁿ
    elseif stage < Nstages
        Ĝ .+= β .* Gⁿ
    else
        Gⁿ .= Ĝ .+ β .* Gⁿ
    end
    return nothing
end

# ψ holds the output of the solve that restarted from ψⁿ.
@inline blend_increment!(ψ, ψⁿ, ψᵐ⁻¹, a, b) = ψ .= a .* ψⁿ .+ b .* (ψᵐ⁻¹ .+ ψ .- ψⁿ)

#####
##### Velocities
#####

function ssp_euler_substep_velocities!(velocities, model, Δt)
    ssp_euler_substep_velocity!(velocities, model, Δt, Val(:u))
    ssp_euler_substep_velocity!(velocities, model, Δt, Val(:v))
    implicit_substep_velocity!(model, Δt, Val(:u))
    implicit_substep_velocity!(model, Δt, Val(:v))
    return nothing
end

@inline function ssp_euler_substep_velocity!(velocities, model, Δt, ::Val{name}) where name
    grid = model.grid
    FT = eltype(grid)
    Gⁿ = model.timestepper.Gⁿ[name]
    launch!(architecture(grid), grid, :xyz, _ssp_euler_substep_field!, velocities[name], convert(FT, Δt), Gⁿ; exclude_periphery=true)
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
    foreach_name(tracers) do val_tracer_index, val_tracer_name
        ssp_substep_tracer!(model, Δt, a, b, val_tracer_index, val_tracer_name)
    end
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

# (σc)ᵐ⁻¹ + Δt Gᵐ on the stage thickness σᵐ, where σ⁻ holds σᵐ⁻¹ after the grid update
@kernel function _ssp_euler_substep_tracer_field!(c, grid, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    σᶜᶜ⁻ = σ⁻(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (σᶜᶜ⁻ * c[i, j, k] + Δt * Gⁿ[i, j, k]) / σᶜᶜⁿ
end

# (σc)ᵐ = a (σc)ⁿ + b (σĉ), with (σc)ⁿ cached in σc⁻
@kernel function _ssp_blend_tracer_field!(c, grid, σc⁻, a, b)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = a * σc⁻[i, j, k] / σᶜᶜⁿ + b * c[i, j, k]
end
