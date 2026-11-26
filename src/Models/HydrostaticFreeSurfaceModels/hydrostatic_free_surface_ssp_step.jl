using Oceananigans.TimeSteppers: _ssp_blend_tracer_field!, implicit_step!, ssp_stage_fractions

step_free_surface!(::ImplicitFreeSurface, model, ::SSPRungeKutta3TimeStepper, Δt) =
    throw(ArgumentError("SSPRungeKutta3TimeStepper does not support ImplicitFreeSurface"))

"""
$(TYPEDSIGNATURES)

Perform one strong-stability-preserving Runge-Kutta stage for `HydrostaticFreeSurfaceModel`.

Velocities and thickness-weighted tracers take an implicit-explicit forward-Euler step of `Δt` from the previous
stage and are blended with the state at `tⁿ` by the Shu-Osher pair `(a, b)`. The barotropic solve restarts from
`(ηⁿ, Uⁿ)` and spans `[tⁿ, tⁿ + Δt]`; its increment is blended, `Ψᵐ = a Ψⁿ + b (Ψᵐ⁻¹ + Ψ⋆ - Ψⁿ)`. Every solve is driven by
the slow forcing of its own stage, and the increment blend gives the stage forcings the weights `(1/6, 1/6, 2/3)`, the
same weights the blend gives the baroclinic tendencies.
"""
function TimeSteppers.ssp_substep!(model::HydrostaticFreeSurfaceModel, Δt, a, b, callbacks)
    grid         = model.grid
    free_surface = model.free_surface
    timestepper  = model.timestepper
    a, b         = convert.(eltype(grid), (a, b))

    # As in split Runge-Kutta, exterior states step from their value at tⁿ to the time the stage reaches
    stage_Δt = ssp_stage_fractions(timestepper.coefficients)[model.clock.stage] * Δt

    @apply_regionally begin
        update_transport_velocities!(model.transport_velocities, model.velocities, free_surface)
        compute_momentum_flux_bcs!(model)
        ssp_euler_substep_velocities!(model.velocities, model, Δt)
    end

    compute_free_surface_tendency!(grid, model, free_surface, Δt)
    update_stage_free_surface!(free_surface, free_surface_fields(free_surface), timestepper)
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
        correct_barotropic_mode!(model, Δt)
        ssp_substep_tracers!(model.tracers, model, Δt, a, b)
        step_boundary_conditions!(model.tracers, model.clock, fields(model), stage_Δt)
    end

    return nothing
end

update_stage_free_surface!(args...) = nothing

function update_stage_free_surface!(::SplitExplicitFreeSurface, Ψᵐ⁻¹, Ψᵐ)
    parent(Ψᵐ⁻¹.η) .= parent(Ψᵐ.η)
    parent(Ψᵐ⁻¹.U) .= parent(Ψᵐ.U)
    parent(Ψᵐ⁻¹.V) .= parent(Ψᵐ.V)
    return nothing
end

#####
##### Velocities: the Euler step from the previous stage, and the blend once the barotropic solve is done
#####

function ssp_euler_substep_velocities!(velocities, model, Δt)
    grid = model.grid
    u, v = velocities.u, velocities.v
    Gⁿ   = model.timestepper.Gⁿ

    launch!(architecture(grid), grid, :xyz, _rk_substep_field!, u, Δt, Gⁿ.u, u; exclude_periphery=true)
    launch!(architecture(grid), grid, :xyz, _rk_substep_field!, v, Δt, Gⁿ.v, v; exclude_periphery=true)
    implicit_substep_velocity!(model, Δt, Val(:u))
    implicit_substep_velocity!(model, Δt, Val(:v))

    return nothing
end

function ssp_blend_velocities!(velocities, model, a, b, stage_Δt)
    grid = model.grid
    u, v = velocities.u, velocities.v
    Ψ⁻   = model.timestepper.Ψ⁻

    launch!(architecture(grid), grid, :xyz, _ssp_blend_field!, u, Ψ⁻.u, a, b; exclude_periphery=true)
    launch!(architecture(grid), grid, :xyz, _ssp_blend_field!, v, Ψ⁻.v, a, b; exclude_periphery=true)
    step_boundary_conditions!(velocities, model.clock, fields(model), stage_Δt)

    return nothing
end

@kernel function _ssp_blend_field!(field, Ψ⁻, a, b)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = a * Ψ⁻[i, j, k] + b * field[i, j, k]
end

#####
##### Tracers: the Euler step from the previous stage, the implicit step, and the blend
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
    Gⁿ   = model.timestepper.Gⁿ[tracer_name]
    σc⁻  = model.timestepper.Ψ⁻[tracer_name]
    c    = model.tracers[tracer_name]

    launch!(architecture(grid), grid, :xyz, _ssp_euler_substep_tracer_field!, c, grid, Δt, Gⁿ)

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

    launch!(architecture(grid), grid, :xyz, _ssp_blend_tracer_field!, c, grid, σc⁻, a, b)

    return nothing
end

# (σĉ) = σᵐ⁻¹ cᵐ⁻¹ + Δt Gᵐ, where σ⁻ holds σᵐ⁻¹ once the grid has moved to σᵐ
@kernel function _ssp_euler_substep_tracer_field!(c, grid, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)
    σᶜᶜ⁻   = σ⁻(i, j, k, grid, Center(), Center(), Center())
    σ⁻¹ᶜᶜⁿ = σ⁻¹ⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (σᶜᶜ⁻ * c[i, j, k] + Δt * Gⁿ[i, j, k]) * σ⁻¹ᶜᶜⁿ
end

#####
##### Barotropic mode: every solve restarts from ψⁿ, so the blend applies its increment to the previous stage
#####

blend_free_surface!(free_surface, timestepper, stage) = nothing

function blend_free_surface!(free_surface, timestepper::SSPRungeKutta3TimeStepper, stage)
    grid = free_surface.displacement.grid
    a, b = convert.(eltype(grid), timestepper.coefficients[stage])
    ψ    = free_surface_fields(free_surface)

    launch!(architecture(grid), grid, :xy, _blend_free_surface!, ψ, timestepper.Ψ⁻[keys(ψ)], timestepper.Ψᵐ⁻¹, a, b, grid.Nz + 1)

    return nothing
end

# ψ holds the output of the solve that restarted from ψⁿ
@kernel function _blend_free_surface!(ψ, ψⁿ, ψᵐ⁻¹, a, b, k_top)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ψ.η[i, j, k_top] = a * ψⁿ.η[i, j, k_top] + b * (ψᵐ⁻¹.η[i, j, k_top] + ψ.η[i, j, k_top] - ψⁿ.η[i, j, k_top])
        ψ.U[i, j, 1]     = a * ψⁿ.U[i, j, 1]     + b * (ψᵐ⁻¹.U[i, j, 1]     + ψ.U[i, j, 1]     - ψⁿ.U[i, j, 1])
        ψ.V[i, j, 1]     = a * ψⁿ.V[i, j, 1]     + b * (ψᵐ⁻¹.V[i, j, 1]     + ψ.V[i, j, 1]     - ψⁿ.V[i, j, 1])
    end
end
