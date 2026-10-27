using Oceananigans: fields
using Oceananigans.Grids: topology, Flat
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, tracer_advecting_velocities, FluxCorrectedWENO,
                                                       compute_antidiffusive_limiters!, rejected_antidiffusion

advecting_velocities(model, tracer_name) = model.velocities

function advecting_velocities(model::HydrostaticFreeSurfaceModel, tracer_name)
    forcing = model.forcing[tracer_name]
    return tracer_advecting_velocities(model.transport_velocities, model.biogeochemistry, model.closure, model.closure_fields, forcing, Val(tracer_name))
end

"""
    stage_weights(timestepper, stage, FT)

The weight with which the fluxes of `stage` enter the fluxes of the step, and whether the sum already under way
is kept, or `nothing` if the fluxes of `stage` do not enter the step.
"""
stage_weights(::QuasiAdamsBashforth2TimeStepper, stage, FT) = (one(FT), zero(FT))

stage_weights(ts::SplitRungeKuttaTimeStepper, stage, FT) = stage == ts.Nstages ? (one(FT), zero(FT)) : nothing

function stage_weights(ts::SSPRungeKuttaTimeStepper, stage, FT)
    β = ssp_quadrature_weights(ts.coefficients)
    return convert(FT, β[stage]), convert(FT, stage == 1 ? 0 : 1)
end

function flux_parameters(grid)
    Nx, Ny, Nz = size(grid)
    TX, TY, TZ = topology(grid)
    Fx = ifelse(TX == Flat, 1:1, 1:Nx+1)
    Fy = ifelse(TY == Flat, 1:1, 1:Ny+1)
    Fz = ifelse(TZ == Flat, 1:1, 1:Nz+1)
    return KernelParameters(Fx, Fy, Fz)
end

# Adams-Bashforth combines the fluxes of the step with those of the previous step
store_previous_fluxes!(dissipation, timestepper) = nothing

function store_previous_fluxes!(dissipation, ::QuasiAdamsBashforth2TimeStepper)
    Fⁿ, Fⁿ⁻¹ = dissipation.advective_fluxes.Fⁿ, dissipation.advective_fluxes.Fⁿ⁻¹
    Vⁿ, Vⁿ⁻¹ = dissipation.diffusive_fluxes.Vⁿ, dissipation.diffusive_fluxes.Vⁿ⁻¹
    Uⁿ, Uⁿ⁻¹ = dissipation.previous_state.Uⁿ, dissipation.previous_state.Uⁿ⁻¹

    for (previous, current) in zip((Fⁿ⁻¹..., Vⁿ⁻¹..., Uⁿ⁻¹...), (Fⁿ..., Vⁿ..., Uⁿ...))
        parent(previous) .= parent(current)
    end

    return nothing
end

# Accumulate the transport and the advective and diffusive fluxes with which the tracer tendency of the stage is computed
function cache_fluxes!(dissipation, model, tracer_name)
    grid = model.grid
    timestepper = model.timestepper
    weights = stage_weights(timestepper, model.clock.stage, eltype(grid))
    isnothing(weights) && return nothing

    weight, keep = weights
    store_previous_fluxes!(dissipation, timestepper)

    c = model.tracers[tracer_name]
    U = advecting_velocities(model, tracer_name)
    params = flux_parameters(grid)

    ####
    #### Update the advective fluxes
    ####

    Fⁿ = dissipation.advective_fluxes.Fⁿ
    Uⁿ = dissipation.previous_state.Uⁿ
    advection = getadvection(model.advection, tracer_name)

    if !isnothing(advection)
        launch!(architecture(grid), grid, params, _accumulate_advective_fluxes!, Fⁿ, Uⁿ, grid, advection, U, c, weight, keep)
        limit_advective_fluxes!(Fⁿ, params, timestepper, advection, model, tracer_name, U)
    end

    ####
    #### Update the diffusive fluxes
    ####

    Vⁿ = dissipation.diffusive_fluxes.Vⁿ
    κᶻ = dissipation.diffusive_fluxes.κᶻ
    tracer_id = findfirst(x -> x == tracer_name, keys(model.tracers))

    launch!(architecture(grid), grid, params, _accumulate_diffusive_fluxes!, Vⁿ, κᶻ, grid, model.closure, model.closure_fields,
            model.buoyancy, c, Val(tracer_id), model.clock, fields(model), weight, keep)

    return nothing
end

# the fluxes the substep takes through the faces where its flux correction limits Fᴴ, recomputed from the same stage state and tendency
limit_advective_fluxes!(Fⁿ, params, timestepper, advection, model, tracer_name, U) = nothing

function limit_advective_fluxes!(Fⁿ, params, timestepper::SplitRungeKuttaTimeStepper, advection::FluxCorrectedWENO, model, tracer_name, U)
    flux_correction = timestepper.flux_correction
    isnothing(flux_correction) && return nothing

    grid = model.grid
    Gⁿ = timestepper.Gⁿ[tracer_name]
    Ψ⁻ = timestepper.Ψ⁻[tracer_name]
    η = isnothing(model.free_surface) ? nothing : model.free_surface.displacement
    compute_antidiffusive_limiters!(flux_correction, Gⁿ, model.tracers[tracer_name], Ψ⁻, η, grid, model.clock.last_stage_Δt, advection, U)

    (; Aˣ, Aʸ, Aᶻ, R⁺, R⁻) = flux_correction
    launch!(architecture(grid), grid, params, _subtract_rejected_fluxes!, Fⁿ, grid, R⁺, R⁻, Aˣ, Aʸ, Aᶻ)

    return nothing
end

# neighbours along flat directions collapse onto the cell, where the antidiffusive fluxes vanish
@kernel function _subtract_rejected_fluxes!(Fⁿ, grid, R⁺, R⁻, Aˣ, Aʸ, Aᶻ)
    i, j, k = @index(Global, NTuple)
    TX, TY, TZ = topology(grid)
    sx, sy, sz = TX !== Flat, TY !== Flat, TZ !== Flat

    @inbounds begin
        Fⁿ.x[i, j, k] -= rejected_antidiffusion(Aˣ[i, j, k], R⁺, R⁻, (i - sx, j, k), (i, j, k))
        Fⁿ.y[i, j, k] -= rejected_antidiffusion(Aʸ[i, j, k], R⁺, R⁻, (i, j - sy, k), (i, j, k))
        Fⁿ.z[i, j, k] -= rejected_antidiffusion(Aᶻ[i, j, k], R⁺, R⁻, (i, j, k - sz), (i, j, k))
    end
end
