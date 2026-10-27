using Oceananigans: fields
using Oceananigans.Grids: topology, Flat
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, tracer_advecting_velocities, FluxCorrectedWENO,
                                                       compute_antidiffusive_limiters!, rejected_antidiffusion

advecting_velocities(model, tracer_name) = model.velocities

function advecting_velocities(model::HydrostaticFreeSurfaceModel, tracer_name)
    forcing = model.forcing[tracer_name]
    return tracer_advecting_velocities(model.transport_velocities, model.biogeochemistry, model.closure, model.closure_fields, forcing, Val(tracer_name))
end

# Store the transport and the advective and diffusive fluxes with which the tracer tendency is computed
function cache_fluxes!(dissipation, model, tracer_name)
    grid = model.grid
    sz   = size(model.tracers[1].data)
    of   = OffsetArrays.offsets(model.tracers[1].data)

    params = KernelParameters(sz, of)

    Uⁿ   = dissipation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation.previous_state.Uⁿ⁻¹
    U    = advecting_velocities(model, tracer_name)
    timestepper = model.timestepper

    update_transport!(Uⁿ, Uⁿ⁻¹, grid, params, timestepper, U)
    tracer_id = findfirst(x -> x == tracer_name, keys(model.tracers))
    cache_fluxes!(dissipation, model, tracer_name, U, Val(tracer_id))

    return nothing
end

function flux_parameters(grid)
    Nx, Ny, Nz = size(grid)
    TX, TY, TZ = topology(grid)
    Fx = ifelse(TX == Flat, 1:1, 1:Nx+1)
    Fy = ifelse(TY == Flat, 1:1, 1:Ny+1)
    Fz = ifelse(TZ == Flat, 1:1, 1:Nz+1)
    return KernelParameters(Fx, Fy, Fz)
end

function cache_fluxes!(dissipation, model, tracer_name::Symbol, U, tracer_id)

    c = model.tracers[tracer_name]

    grid = model.grid
    params = flux_parameters(grid)
    timestepper = model.timestepper

    ####
    #### Update the advective fluxes
    ####

    Fⁿ   = dissipation.advective_fluxes.Fⁿ
    Fⁿ⁻¹ = dissipation.advective_fluxes.Fⁿ⁻¹
    advection = getadvection(model.advection, tracer_name)

    cache_advective_fluxes!(Fⁿ, Fⁿ⁻¹, grid, params, timestepper, advection, U, c)
    limit_advective_fluxes!(Fⁿ, params, timestepper, advection, model, tracer_name, U)

    ####
    #### Update the diffusive fluxes
    ####

    Vⁿ   = dissipation.diffusive_fluxes.Vⁿ
    Vⁿ⁻¹ = dissipation.diffusive_fluxes.Vⁿ⁻¹
    κᶻ   = dissipation.diffusive_fluxes.κᶻ

    D = model.closure_fields
    B = model.buoyancy
    clk  = model.clock
    clo  = model.closure
    model_fields = fields(model)

    cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, κᶻ, grid, params, timestepper, clo, D, B, c, tracer_id, clk, model_fields)

    return nothing
end

cache_advective_fluxes!(Fⁿ, Fⁿ⁻¹, grid, params, ::QuasiAdamsBashforth2TimeStepper, advection, U, c) =
    launch!(architecture(grid), grid, params, _cache_advective_fluxes!, Fⁿ, Fⁿ⁻¹, grid, advection, U, c)

cache_advective_fluxes!(Fⁿ, Fⁿ⁻¹, grid, params, ::SplitRungeKuttaTimeStepper, advection, U, c) =
    launch!(architecture(grid), grid, params, _cache_advective_fluxes!, Fⁿ, grid, advection, U, c)

cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, κᶻ, grid, params, ::QuasiAdamsBashforth2TimeStepper, clo, D, B, c, tracer_id, clk, model_fields) =
    launch!(architecture(grid), grid, params, _cache_diffusive_fluxes!, Vⁿ, Vⁿ⁻¹, κᶻ, grid, clo, D, B, c, tracer_id, clk, model_fields)

cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, κᶻ, grid, params, ::SplitRungeKuttaTimeStepper, clo, D, B, c, tracer_id, clk, model_fields) =
    launch!(architecture(grid), grid, params, _cache_diffusive_fluxes!, Vⁿ, κᶻ, grid, clo, D, B, c, tracer_id, clk, model_fields)

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

update_transport!(Uⁿ, Uⁿ⁻¹, grid, params, ::QuasiAdamsBashforth2TimeStepper, U) =
    launch!(architecture(grid), grid, params, _update_transport!, Uⁿ, Uⁿ⁻¹, grid, U)

update_transport!(Uⁿ, Uⁿ⁻¹, grid, params, ::SplitRungeKuttaTimeStepper, U) =
    launch!(architecture(grid), grid, params, _update_transport!, Uⁿ, grid, U)

@kernel function _update_transport!(Uⁿ, Uⁿ⁻¹, grid, U)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Uⁿ⁻¹.u[i, j, k] = Uⁿ.u[i, j, k]
        Uⁿ⁻¹.v[i, j, k] = Uⁿ.v[i, j, k]
        Uⁿ⁻¹.w[i, j, k] = Uⁿ.w[i, j, k]
          Uⁿ.u[i, j, k] = U.u[i, j, k] * Axᶠᶜᶜ(i, j, k, grid)
          Uⁿ.v[i, j, k] = U.v[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid)
          Uⁿ.w[i, j, k] = U.w[i, j, k] * Azᶜᶜᶠ(i, j, k, grid)
    end
end

@kernel function _update_transport!(Uⁿ, grid, U)
    i, j, k = @index(Global, NTuple)

    @inbounds Uⁿ.u[i, j, k] = U.u[i, j, k] * Axᶠᶜᶜ(i, j, k, grid)
    @inbounds Uⁿ.v[i, j, k] = U.v[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid)
    @inbounds Uⁿ.w[i, j, k] = U.w[i, j, k] * Azᶜᶜᶠ(i, j, k, grid)
end
