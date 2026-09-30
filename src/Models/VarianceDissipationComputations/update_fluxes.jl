using Oceananigans: fields
using Oceananigans.Grids: topology, Flat
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel

# Store advective and diffusive fluxes for dissipation computation
function cache_fluxes!(dissipation, model, tracer_name)
    grid = model.grid
    sz   = size(model.tracers[1].data)
    of   = model.tracers[1].data.offsets

    params = KernelParameters(sz, of)

    Uⁿ   = dissipation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation.previous_state.Uⁿ⁻¹
    U    = model isa HydrostaticFreeSurfaceModel ? model.transport_velocities : model.velocities
    timestepper = model.timestepper
    stage = model.clock.stage

    update_transport!(Uⁿ, Uⁿ⁻¹, grid, params, timestepper, stage, U)
    store_specific_thickness!(dissipation.previous_state.specific_thickness, grid, params, timestepper, stage)
    tracer_id = findfirst(x -> x == tracer_name, keys(model.tracers))
    cache_fluxes!(dissipation, model, tracer_name, Val(tracer_id))

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

function cache_fluxes!(dissipation, model, tracer_name::Symbol, tracer_id)

    # Grab tracer properties
    c    = model.tracers[tracer_name]
    cⁿ⁻¹ = dissipation.previous_state.cⁿ⁻¹

    grid = model.grid
    U = model isa HydrostaticFreeSurfaceModel ? model.transport_velocities : model.velocities
    params = flux_parameters(grid)
    stage  = model.clock.stage
    timestepper = model.timestepper

    ####
    #### Update the advective fluxes and compute gradient squared
    ####

    Fⁿ   = dissipation.advective_fluxes.Fⁿ
    Fⁿ⁻¹ = dissipation.advective_fluxes.Fⁿ⁻¹
    advection = getadvection(model.advection, tracer_name)

    cache_advective_fluxes!(Fⁿ, Fⁿ⁻¹, grid, params, timestepper, stage, advection, U, c)

    ####
    #### Update the diffusive fluxes
    ####

    Vⁿ   = dissipation.diffusive_fluxes.Vⁿ
    Vⁿ⁻¹ = dissipation.diffusive_fluxes.Vⁿ⁻¹

    D = model.closure_fields
    B = model.buoyancy
    clk  = model.clock
    clo  = model.closure
    model_fields = fields(model)

    cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, grid, params, timestepper, stage, clo, D, B, c, tracer_id, clk, model_fields)

    if timestepper isa QuasiAdamsBashforth2TimeStepper
        parent(cⁿ⁻¹) .= parent(c)
    elseif (timestepper isa MultiStageTimeStepper) && (stage == timestepper.Nstages)
        parent(cⁿ⁻¹) .= parent(c)
    end

    return nothing
end

cache_advective_fluxes!(Fⁿ, Fⁿ⁻¹, grid, params, ::QuasiAdamsBashforth2TimeStepper, stage, advection, U, c) =
    launch!(architecture(grid), grid, params, _cache_advective_fluxes!, Fⁿ, Fⁿ⁻¹, grid, advection, U, c)

function cache_advective_fluxes!(Fⁿ, Fⁿ⁻¹, grid, params, ts::SplitRungeKuttaTimeStepper, stage, advection, U, c)
    if stage == ts.Nstages-1
        launch!(architecture(grid), grid, params, _cache_advective_fluxes!, Fⁿ, grid, advection, U, c)
    end
end

# Weight of the flux in the running sum Σₘ βₘ Fᵐ, and whether the sum is kept. Stage m ends on the state that stage
# m+1 differentiates, and the last stage, whose sum is already assembled, seeds the next time step.
@inline function ssp_accumulation_weights(ts::SSPRungeKutta3TimeStepper, stage, FT)
    β = ssp_quadrature_weights(ts.coefficients)
    last_stage = stage == ts.Nstages
    return convert(FT, last_stage ? β[1] : β[stage+1]), convert(FT, !last_stage)
end

function cache_advective_fluxes!(Fⁿ, Fⁿ⁻¹, grid, params, ts::SSPRungeKutta3TimeStepper, stage, advection, U, c)
    β, keep = ssp_accumulation_weights(ts, stage, eltype(grid))
    launch!(architecture(grid), grid, params, _accumulate_ssp_advective_fluxes!, Fⁿ, grid, advection, U, c, β, keep)
end

cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, grid, params, ::QuasiAdamsBashforth2TimeStepper, stage, clo, D, B, c, tracer_id, clk, model_fields) =
    launch!(architecture(grid), grid, params, _cache_diffusive_fluxes!, Vⁿ, Vⁿ⁻¹, grid, clo, D, B, c, tracer_id, clk, model_fields)

function cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, grid, params, ts::SplitRungeKuttaTimeStepper, stage, clo, D, B, c, tracer_id, clk, model_fields)
    if stage == ts.Nstages-1
        launch!(architecture(grid), grid, params, _cache_diffusive_fluxes!, Vⁿ, grid, clo, D, B, c, tracer_id, clk, model_fields)
    end
end

function cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, grid, params, ts::SSPRungeKutta3TimeStepper, stage, clo, D, B, c, tracer_id, clk, model_fields)
    β, keep = ssp_accumulation_weights(ts, stage, eltype(grid))
    launch!(architecture(grid), grid, params, _accumulate_ssp_diffusive_fluxes!, Vⁿ, Vⁿ⁻¹, grid, clo, D, B, c, tracer_id, clk, model_fields, β, keep)
end

update_transport!(Uⁿ, Uⁿ⁻¹, grid, params, ::QuasiAdamsBashforth2TimeStepper, stage, U) =
    launch!(architecture(grid), grid, params, _update_transport!, Uⁿ, Uⁿ⁻¹, grid, U)

function update_transport!(Uⁿ, Uⁿ⁻¹, grid, params, ts::SplitRungeKuttaTimeStepper, stage, U)
    if stage == ts.Nstages-1
        launch!(architecture(grid), grid, params, _update_transport!, Uⁿ, grid, U)
    end
end

function update_transport!(Uⁿ, Uⁿ⁻¹, grid, params, ts::SSPRungeKutta3TimeStepper, stage, U)
    β, keep = ssp_accumulation_weights(ts, stage, eltype(grid))
    launch!(architecture(grid), grid, params, _accumulate_ssp_transport!, Uⁿ, grid, U, β, keep)
end

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

store_specific_thickness!(σ, grid, params, timestepper, stage) = nothing

function store_specific_thickness!(σ, grid, params, ts::SplitRungeKuttaTimeStepper, stage)
    if stage == ts.Nstages-1
        launch!(architecture(grid), grid, params, _store_specific_thickness!, σ, grid)
    end
    return nothing
end

@kernel function _store_specific_thickness!(σ, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        σ.x[i, j, 1] = σⁿ(i, j, k, grid, f, c, c)
        σ.y[i, j, 1] = σⁿ(i, j, k, grid, c, f, c)
        σ.z[i, j, 1] = σⁿ(i, j, k, grid, c, c, f)
    end
end
