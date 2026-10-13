using Oceananigans: fields
using Oceananigans.Grids: topology, Flat
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel

"""
    StageTransport()

The tracers of a stage are advected by a transport that the stage itself builds, after its barotropic step, from
the velocities the previous stage ended with. The flux of a stage is therefore known only at its end, from the
tracer and the grid the previous stage ended with.
"""
struct StageTransport end

"""
    NextStageTransport()

The tracers of a stage are advected by the velocities the previous stage ended with. The flux of the next stage is
known at the end of the current one.
"""
struct NextStageTransport end

transport_timing(model) = NextStageTransport()
transport_timing(::HydrostaticFreeSurfaceModel) = StageTransport()

advecting_velocities(model) = model.velocities
advecting_velocities(model::HydrostaticFreeSurfaceModel) = model.transport_velocities

function update_variance_budget!(ϵ, model, timing::StageTransport)
    cache_advective_fluxes!(ϵ, model, timing)
    compute_dissipation!(ϵ, model, ϵ.tracer_name)
    cache_diffusive_fluxes!(ϵ, model)
    store_stage_state!(ϵ, model)
    return nothing
end

function update_variance_budget!(ϵ, model, timing::NextStageTransport)
    compute_dissipation!(ϵ, model, ϵ.tracer_name)
    store_stage_state!(ϵ, model)
    cache_advective_fluxes!(ϵ, model, timing)
    cache_diffusive_fluxes!(ϵ, model)
    return nothing
end

function face_ranges(grid)
    Nx, Ny, Nz = size(grid)
    TX, TY, TZ = topology(grid)
    Fx = ifelse(TX == Flat, 1:1, 1:Nx+1)
    Fy = ifelse(TY == Flat, 1:1, 1:Ny+1)
    Fz = ifelse(TZ == Flat, 1:1, 1:Nz+1)
    return Fx, Fy, Fz
end

flux_parameters(grid) = KernelParameters(face_ranges(grid)...)

function horizontal_flux_parameters(grid)
    Fx, Fy, _ = face_ranges(grid)
    return KernelParameters(Fx, Fy)
end

"""
    stage_weights(timestepper, stage, timing, FT)

The weight with which the flux cached at the end of `stage` enters the flux of the step, and whether the sum
already under way is kept, or `nothing` if that flux does not enter the step.
"""
stage_weights(::QuasiAdamsBashforth2TimeStepper, stage, timing, FT) = (one(FT), zero(FT))

stage_weights(ts::SplitRungeKuttaTimeStepper, stage, ::StageTransport, FT) =
    stage == ts.Nstages ? (one(FT), zero(FT)) : nothing

stage_weights(ts::SplitRungeKuttaTimeStepper, stage, ::NextStageTransport, FT) =
    stage == ts.Nstages - 1 ? (one(FT), zero(FT)) : nothing

function stage_weights(ts::SSPRungeKuttaTimeStepper, stage, ::StageTransport, FT)
    β = ssp_quadrature_weights(ts.coefficients)
    return convert(FT, β[stage]), convert(FT, stage == 1 ? 0 : 1)
end

stage_weights(ts::SSPRungeKuttaTimeStepper, stage, ::NextStageTransport, FT) = ssp_accumulation_weights(ts, stage, FT)

"""
    ssp_accumulation_weights(ts, stage, FT)

The weight this stage's flux carries in the running sum `F̄ = Σₘ βₘ Fᵐ`, and whether the sum already
under way is kept or restarted.

The callback fires at the end of stage `m`, where the state is `yᵐ`, the argument of the tendency of
stage `m+1`, hence weight `β_{m+1}`. At the final stage the assembly has already consumed the sum, so
that flux restarts it with `β₁` and seeds the next step.
"""
@inline function ssp_accumulation_weights(ts::SSPRungeKuttaTimeStepper, stage, FT)
    β     = ssp_quadrature_weights(ts.coefficients)
    seeds = stage == ts.Nstages
    w     = seeds ? β[1] : β[stage+1]
    return convert(FT, w), convert(FT, seeds ? 0 : 1)
end

# Adams-Bashforth combines the flux of the step with the one of the previous step
store_previous_advective_fluxes!(dissipation, timestepper) = nothing

function store_previous_advective_fluxes!(dissipation, ::QuasiAdamsBashforth2TimeStepper)
    Fⁿ, Fⁿ⁻¹ = dissipation.advective_fluxes
    Uⁿ, Uⁿ⁻¹ = dissipation.previous_state.Uⁿ, dissipation.previous_state.Uⁿ⁻¹

    for (previous, current) in zip((Fⁿ⁻¹..., Uⁿ⁻¹...), (Fⁿ..., Uⁿ...))
        parent(previous) .= parent(current)
    end

    return nothing
end

function cache_advective_fluxes!(dissipation, model, timing)
    grid = model.grid
    timestepper = model.timestepper
    weights = stage_weights(timestepper, model.clock.stage, timing, eltype(grid))
    isnothing(weights) && return nothing

    advection = getadvection(model.advection, dissipation.tracer_name)
    advection isa Nothing && return nothing

    weight, keep = weights
    state = dissipation.previous_state

    store_previous_advective_fluxes!(dissipation, timestepper)

    launch!(architecture(grid), grid, flux_parameters(grid), _accumulate_advective_fluxes!,
            dissipation.advective_fluxes.Fⁿ, state.Uⁿ, grid, advection, advecting_velocities(model),
            state.cᵐ⁻¹, state.σᵐ⁻¹, weight, keep)

    return nothing
end

function cache_diffusive_fluxes!(dissipation, model)
    grid   = model.grid
    params = flux_parameters(grid)
    tracer_name = dissipation.tracer_name
    tracer_id   = findfirst(x -> x == tracer_name, keys(model.tracers))
    c = model.tracers[tracer_name]

    Vⁿ   = dissipation.diffusive_fluxes.Vⁿ
    Vⁿ⁻¹ = dissipation.diffusive_fluxes.Vⁿ⁻¹

    cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, grid, params, model.timestepper, model.clock.stage, model.closure,
                           model.closure_fields, model.buoyancy, c, Val(tracer_id), model.clock, fields(model))

    return nothing
end

cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, grid, params, ::QuasiAdamsBashforth2TimeStepper, stage, clo, D, B, c, tracer_id, clk, model_fields) =
    launch!(architecture(grid), grid, params, _cache_diffusive_fluxes!, Vⁿ, Vⁿ⁻¹, grid, clo, D, B, c, tracer_id, clk, model_fields)

function cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, grid, params, ts::SplitRungeKuttaTimeStepper, stage, clo, D, B, c, tracer_id, clk, model_fields)
    if stage == ts.Nstages-1
        launch!(architecture(grid), grid, params, _cache_diffusive_fluxes!, Vⁿ, grid, clo, D, B, c, tracer_id, clk, model_fields)
    end
end

function cache_diffusive_fluxes(Vⁿ, V̄, grid, params, ts::SSPRungeKuttaTimeStepper, stage, clo, D, B, c, tracer_id, clk, model_fields)
    β, keep = ssp_accumulation_weights(ts, stage, eltype(grid))
    launch!(architecture(grid), grid, params, _accumulate_ssp_diffusive_fluxes!,
            V̄, Vⁿ, grid, clo, D, B, c, tracer_id, clk, model_fields, β, keep)
end

end_of_step(::QuasiAdamsBashforth2TimeStepper, stage) = true
end_of_step(ts::MultiStageTimeStepper, stage) = stage == ts.Nstages

function store_stage_state!(dissipation, model)
    grid  = model.grid
    state = dissipation.previous_state
    c     = model.tracers[dissipation.tracer_name]

    parent(state.cᵐ⁻¹) .= parent(c)
    launch!(architecture(grid), grid, horizontal_flux_parameters(grid), _store_horizontal_stretching!, state.σᵐ⁻¹, grid)

    if end_of_step(model.timestepper, model.clock.stage)
        parent(state.cⁿ⁻¹) .= parent(c)
    end

    return nothing
end

@kernel function _store_horizontal_stretching!(σ, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        σ.x[i, j, 1] = σⁿ(i, j, 1, grid, f, c, c)
        σ.y[i, j, 1] = σⁿ(i, j, 1, grid, c, f, c)
    end
end
