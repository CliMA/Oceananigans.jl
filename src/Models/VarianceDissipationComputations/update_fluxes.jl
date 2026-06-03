using Oceananigans: fields
using Oceananigans.Advection: spherical_shell_volume_flux_velocities,
                              u_velocity, v_velocity, w_velocity
using Oceananigans.BoundaryConditions: update_boundary_conditions!
using Oceananigans.Fields: compute!
using Oceananigans.Grids: topology, Flat
using Oceananigans.Models: refresh_tracer_auxiliary_velocity_halos!,
                           refresh_tracer_advective_forcing_halos!,
                           update_model_field_time_series!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel,
                                                        refresh_transport_advection_state!
using Oceananigans.Models.NonhydrostaticModels: NonhydrostaticModel,
                                                refresh_background_field_halos!
using Oceananigans.TurbulenceClosures: compute_closure_fields!

@inline advective_velocity_fields(model) = model.velocities
@inline advective_velocity_fields(model::HydrostaticFreeSurfaceModel) = model.transport_velocities
@inline advective_velocity_fields(model, tracer_name) = advective_velocity_fields(model)

@inline function refresh_variance_dissipation_advection_state!(model)
    update_model_field_time_series!(model, model.clock)
    return nothing
end

@inline function refresh_variance_dissipation_advection_state!(model::HydrostaticFreeSurfaceModel)
    update_model_field_time_series!(model, model.clock)
    refresh_transport_advection_state!(model, model.velocities)
    compute_closure_fields!(model.closure_fields, model.closure, model)
    compute!(model.auxiliary_fields)
    update_boundary_conditions!(fields(model), model)
    return nothing
end

@inline function refresh_variance_dissipation_advection_state!(model::NonhydrostaticModel)
    update_model_field_time_series!(model, model.clock)
    refresh_background_field_halos!(model.background_fields)
    compute_closure_fields!(model.closure_fields, model.closure, model)
    compute!(model.auxiliary_fields)
    update_boundary_conditions!(fields(model), model)
    return nothing
end

@inline function hydrostatic_tracer_advective_velocity_fields(model::HydrostaticFreeSurfaceModel, tracer_name)
    tracer_name_val = Val(tracer_name)
    @inbounds forcing = model.forcing[tracer_name]

    biogeochemical_velocities = biogeochemical_drift_velocity(model.biogeochemistry, tracer_name_val)
    closure_velocities = closure_auxiliary_velocity(model.closure, model.closure_fields, tracer_name_val)

    refresh_tracer_auxiliary_velocity_halos!(biogeochemical_velocities)
    refresh_tracer_auxiliary_velocity_halos!(closure_velocities)
    refresh_tracer_advective_forcing_halos!(forcing)

    auxiliary_velocities = tracer_auxiliary_velocities(biogeochemical_velocities,
                                                       closure_velocities,
                                                       forcing)

    return total_tracer_advection_velocities(model.grid, model.transport_velocities, auxiliary_velocities)
end

@inline advective_velocity_fields(model::HydrostaticFreeSurfaceModel, tracer_name) =
    hydrostatic_tracer_advective_velocity_fields(model, tracer_name)

@inline function nonhydrostatic_tracer_advective_velocity_fields(model::NonhydrostaticModel, tracer_name)
    tracer_name_val = Val(tracer_name)
    @inbounds forcing = model.forcing[tracer_name]

    biogeochemical_velocities = biogeochemical_drift_velocity(model.biogeochemistry, tracer_name_val)
    closure_velocities = closure_auxiliary_velocity(model.closure, model.closure_fields, tracer_name_val)
    auxiliary_velocities = sum_of_velocities(biogeochemical_velocities, closure_velocities)

    refresh_tracer_auxiliary_velocity_halos!(biogeochemical_velocities)
    refresh_tracer_auxiliary_velocity_halos!(closure_velocities)
    refresh_tracer_advective_forcing_halos!(forcing)

    total_velocities = sum_of_velocities(model.velocities,
                                         model.background_fields.velocities,
                                         auxiliary_velocities)

    return with_advective_forcing(forcing, total_velocities)
end

@inline advective_velocity_fields(model::NonhydrostaticModel, tracer_name) =
    nonhydrostatic_tracer_advective_velocity_fields(model, tracer_name)

@inline function advective_velocity_fields(model::NonhydrostaticModel{<:Any, <:Any, <:Any, <:SphericalShellGrid})
    return spherical_shell_volume_flux_velocities(model.grid, model.velocities)
end

@inline function advective_velocity_fields(model::NonhydrostaticModel{<:Any, <:Any, <:Any, <:SphericalShellGrid}, tracer_name)
    total_velocities = nonhydrostatic_tracer_advective_velocity_fields(model, tracer_name)

    return spherical_shell_volume_flux_velocities(model.grid, total_velocities)
end

# Store advective and diffusive fluxes for dissipation computation
function cache_fluxes!(dissipation, model, tracer_name)
    refresh_variance_dissipation_advection_state!(model)

    grid = model.grid
    sz   = size(model.tracers[1].data)
    of   = model.tracers[1].data.offsets

    params = KernelParameters(sz, of)

    Uⁿ   = dissipation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation.previous_state.Uⁿ⁻¹
    U    = advective_velocity_fields(model, tracer_name)
    timestepper = model.timestepper
    stage = model.clock.stage

    update_transport!(Uⁿ, Uⁿ⁻¹, model, grid, params, timestepper, stage, U)
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
    U = advective_velocity_fields(model, tracer_name)
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
    elseif (timestepper isa SplitRungeKuttaTimeStepper) && (stage == timestepper.Nstages)
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

cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, grid, params, ::QuasiAdamsBashforth2TimeStepper, stage, clo, D, B, c, tracer_id, clk, model_fields) =
    launch!(architecture(grid), grid, params, _cache_diffusive_fluxes!, Vⁿ, Vⁿ⁻¹, grid, clo, D, B, c, tracer_id, clk, model_fields)

function cache_diffusive_fluxes(Vⁿ, Vⁿ⁻¹, grid, params, ts::SplitRungeKuttaTimeStepper, stage, clo, D, B, c, tracer_id, clk, model_fields)
    if stage == ts.Nstages-1
        launch!(architecture(grid), grid, params, _cache_diffusive_fluxes!, Vⁿ, grid, clo, D, B, c, tracer_id, clk, model_fields)
    end
end

update_transport!(Uⁿ, Uⁿ⁻¹, model, grid, params, ::QuasiAdamsBashforth2TimeStepper, stage, U) =
    launch!(architecture(grid), grid, params, _update_transport!, Uⁿ, Uⁿ⁻¹, grid, U)

function update_transport!(Uⁿ, Uⁿ⁻¹, model, grid, params, ts::SplitRungeKuttaTimeStepper, stage, U)
    if stage == ts.Nstages-1
        launch!(architecture(grid), grid, params, _update_transport!, Uⁿ, grid, U)
    end
end

update_transport!(Uⁿ, Uⁿ⁻¹, model::HydrostaticFreeSurfaceModel, grid::SphericalShellGrid, params, ::QuasiAdamsBashforth2TimeStepper, stage, U) =
    launch!(architecture(grid), grid, params, _update_nonorthogonal_transport!, Uⁿ, Uⁿ⁻¹, grid, U)

update_transport!(Uⁿ, Uⁿ⁻¹, model::NonhydrostaticModel{<:Any, <:Any, <:Any, <:SphericalShellGrid}, grid::SphericalShellGrid, params, ::QuasiAdamsBashforth2TimeStepper, stage, U) =
    launch!(architecture(grid), grid, params, _update_nonorthogonal_transport!, Uⁿ, Uⁿ⁻¹, grid, U)

function update_transport!(Uⁿ, Uⁿ⁻¹, model::HydrostaticFreeSurfaceModel, grid::SphericalShellGrid, params, ts::SplitRungeKuttaTimeStepper, stage, U)
    if stage == ts.Nstages-1
        launch!(architecture(grid), grid, params, _update_nonorthogonal_transport!, Uⁿ, grid, U)
    end
end

function update_transport!(Uⁿ, Uⁿ⁻¹, model::NonhydrostaticModel{<:Any, <:Any, <:Any, <:SphericalShellGrid}, grid::SphericalShellGrid, params, ts::SplitRungeKuttaTimeStepper, stage, U)
    if stage == ts.Nstages-1
        launch!(architecture(grid), grid, params, _update_nonorthogonal_transport!, Uⁿ, grid, U)
    end
end

@kernel function _update_transport!(Uⁿ, Uⁿ⁻¹, grid, U)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        u = u_velocity(U)
        v = v_velocity(U)
        w = w_velocity(U)

        Uⁿ⁻¹.u[i, j, k] = Uⁿ.u[i, j, k]
        Uⁿ⁻¹.v[i, j, k] = Uⁿ.v[i, j, k]
        Uⁿ⁻¹.w[i, j, k] = Uⁿ.w[i, j, k]
          Uⁿ.u[i, j, k] = u[i, j, k] * Axᶠᶜᶜ(i, j, k, grid)
          Uⁿ.v[i, j, k] = v[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid)
          Uⁿ.w[i, j, k] = w[i, j, k] * Azᶜᶜᶠ(i, j, k, grid)
    end
end

@kernel function _update_transport!(Uⁿ, grid, U)
    i, j, k = @index(Global, NTuple)

    u = u_velocity(U)
    v = v_velocity(U)
    w = w_velocity(U)

    @inbounds Uⁿ.u[i, j, k] = u[i, j, k] * Axᶠᶜᶜ(i, j, k, grid)
    @inbounds Uⁿ.v[i, j, k] = v[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid)
    @inbounds Uⁿ.w[i, j, k] = w[i, j, k] * Azᶜᶜᶠ(i, j, k, grid)
end

@kernel function _update_nonorthogonal_transport!(Uⁿ, Uⁿ⁻¹, grid, U)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        u = u_velocity(U)
        v = v_velocity(U)
        w = w_velocity(U)

        Uⁿ⁻¹.u[i, j, k] = Uⁿ.u[i, j, k]
        Uⁿ⁻¹.v[i, j, k] = Uⁿ.v[i, j, k]
        Uⁿ⁻¹.w[i, j, k] = Uⁿ.w[i, j, k]
          Uⁿ.u[i, j, k] = u[i, j, k]
          Uⁿ.v[i, j, k] = v[i, j, k]
          Uⁿ.w[i, j, k] = w[i, j, k] * Azᶜᶜᶠ(i, j, k, grid)
    end
end

@kernel function _update_nonorthogonal_transport!(Uⁿ, grid, U)
    i, j, k = @index(Global, NTuple)

    u = u_velocity(U)
    v = v_velocity(U)
    w = w_velocity(U)

    @inbounds begin
        Uⁿ.u[i, j, k] = u[i, j, k]
        Uⁿ.v[i, j, k] = v[i, j, k]
        Uⁿ.w[i, j, k] = w[i, j, k] * Azᶜᶜᶠ(i, j, k, grid)
    end
end
