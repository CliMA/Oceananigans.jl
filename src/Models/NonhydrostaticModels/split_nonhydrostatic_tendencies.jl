using Oceananigans.Advection: AbstractUpwindBiasedAdvectionScheme, div_𝐯u, div_𝐯v, div_𝐯w, div_Uc
using Oceananigans.Biogeochemistry: biogeochemical_transition, biogeochemical_drift_velocity
using Oceananigans.Forcings: with_advective_forcing

#####
##### Split sub-kernels for u-velocity
#####

@kernel function _compute_u_advection!(Gu, grid, advection, total_velocities, velocities, background_u)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = - div_𝐯u(i, j, k, grid, advection, total_velocities, velocities.u) -
                              div_𝐯u(i, j, k, grid, advection, velocities, background_u)
end

@kernel function _compute_u_forcing_diffusion!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] += nonhydrostatic_u_forcing_diffusion(i, j, k, grid, args...)
end

@inline function nonhydrostatic_u_forcing_diffusion(i, j, k, grid,
                                                coriolis, stokes_drift, closure,
                                                u_immersed_bc, buoyancy,
                                                background_fields, velocities,
                                                tracers, auxiliary_fields, diffusivities,
                                                hydrostatic_pressure, clock, forcing)

    closure_velocities = assemble_closure_velocities(velocities, background_fields)
    closure_model_fields = merge(closure_velocities, tracers, auxiliary_fields)
    model_fields = merge(velocities, tracers, auxiliary_fields)

    return ( + x_dot_g_bᶠᶜᶜ(i, j, k, grid, buoyancy, tracers)
             - x_f_cross_U(i, j, k, grid, coriolis, velocities)
             - hydrostatic_pressure_gradient_x(i, j, k, grid, hydrostatic_pressure)
             - ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, diffusivities, clock, closure_model_fields, buoyancy)
             - immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid, velocities, u_immersed_bc, closure, diffusivities, clock, model_fields)
             + x_curl_Uˢ_cross_U(i, j, k, grid, stokes_drift, velocities, clock.time)
             + ∂t_uˢ(i, j, k, grid, stokes_drift, clock.time)
             + forcing(i, j, k, grid, clock, model_fields))
end

#####
##### Split sub-kernels for v-velocity
#####

@kernel function _compute_v_advection!(Gv, grid, advection, total_velocities, velocities, background_v)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = - div_𝐯v(i, j, k, grid, advection, total_velocities, velocities.v) -
                              div_𝐯v(i, j, k, grid, advection, velocities, background_v)
end

@kernel function _compute_v_forcing_diffusion!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] += nonhydrostatic_v_forcing_diffusion(i, j, k, grid, args...)
end

@inline function nonhydrostatic_v_forcing_diffusion(i, j, k, grid,
                                                coriolis, stokes_drift, closure,
                                                v_immersed_bc, buoyancy,
                                                background_fields, velocities,
                                                tracers, auxiliary_fields, diffusivities,
                                                hydrostatic_pressure, clock, forcing)

    closure_velocities = assemble_closure_velocities(velocities, background_fields)
    closure_model_fields = merge(closure_velocities, tracers, auxiliary_fields)
    model_fields = merge(velocities, tracers, auxiliary_fields)

    return ( + y_dot_g_bᶜᶠᶜ(i, j, k, grid, buoyancy, tracers)
             - y_f_cross_U(i, j, k, grid, coriolis, velocities)
             - hydrostatic_pressure_gradient_y(i, j, k, grid, hydrostatic_pressure)
             - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, diffusivities, clock, closure_model_fields, buoyancy)
             - immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields)
             + y_curl_Uˢ_cross_U(i, j, k, grid, stokes_drift, velocities, clock.time)
             + ∂t_vˢ(i, j, k, grid, stokes_drift, clock.time)
             + forcing(i, j, k, grid, clock, model_fields))
end

#####
##### Split sub-kernels for w-velocity
#####

@kernel function _compute_w_advection!(Gw, grid, advection, total_velocities, velocities, background_w)
    i, j, k = @index(Global, NTuple)
    @inbounds Gw[i, j, k] = - div_𝐯w(i, j, k, grid, advection, total_velocities, velocities.w) -
                               div_𝐯w(i, j, k, grid, advection, velocities, background_w)
end

@kernel function _compute_w_forcing_diffusion!(Gw, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gw[i, j, k] += nonhydrostatic_w_forcing_diffusion(i, j, k, grid, args...)
end

@inline function nonhydrostatic_w_forcing_diffusion(i, j, k, grid,
                                                coriolis, stokes_drift, closure,
                                                w_immersed_bc, buoyancy,
                                                background_fields, velocities,
                                                tracers, auxiliary_fields, diffusivities,
                                                hydrostatic_pressure, clock, forcing)

    closure_velocities = assemble_closure_velocities(velocities, background_fields)
    closure_model_fields = merge(closure_velocities, tracers, auxiliary_fields)
    model_fields = merge(velocities, tracers, auxiliary_fields)

    return ( + maybe_z_dot_g_bᶜᶜᶠ(i, j, k, grid, hydrostatic_pressure, buoyancy, tracers)
             - z_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂ⱼ_τ₃ⱼ(i, j, k, grid, closure, diffusivities, clock, closure_model_fields, buoyancy)
             - immersed_∂ⱼ_τ₃ⱼ(i, j, k, grid, velocities, w_immersed_bc, closure, diffusivities, clock, model_fields)
             + z_curl_Uˢ_cross_U(i, j, k, grid, stokes_drift, velocities, clock.time)
             + ∂t_wˢ(i, j, k, grid, stokes_drift, clock.time)
             + forcing(i, j, k, grid, clock, model_fields))
end

#####
##### Split sub-kernels for tracers
#####

@kernel function _compute_nonhydrostatic_tracer_advection!(Gc, grid, advection, total_velocities, c, background_c)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = - div_Uc(i, j, k, grid, advection, total_velocities, c) -
                               div_Uc(i, j, k, grid, advection, total_velocities, background_c)
end

@kernel function _compute_nonhydrostatic_tracer_forcing_diffusion!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] += nonhydrostatic_tracer_forcing_diffusion(i, j, k, grid, args...)
end

@inline function nonhydrostatic_tracer_forcing_diffusion(i, j, k, grid,
                                                     val_index::Val{tracer_index},
                                                     val_tracer_name,
                                                     closure, c_immersed_bc,
                                                     buoyancy, biogeochemistry,
                                                     background_fields, velocities,
                                                     tracers, auxiliary_fields,
                                                     diffusivities, clock, forcing) where tracer_index

    @inbounds c = tracers[tracer_index]
    @inbounds background_fields_c = background_fields.tracers[tracer_index]

    closure_c = if background_fields isa BackgroundFieldsWithClosureFluxes
        sum_fields(c, background_fields_c)
    else
        c
    end

    closure_velocities = assemble_closure_velocities(velocities, background_fields)
    closure_model_fields = merge(closure_velocities, tracers, auxiliary_fields)
    model_fields = merge(velocities, tracers, auxiliary_fields)

    return ( - ∇_dot_qᶜ(i, j, k, grid, closure, diffusivities, val_index, closure_c, clock, closure_model_fields, buoyancy)
             - immersed_∇_dot_qᶜ(i, j, k, grid, closure_c, c_immersed_bc, closure, diffusivities, val_index, clock, model_fields)
             + biogeochemical_transition(i, j, k, grid, biogeochemistry, val_tracer_name, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end

#####
##### Split dispatch for upwind advection schemes
#####

function compute_interior_tendency_contributions!(model,
                                                  advection::AbstractUpwindBiasedAdvectionScheme,
                                                  kernel_parameters;
                                                  active_cells_map = nothing)

    tendencies           = model.timestepper.Gⁿ
    arch                 = model.architecture
    grid                 = model.grid
    coriolis             = model.coriolis
    buoyancy             = model.buoyancy
    biogeochemistry      = model.biogeochemistry
    stokes_drift         = model.stokes_drift
    closure              = model.closure
    background_fields    = model.background_fields
    velocities           = model.velocities
    tracers              = model.tracers
    auxiliary_fields     = model.auxiliary_fields
    hydrostatic_pressure = model.pressures.pHY′
    diffusivities        = model.closure_fields
    forcings             = model.forcing
    clock                = model.clock
    u_immersed_bc        = velocities.u.boundary_conditions.immersed
    v_immersed_bc        = velocities.v.boundary_conditions.immersed
    w_immersed_bc        = velocities.w.boundary_conditions.immersed

    total_velocities = sum_of_velocities(velocities, background_fields.velocities)
    total_velocities = with_advective_forcing(forcings.u, total_velocities)

    # --- U-velocity: advection (split) + forcing_diffusion ---

    split_advection_launch!(arch, grid, kernel_parameters, active_cells_map,
        _compute_u_advection!, tendencies.u,
        advection, total_velocities, velocities, background_fields.velocities.u;
        exclude_periphery=true)

    u_forcing_diffusion_args = (coriolis, stokes_drift, closure,
                           u_immersed_bc, buoyancy,
                           background_fields, velocities,
                           tracers, auxiliary_fields, diffusivities,
                           hydrostatic_pressure, clock, forcings.u)

    launch!(arch, grid, kernel_parameters,
            _compute_u_forcing_diffusion!, tendencies.u, grid,
            u_forcing_diffusion_args;
            active_cells_map, exclude_periphery=true)

    # --- V-velocity: advection (split) + forcing_diffusion ---

    total_velocities_v = sum_of_velocities(velocities, background_fields.velocities)
    total_velocities_v = with_advective_forcing(forcings.v, total_velocities_v)

    split_advection_launch!(arch, grid, kernel_parameters, active_cells_map,
        _compute_v_advection!, tendencies.v,
        advection, total_velocities_v, velocities, background_fields.velocities.v;
        exclude_periphery=true)

    v_forcing_diffusion_args = (coriolis, stokes_drift, closure,
                           v_immersed_bc, buoyancy,
                           background_fields, velocities,
                           tracers, auxiliary_fields, diffusivities,
                           hydrostatic_pressure, clock, forcings.v)

    launch!(arch, grid, kernel_parameters,
            _compute_v_forcing_diffusion!, tendencies.v, grid,
            v_forcing_diffusion_args;
            active_cells_map, exclude_periphery=true)

    # --- W-velocity: advection (split) + forcing_diffusion ---

    total_velocities_w = sum_of_velocities(velocities, background_fields.velocities)
    total_velocities_w = with_advective_forcing(forcings.w, total_velocities_w)

    split_advection_launch!(arch, grid, kernel_parameters, active_cells_map,
        _compute_w_advection!, tendencies.w,
        advection, total_velocities_w, velocities, background_fields.velocities.w;
        exclude_periphery=true)

    w_forcing_diffusion_args = (coriolis, stokes_drift, closure,
                           w_immersed_bc, buoyancy,
                           background_fields, velocities,
                           tracers, auxiliary_fields, diffusivities,
                           hydrostatic_pressure, clock, forcings.w)

    launch!(arch, grid, kernel_parameters,
            _compute_w_forcing_diffusion!, tendencies.w, grid,
            w_forcing_diffusion_args;
            active_cells_map, exclude_periphery=true)

    # --- Tracers: advection (split) + forcing_diffusion ---

    for tracer_index in 1:length(tracers)
        @inbounds c_tendency = tendencies[tracer_index + 3]
        @inbounds forcing = forcings[tracer_index + 3]
        @inbounds c_immersed_bc = tracers[tracer_index].boundary_conditions.immersed
        @inbounds tracer_name = keys(tracers)[tracer_index]

        @inbounds c = tracers[tracer_index]
        @inbounds background_c = background_fields.tracers[tracer_index]

        biogeochemical_velocities = biogeochemical_drift_velocity(biogeochemistry, Val(tracer_name))
        total_velocities_c = sum_of_velocities(velocities, background_fields.velocities, biogeochemical_velocities)
        total_velocities_c = with_advective_forcing(forcing, total_velocities_c)

        split_advection_launch!(arch, grid, kernel_parameters, active_cells_map,
            _compute_nonhydrostatic_tracer_advection!, c_tendency,
            advection, total_velocities_c, c, background_c)

        forcing_diffusion_args = (Val(tracer_index), Val(tracer_name),
                             closure, c_immersed_bc, buoyancy, biogeochemistry,
                             background_fields, velocities,
                             tracers, auxiliary_fields, diffusivities,
                             clock, forcing)

        launch!(arch, grid, kernel_parameters,
                _compute_nonhydrostatic_tracer_forcing_diffusion!, c_tendency, grid,
                forcing_diffusion_args;
                active_cells_map)
    end

    return nothing
end
