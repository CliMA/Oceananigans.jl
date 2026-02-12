using Oceananigans.Advection: horizontal_advection_U, horizontal_advection_V,
                               vertical_advection_U, vertical_advection_V,
                               bernoulli_head_U, bernoulli_head_V,
                               VectorInvariantUpwindVorticity,
                               div_Uc

using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ,
                                       immersed_∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₂ⱼ,
                                       ∇_dot_qᶜ, immersed_∇_dot_qᶜ,
                                       closure_auxiliary_velocity

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

using Oceananigans.Biogeochemistry: biogeochemical_transition,
                                     biogeochemical_drift_velocity,
                                     biogeochemical_auxiliary_fields

using Oceananigans.Forcings: with_advective_forcing
using Oceananigans.Utils: sum_of_velocities

#####
##### Split sub-kernels for u-velocity
#####

@kernel function _compute_u_horizontal_advection!(Gu, grid, scheme, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = - horizontal_advection_U(i, j, k, grid, scheme, u, v)
end

@kernel function _compute_u_vertical_advection!(Gu, grid, scheme, velocities)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] -= vertical_advection_U(i, j, k, grid, scheme, velocities)
end

@kernel function _compute_u_bernoulli_head!(Gu, grid, scheme, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] -= bernoulli_head_U(i, j, k, grid, scheme, u, v)
end

@kernel function _compute_u_forcing_diffusion!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] += hydrostatic_free_surface_u_forcing_diffusion(i, j, k, grid, args...)
end

#####
##### Split sub-kernels for v-velocity
#####

@kernel function _compute_v_horizontal_advection!(Gv, grid, scheme, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = - horizontal_advection_V(i, j, k, grid, scheme, u, v)
end

@kernel function _compute_v_vertical_advection!(Gv, grid, scheme, velocities)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] -= vertical_advection_V(i, j, k, grid, scheme, velocities)
end

@kernel function _compute_v_bernoulli_head!(Gv, grid, scheme, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] -= bernoulli_head_V(i, j, k, grid, scheme, u, v)
end

@kernel function _compute_v_forcing_diffusion!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] += hydrostatic_free_surface_v_forcing_diffusion(i, j, k, grid, args...)
end

#####
##### Split sub-kernels for tracers
#####

@kernel function _compute_tracer_advection!(Gc, grid, advection, total_velocities, tracer)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = - div_Uc(i, j, k, grid, advection, total_velocities, tracer)
end

@kernel function _compute_tracer_forcing_diffusion!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] += hydrostatic_free_surface_tracer_forcing_diffusion(i, j, k, grid, args...)
end

#####
##### Non-advection tendency functions
#####

@inline function hydrostatic_free_surface_u_forcing_diffusion(i, j, k, grid,
                                                          coriolis, closure, u_immersed_bc,
                                                          velocities, free_surface, tracers,
                                                          buoyancy, diffusivities,
                                                          hydrostatic_pressure_anomaly,
                                                          auxiliary_fields, ztype, clock, forcing)

    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers), auxiliary_fields)

    return ( - explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface)
             - x_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂xᶠᶜᶜ(i, j, k, grid, hydrostatic_pressure_anomaly)
             - grid_slope_contribution_x(i, j, k, grid, buoyancy, ztype, model_fields)
             - ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy)
             - immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid, velocities, u_immersed_bc, closure, diffusivities, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end

@inline function hydrostatic_free_surface_v_forcing_diffusion(i, j, k, grid,
                                                          coriolis, closure, v_immersed_bc,
                                                          velocities, free_surface, tracers,
                                                          buoyancy, diffusivities,
                                                          hydrostatic_pressure_anomaly,
                                                          auxiliary_fields, ztype, clock, forcing)

    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers), auxiliary_fields)

    return ( - explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface)
             - y_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂yᶜᶠᶜ(i, j, k, grid, hydrostatic_pressure_anomaly)
             - grid_slope_contribution_y(i, j, k, grid, buoyancy, ztype, model_fields)
             - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy)
             - immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end

@inline function hydrostatic_free_surface_tracer_forcing_diffusion(i, j, k, grid,
                                                               val_tracer_index::Val{tracer_index},
                                                               val_tracer_name,
                                                               closure, c_immersed_bc,
                                                               buoyancy, biogeochemistry,
                                                               velocities, free_surface, tracers,
                                                               diffusivities, auxiliary_fields,
                                                               clock, forcing) where tracer_index

    @inbounds c = tracers[tracer_index]
    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers),
                         auxiliary_fields,
                         biogeochemical_auxiliary_fields(biogeochemistry))

    return ( - ∇_dot_qᶜ(i, j, k, grid, closure, diffusivities, val_tracer_index, c, clock, model_fields, buoyancy)
             - immersed_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, diffusivities, val_tracer_index, clock, model_fields)
             + biogeochemical_transition(i, j, k, grid, biogeochemistry, val_tracer_name, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end

#####
##### Split dispatch for VectorInvariant upwind vorticity schemes
#####

function compute_hydrostatic_momentum_tendencies!(advection::VectorInvariantUpwindVorticity,
                                                  model, velocities, kernel_parameters;
                                                  active_cells_map=nothing)
    grid = model.grid
    arch = architecture(grid)

    u_immersed_bc = immersed_boundary_condition(velocities.u)
    v_immersed_bc = immersed_boundary_condition(velocities.v)

    scheme = advection

    ####
    #### U-velocity: 4 sub-kernels
    ####

    # Advection sub-kernels: split interior/boundary for register pressure reduction
    split_advection_launch!(arch, grid, kernel_parameters, active_cells_map,
        _compute_u_horizontal_advection!, model.timestepper.Gⁿ.u, scheme, velocities.u, velocities.v)

    split_advection_launch!(arch, grid, kernel_parameters, active_cells_map,
        _compute_u_vertical_advection!, model.timestepper.Gⁿ.u, scheme, velocities)

    split_advection_launch!(arch, grid, kernel_parameters, active_cells_map,
        _compute_u_bernoulli_head!, model.timestepper.Gⁿ.u, scheme, velocities.u, velocities.v)

    u_forcing_diffusion_args = (model.coriolis, model.closure, u_immersed_bc,
                           velocities, model.free_surface, model.tracers,
                           model.buoyancy, model.closure_fields,
                           model.pressure.pHY′, model.auxiliary_fields,
                           model.vertical_coordinate, model.clock, model.forcing.u)

    launch!(arch, grid, kernel_parameters,
            _compute_u_forcing_diffusion!, model.timestepper.Gⁿ.u, grid,
            u_forcing_diffusion_args;
            active_cells_map)

    ####
    #### U-velocity: 4 sub-kernels
    ####

    split_advection_launch!(arch, grid, kernel_parameters, active_cells_map,
        _compute_v_horizontal_advection!, model.timestepper.Gⁿ.v, scheme, velocities.u, velocities.v)

    split_advection_launch!(arch, grid, kernel_parameters, active_cells_map,
        _compute_v_vertical_advection!, model.timestepper.Gⁿ.v, scheme, velocities)

    split_advection_launch!(arch, grid, kernel_parameters, active_cells_map,
        _compute_v_bernoulli_head!, model.timestepper.Gⁿ.v, scheme, velocities.u, velocities.v)

    v_forcing_diffusion_args = (model.coriolis, model.closure, v_immersed_bc,
                           velocities, model.free_surface, model.tracers,
                           model.buoyancy, model.closure_fields,
                           model.pressure.pHY′, model.auxiliary_fields,
                           model.vertical_coordinate, model.clock, model.forcing.v)

    launch!(arch, grid, kernel_parameters,
            _compute_v_forcing_diffusion!, model.timestepper.Gⁿ.v, grid,
            v_forcing_diffusion_args;
            active_cells_map)

    return nothing
end

#####
##### Split dispatch for tracer tendencies
#####

function compute_hydrostatic_tracer_tendencies!(model, kernel_parameters; active_cells_map=nothing)

    arch = model.architecture
    grid = model.grid

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))

        @inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection   = model.advection[tracer_name]
        @inbounds c_forcing     = model.forcing[tracer_name]
        @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])

        # Compute total_velocities (same logic as the monolithic kernel)
        biogeochemical_velocities = biogeochemical_drift_velocity(model.biogeochemistry, Val(tracer_name))
        closure_velocities = closure_auxiliary_velocity(model.closure, model.closure_fields, Val(tracer_name))
        total_velocities = sum_of_velocities(model.transport_velocities, biogeochemical_velocities, closure_velocities)
        total_velocities = with_advective_forcing(c_forcing, total_velocities)

        # Sub-kernel 1: Advection (split interior/boundary on IBG)
        split_advection_launch!(arch, grid, kernel_parameters, active_cells_map,
            _compute_tracer_advection!, c_tendency,
            c_advection, total_velocities, model.tracers[tracer_name])

        # Sub-kernel 2: Non-advection (single launch, no split needed)
        forcing_diffusion_args = (Val(tracer_index), Val(tracer_name),
                                  model.closure, c_immersed_bc, model.buoyancy,
                                  model.biogeochemistry, model.transport_velocities,
                                  model.free_surface, model.tracers, model.closure_fields,
                                  model.auxiliary_fields, model.clock, c_forcing)

        launch!(arch, grid, kernel_parameters,
                _compute_tracer_forcing_diffusion!, c_tendency, grid,
                forcing_diffusion_args;
                active_cells_map)
    end

    return nothing
end
