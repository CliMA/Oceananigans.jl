using Oceananigans.Advection: horizontal_advection_U, horizontal_advection_V,
                               bernoulli_head_U, bernoulli_head_V,
                               upwinded_divergence_flux_Uᶠᶜᶜ, upwinded_divergence_flux_Vᶜᶠᶜ,
                               _advective_momentum_flux_Wu, _advective_momentum_flux_Wv,
                               VectorInvariantUpwindVorticity

using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, δzᵃᵃᶜ, Vᶠᶜᶜ, Vᶜᶠᶜ
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ,
                                       immersed_∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₂ⱼ

#####
##### Split-kernel momentum tendency computation for VectorInvariantUpwindVorticity schemes.
#####
##### When using WENO-based VectorInvariant advection, the single momentum tendency kernel
##### uses ~209 registers (3+ independent WENO interpolations inlined together), limiting
##### GPU occupancy to ~12.5%. By splitting into separate kernels (one per WENO interpolation
##### + non-advection remainder), each kernel uses ~64-128 registers, improving occupancy to 25-50%.
#####
##### The vertical advection is further split into divergence flux and vertical flux sub-kernels
##### to keep each kernel at a single WENO interpolation, avoiding the need for maxregs capping.
#####

#####
##### Split @kernel functions for u-velocity tendency
#####

@kernel function _compute_u_horizontal_advection!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    scheme, u, v = args
    @inbounds Gu[i, j, k] = -horizontal_advection_U(i, j, k, grid, scheme, u, v)
end

@kernel function _accumulate_u_divergence_flux!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    scheme, u, v = args
    @inbounds Gu[i, j, k] -= upwinded_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme, u, v) / Vᶠᶜᶜ(i, j, k, grid)
end

@kernel function _accumulate_u_vertical_flux!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    vertical_scheme, w, u = args
    @inbounds Gu[i, j, k] -= δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, vertical_scheme, w, u) / Vᶠᶜᶜ(i, j, k, grid)
end

@kernel function _accumulate_u_bernoulli_head!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    scheme, u, v = args
    @inbounds Gu[i, j, k] -= bernoulli_head_U(i, j, k, grid, scheme, u, v)
end

@kernel function _accumulate_u_nonadvection!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] += hydrostatic_free_surface_u_velocity_tendency_nonadvection(i, j, k, grid, args...)
end

#####
##### Split @kernel functions for v-velocity tendency
#####

@kernel function _compute_v_horizontal_advection!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    scheme, u, v = args
    @inbounds Gv[i, j, k] = -horizontal_advection_V(i, j, k, grid, scheme, u, v)
end

@kernel function _accumulate_v_divergence_flux!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    scheme, u, v = args
    @inbounds Gv[i, j, k] -= upwinded_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme, u, v) / Vᶜᶠᶜ(i, j, k, grid)
end

@kernel function _accumulate_v_vertical_flux!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    vertical_scheme, w, v = args
    @inbounds Gv[i, j, k] -= δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, vertical_scheme, w, v) / Vᶜᶠᶜ(i, j, k, grid)
end

@kernel function _accumulate_v_bernoulli_head!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    scheme, u, v = args
    @inbounds Gv[i, j, k] -= bernoulli_head_V(i, j, k, grid, scheme, u, v)
end

@kernel function _accumulate_v_nonadvection!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] += hydrostatic_free_surface_v_velocity_tendency_nonadvection(i, j, k, grid, args...)
end

#####
##### Non-advection tendency functions (everything except U_dot_∇u / U_dot_∇v)
#####

@inline function hydrostatic_free_surface_u_velocity_tendency_nonadvection(i, j, k, grid,
                                                                           coriolis,
                                                                           closure,
                                                                           u_immersed_bc,
                                                                           velocities,
                                                                           free_surface,
                                                                           tracers,
                                                                           buoyancy,
                                                                           diffusivities,
                                                                           hydrostatic_pressure_anomaly,
                                                                           auxiliary_fields,
                                                                           ztype,
                                                                           clock,
                                                                           forcing)

    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers), auxiliary_fields)

    return ( - explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface)
             - x_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂xᶠᶜᶜ(i, j, k, grid, hydrostatic_pressure_anomaly)
             - grid_slope_contribution_x(i, j, k, grid, buoyancy, ztype, model_fields)
             - ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy)
             - immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid, velocities, u_immersed_bc, closure, diffusivities, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end

@inline function hydrostatic_free_surface_v_velocity_tendency_nonadvection(i, j, k, grid,
                                                                           coriolis,
                                                                           closure,
                                                                           v_immersed_bc,
                                                                           velocities,
                                                                           free_surface,
                                                                           tracers,
                                                                           buoyancy,
                                                                           diffusivities,
                                                                           hydrostatic_pressure_anomaly,
                                                                           auxiliary_fields,
                                                                           ztype,
                                                                           clock,
                                                                           forcing)

    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers), auxiliary_fields)

    return ( - explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface)
             - y_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂yᶜᶠᶜ(i, j, k, grid, hydrostatic_pressure_anomaly)
             - grid_slope_contribution_y(i, j, k, grid, buoyancy, ztype, model_fields)
             - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy)
             - immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end

#####
##### Split-kernel dispatch for VectorInvariantUpwindVorticity (includes WENOVectorInvariant)
#####

function _compute_hydrostatic_momentum_tendencies!(advection::VectorInvariantUpwindVorticity,
                                                   model, velocities, kernel_parameters;
                                                   active_cells_map=nothing)
    grid = model.grid
    arch = architecture(grid)

    Gu = model.timestepper.Gⁿ.u
    Gv = model.timestepper.Gⁿ.v

    u = velocities.u
    v = velocities.v
    w = velocities.w

    u_immersed_bc = immersed_boundary_condition(velocities.u)
    v_immersed_bc = immersed_boundary_condition(velocities.v)

    # --- Kernel 1: Horizontal advection (= assignment, initializes tendency) ---
    # Uses WENO{5} vorticity interpolation
    horizontal_advection_args = (advection, u, v)

    launch!(arch, grid, kernel_parameters,
            _compute_u_horizontal_advection!, Gu, grid,
            horizontal_advection_args; active_cells_map)

    launch!(arch, grid, kernel_parameters,
            _compute_v_horizontal_advection!, Gv, grid,
            horizontal_advection_args; active_cells_map)

    # --- Kernel 2a: Divergence flux (-= accumulate) ---
    # Uses WENO{3} divergence interpolation
    divergence_flux_args = (advection, u, v)

    launch!(arch, grid, kernel_parameters,
            _accumulate_u_divergence_flux!, Gu, grid,
            divergence_flux_args; active_cells_map)

    launch!(arch, grid, kernel_parameters,
            _accumulate_v_divergence_flux!, Gv, grid,
            divergence_flux_args; active_cells_map)

    # --- Kernel 2b: Vertical advective flux (-= accumulate) ---
    # Uses WENO vertical interpolation
    vertical_scheme = advection.vertical_advection_scheme
    u_vertical_flux_args = (vertical_scheme, w, u)
    v_vertical_flux_args = (vertical_scheme, w, v)

    launch!(arch, grid, kernel_parameters,
            _accumulate_u_vertical_flux!, Gu, grid,
            u_vertical_flux_args; active_cells_map)

    launch!(arch, grid, kernel_parameters,
            _accumulate_v_vertical_flux!, Gv, grid,
            v_vertical_flux_args; active_cells_map)

    # --- Kernel 3: Bernoulli head (-= accumulate) ---
    # Uses WENO{3} KE interpolation
    bernoulli_args = (advection, u, v)

    launch!(arch, grid, kernel_parameters,
            _accumulate_u_bernoulli_head!, Gu, grid,
            bernoulli_args; active_cells_map)

    launch!(arch, grid, kernel_parameters,
            _accumulate_v_bernoulli_head!, Gv, grid,
            bernoulli_args; active_cells_map)

    # --- Kernel 4: Non-advection terms (+= accumulate) ---
    # No WENO — lightweight
    nonadvection_args_u = (model.coriolis,
                           model.closure,
                           u_immersed_bc,
                           velocities,
                           model.free_surface,
                           model.tracers,
                           model.buoyancy,
                           model.closure_fields,
                           model.pressure.pHY′,
                           model.auxiliary_fields,
                           model.vertical_coordinate,
                           model.clock,
                           model.forcing.u)

    nonadvection_args_v = (model.coriolis,
                           model.closure,
                           v_immersed_bc,
                           velocities,
                           model.free_surface,
                           model.tracers,
                           model.buoyancy,
                           model.closure_fields,
                           model.pressure.pHY′,
                           model.auxiliary_fields,
                           model.vertical_coordinate,
                           model.clock,
                           model.forcing.v)

    launch!(arch, grid, kernel_parameters,
            _accumulate_u_nonadvection!, Gu, grid,
            nonadvection_args_u; active_cells_map)

    launch!(arch, grid, kernel_parameters,
            _accumulate_v_nonadvection!, Gv, grid,
            nonadvection_args_v; active_cells_map)

    return nothing
end
