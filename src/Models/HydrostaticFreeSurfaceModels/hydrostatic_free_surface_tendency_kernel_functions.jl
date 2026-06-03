using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v,
                              U_dot_∇u_hydrostatic_metric, U_dot_∇v_hydrostatic_metric
using Oceananigans.Advection: VectorInvariant
using Oceananigans.Biogeochemistry: biogeochemical_transition, biogeochemical_drift_velocity
using Oceananigans.Grids: OctaHEALPixMapping, SphericalShellGrid
using Oceananigans.Forcings: with_advective_forcing
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∇_dot_qᶜ,
                                       immersed_∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₂ⱼ, immersed_∇_dot_qᶜ,
                                       closure_auxiliary_velocity
using Oceananigans.Utils: sum_of_velocities

const OHPSG = SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:OctaHEALPixMapping}

"""
Return the tendency for the horizontal velocity in the ``x``-direction, or the east-west
direction, ``u``, at grid point `i, j, k` for a `HydrostaticFreeSurfaceModel`.

The tendency for ``u`` is called ``G_u`` and defined via

```
∂_t u = G_u - ∂_x p_n
```

where `p_n` is the part of the barotropic kinematic pressure that's treated
implicitly during time-stepping.
"""
@inline function hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid,
                                                              advection,
                                                              coriolis,
                                                              closure,
                                                              u_immersed_bc,
                                                              velocities,
                                                              transport_velocities,
                                                              free_surface,
                                                              tracers,
                                                              buoyancy,
                                                              closure_fields,
                                                              hydrostatic_pressure_anomaly,
                                                              auxiliary_fields,
                                                              ztype,
                                                              clock,
                                                              forcing)

    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers), auxiliary_fields)
    momentum_advection = hydrostatic_free_surface_momentum_advection_u(i, j, k, grid,
                                                                       advection,
                                                                       velocities,
                                                                       transport_velocities,
                                                                       free_surface,
                                                                       forcing)

    # Note: For mutable grids (z-star), the chain-rule correction for grid slope
    # is automatically included in ∂xᶠᶜᶜ, so no explicit grid_slope_contribution is needed.
    tendency = ( - momentum_advection
                 - explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface)
                 - x_f_cross_U(i, j, k, grid, coriolis, velocities)
                 - hydrostatic_pressure_anomaly_gradient_x(i, j, k, grid, hydrostatic_pressure_anomaly)
                 - ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, closure_fields, clock, model_fields, buoyancy)
                 - immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid, velocities, u_immersed_bc, closure, closure_fields, clock, model_fields)
                 + forcing(i, j, k, grid, clock, model_fields))

    return tendency
end

"""
Return the tendency for the horizontal velocity in the ``y``-direction, or the east-west
direction, ``v``, at grid point `i, j, k` for a `HydrostaticFreeSurfaceModel`.

The tendency for ``v`` is called ``G_v`` and defined via

```
∂_t v = G_v - ∂_y p_n
```

where `p_n` is the part of the barotropic kinematic pressure that's treated
implicitly during time-stepping.
"""
@inline function hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid,
                                                              advection,
                                                              coriolis,
                                                              closure,
                                                              v_immersed_bc,
                                                              velocities,
                                                              transport_velocities,
                                                              free_surface,
                                                              tracers,
                                                              buoyancy,
                                                              closure_fields,
                                                              hydrostatic_pressure_anomaly,
                                                              auxiliary_fields,
                                                              ztype,
                                                              clock,
                                                              forcing)

    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers), auxiliary_fields)
    momentum_advection = hydrostatic_free_surface_momentum_advection_v(i, j, k, grid,
                                                                       advection,
                                                                       velocities,
                                                                       transport_velocities,
                                                                       free_surface,
                                                                       forcing)

    # Note: For mutable grids (z-star), the chain-rule correction for grid slope
    # is automatically included in ∂yᶜᶠᶜ, so no explicit grid_slope_contribution is needed.
    tendency = ( - momentum_advection
                 - explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface)
                 - y_f_cross_U(i, j, k, grid, coriolis, velocities)
                 - hydrostatic_pressure_anomaly_gradient_y(i, j, k, grid, hydrostatic_pressure_anomaly)
                 - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, closure_fields, clock, model_fields, buoyancy)
                 - immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities, v_immersed_bc, closure, closure_fields, clock, model_fields)
                 + forcing(i, j, k, grid, clock, model_fields))

    return tendency
end

@inline function hydrostatic_free_surface_momentum_advection_u(i, j, k, grid,
                                                               advection,
                                                               velocities,
                                                               transport_velocities,
                                                               free_surface,
                                                               forcing)
    total_velocities = with_advective_forcing(forcing, velocities)

    return U_dot_∇u(i, j, k, grid, advection, total_velocities) +
           U_dot_∇u_hydrostatic_metric(i, j, k, grid, advection, total_velocities, velocities)
end

@inline function hydrostatic_free_surface_momentum_advection_v(i, j, k, grid,
                                                               advection,
                                                               velocities,
                                                               transport_velocities,
                                                               free_surface,
                                                               forcing)
    total_velocities = with_advective_forcing(forcing, velocities)

    return U_dot_∇v(i, j, k, grid, advection, total_velocities) +
           U_dot_∇v_hydrostatic_metric(i, j, k, grid, advection, total_velocities, velocities)
end

@inline function hydrostatic_free_surface_momentum_advection_u(i, j, k, grid::SphericalShellGrid,
                                                               advection::VectorInvariant,
                                                               velocities,
                                                               transport_velocities,
                                                               ::Nothing,
                                                               forcing)
    total_velocities = with_advective_forcing(forcing, velocities)
    return U_dot_∇u(i, j, k, grid, advection, total_velocities) +
           U_dot_∇u_hydrostatic_metric(i, j, k, grid, advection, total_velocities, velocities)
end

@inline function hydrostatic_free_surface_momentum_advection_v(i, j, k, grid::SphericalShellGrid,
                                                               advection::VectorInvariant,
                                                               velocities,
                                                               transport_velocities,
                                                               ::Nothing,
                                                               forcing)
    total_velocities = with_advective_forcing(forcing, velocities)
    return U_dot_∇v(i, j, k, grid, advection, total_velocities) +
           U_dot_∇v_hydrostatic_metric(i, j, k, grid, advection, total_velocities, velocities)
end

@inline hydrostatic_pressure_anomaly_gradient_x(i, j, k, grid, pHY′) =
    ∂xᶠᶜᶜ(i, j, k, grid, pHY′)

@inline hydrostatic_pressure_anomaly_gradient_y(i, j, k, grid, pHY′) =
    ∂yᶜᶠᶜ(i, j, k, grid, pHY′)

@inline tracer_auxiliary_velocities(::Nothing, ::Nothing, forcing) =
    with_advective_forcing(forcing, nothing)

@inline tracer_auxiliary_velocities(biogeochemical_velocities, closure_velocities, forcing) =
    with_advective_forcing(forcing, sum_of_velocities(biogeochemical_velocities, closure_velocities))

@inline total_tracer_advection_velocities(grid, transport_velocities, ::Nothing) = transport_velocities
@inline total_tracer_advection_velocities(grid::SphericalShellGrid, transport_velocities, ::Nothing) = transport_velocities

@inline total_tracer_advection_velocities(grid, transport_velocities, auxiliary_velocities) =
    sum_of_velocities(transport_velocities, auxiliary_velocities)

@inline spherical_shell_auxiliary_tracer_transport_velocities(grid::SphericalShellGrid, auxiliary_velocities) =
    Oceananigans.Advection.spherical_shell_volume_flux_velocities(grid, auxiliary_velocities)

@inline function total_tracer_advection_velocities(grid::SphericalShellGrid, transport_velocities, auxiliary_velocities)
    auxiliary_transport_velocities = spherical_shell_auxiliary_tracer_transport_velocities(grid, auxiliary_velocities)
    return sum_of_velocities(transport_velocities, auxiliary_transport_velocities)
end

"""
Return the tendency for a tracer field with index `tracer_index`
at grid point `i, j, k`.

The tendency is called ``G_c`` and defined via

```
∂_t c = G_c
```

where `c = C[tracer_index]`.
"""
@inline function hydrostatic_free_surface_tracer_tendency(i, j, k, grid,
                                                          val_tracer_index::Val{tracer_index},
                                                          val_tracer_name,
                                                          advection,
                                                          closure,
                                                          c_immersed_bc,
                                                          buoyancy,
                                                          biogeochemistry,
                                                          transport_velocities,
                                                          auxiliary_velocities,
                                                          state_velocities,
                                                          free_surface,
                                                          tracers,
                                                          closure_fields,
                                                          auxiliary_fields,
                                                          clock,
                                                          forcing) where tracer_index

    @inbounds c = tracers[tracer_index]
    model_fields = merge(hydrostatic_fields(state_velocities, free_surface, tracers),
                         auxiliary_fields,
                         biogeochemical_auxiliary_fields(biogeochemistry))

    total_velocities = total_tracer_advection_velocities(grid, transport_velocities, auxiliary_velocities)

    return ( - div_Uc(i, j, k, grid, advection, total_velocities, c)
             - ∇_dot_qᶜ(i, j, k, grid, closure, closure_fields, val_tracer_index, c, clock, model_fields, buoyancy)
             - immersed_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, closure_fields, val_tracer_index, clock, model_fields)
             + biogeochemical_transition(i, j, k, grid, biogeochemistry, val_tracer_name, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end
