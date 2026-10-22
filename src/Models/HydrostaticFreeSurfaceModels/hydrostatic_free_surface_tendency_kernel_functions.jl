using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v, VectorInvariant,
                              horizontal_advection_U, horizontal_advection_V, bernoulli_head_U, bernoulli_head_V,
                              vertical_advection_U, vertical_advection_V,
                              U_dot_∇u_hydrostatic_metric, U_dot_∇v_hydrostatic_metric
using Oceananigans.Biogeochemistry: biogeochemical_transition,
                                    biogeochemical_drift_velocity,
                                    include_biogeochemistry_transitions
using Oceananigans.Forcings: with_advective_forcing
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∇_dot_qᶜ,
                                       boundary_∂ⱼ_τ₁ⱼ, boundary_∂ⱼ_τ₂ⱼ, boundary_∇_dot_qᶜ,
                                       closure_auxiliary_velocity
using Oceananigans.Utils: sum_of_velocities

@inline momentum_advection_and_coriolis_u(i, j, k, grid, advection, coriolis, U) = U_dot_∇u(i, j, k, grid, advection, U) + x_f_cross_U(i, j, k, grid, coriolis, U)
@inline momentum_advection_and_coriolis_v(i, j, k, grid, advection, coriolis, U) = U_dot_∇v(i, j, k, grid, advection, U) + y_f_cross_U(i, j, k, grid, coriolis, U)

# Coriolis right after the vorticity term reuses its velocity interpolations before the kinetic energy and vertical terms
@inline function momentum_advection_and_coriolis_u(i, j, k, grid, advection::VectorInvariant, coriolis, U)
    𝒜ʰ = horizontal_advection_U(i, j, k, grid, advection, U.u, U.v)
    ℱ  = x_f_cross_U(i, j, k, grid, coriolis, U)
    ℬ  = bernoulli_head_U(i, j, k, grid, advection, U.u, U.v)
    𝒜ᶻ = vertical_advection_U(i, j, k, grid, advection, U)
    return 𝒜ʰ + 𝒜ᶻ + ℬ + ℱ
end

@inline function momentum_advection_and_coriolis_v(i, j, k, grid, advection::VectorInvariant, coriolis, U)
    𝒜ʰ = horizontal_advection_V(i, j, k, grid, advection, U.u, U.v)
    ℱ  = y_f_cross_U(i, j, k, grid, coriolis, U)
    ℬ  = bernoulli_head_V(i, j, k, grid, advection, U.u, U.v)
    𝒜ᶻ = vertical_advection_V(i, j, k, grid, advection, U)
    return 𝒜ʰ + 𝒜ᶻ + ℬ + ℱ
end

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
                                                              u_bcs,
                                                              velocities,
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

    # Note: For mutable grids (z-star), the chain-rule correction for grid slope
    # is automatically included in ∂xᶠᶜᶜ, so no explicit grid_slope_contribution is needed.
    return ( - momentum_advection_and_coriolis_u(i, j, k, grid, advection, coriolis, velocities)
             - U_dot_∇u_hydrostatic_metric(i, j, k, grid, advection, velocities, velocities)
             - explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface)
             - ∂xᶠᶜᶜ(i, j, k, grid, hydrostatic_pressure_anomaly)
             - ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, closure_fields, clock, model_fields, buoyancy)
             - boundary_∂ⱼ_τ₁ⱼ(i, j, k, grid, velocities.u, u_bcs, closure, closure_fields, clock, model_fields, buoyancy)
             + forcing(i, j, k, grid, clock, model_fields))
end

"""
Return the tendency for the horizontal velocity in the ``y``-direction, or the north-south
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
                                                              v_bcs,
                                                              velocities,
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

    # Note: For mutable grids (z-star), the chain-rule correction for grid slope
    # is automatically included in ∂yᶜᶠᶜ, so no explicit grid_slope_contribution is needed.
    return ( - momentum_advection_and_coriolis_v(i, j, k, grid, advection, coriolis, velocities)
             - U_dot_∇v_hydrostatic_metric(i, j, k, grid, advection, velocities, velocities)
             - explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface)
             - ∂yᶜᶠᶜ(i, j, k, grid, hydrostatic_pressure_anomaly)
             - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, closure_fields, clock, model_fields, buoyancy)
             - boundary_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities.v, v_bcs, closure, closure_fields, clock, model_fields, buoyancy)
             + forcing(i, j, k, grid, clock, model_fields))
end

"""
$(TYPEDSIGNATURES)

Return the velocities that advect the tracer `val_tracer_name`: the transport `velocities` plus the
biogeochemical drift velocity, the closure's auxiliary velocity and any advective forcing.
"""
@inline function tracer_advecting_velocities(velocities, biogeochemistry, closure, closure_fields, forcing, val_tracer_name)
    biogeochemical_velocities = biogeochemical_drift_velocity(biogeochemistry, val_tracer_name)
    closure_velocities = closure_auxiliary_velocity(closure, closure_fields, val_tracer_name)
    total_velocities = sum_of_velocities(velocities, biogeochemical_velocities, closure_velocities)
    return with_advective_forcing(forcing, total_velocities)
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
                                                          c_bcs,
                                                          buoyancy,
                                                          biogeochemistry,
                                                          velocities,
                                                          free_surface,
                                                          tracers,
                                                          closure_fields,
                                                          auxiliary_fields,
                                                          clock,
                                                          forcing) where tracer_index

    @inbounds c = tracers[tracer_index]
    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers),
                         auxiliary_fields,
                         biogeochemical_auxiliary_fields(biogeochemistry))

    total_velocities = tracer_advecting_velocities(velocities, biogeochemistry, closure, closure_fields, forcing, val_tracer_name)

    bgc_tendency = include_biogeochemistry_transitions(biogeochemistry, val_tracer_name) ?
                   biogeochemical_transition(i, j, k, grid, biogeochemistry, val_tracer_name, clock, model_fields) :
                   zero(grid)

    return ( - div_Uc(i, j, k, grid, advection, total_velocities, c)
             - ∇_dot_qᶜ(i, j, k, grid, closure, closure_fields, val_tracer_index, c, clock, model_fields, buoyancy)
             - boundary_∇_dot_qᶜ(i, j, k, grid, c, c_bcs, closure, closure_fields, val_tracer_index, clock, model_fields, buoyancy)
             + bgc_tendency
             + forcing(i, j, k, grid, clock, model_fields))
end
