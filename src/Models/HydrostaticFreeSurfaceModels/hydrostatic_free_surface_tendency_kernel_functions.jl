using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v,
                              U_dot_∇u_hydrostatic_metric, U_dot_∇v_hydrostatic_metric
using Oceananigans.Biogeochemistry: biogeochemical_transition, biogeochemical_drift_velocity
using Oceananigans.Forcings: with_advective_forcing
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂xᵣᶠᶜᶜ, ∂yᵣᶜᶠᶜ, ∂x_zᶠᶜᶜ, ∂y_zᶜᶠᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Grids: MultiEnvelopeGrid
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∇_dot_qᶜ,
                                       immersed_∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₂ⱼ, immersed_∇_dot_qᶜ,
                                       closure_auxiliary_velocity
using Oceananigans.Utils: sum_of_velocities

#####
##### Hydrostatic pressure-gradient force
#####
##### Default (z-star / static grids): the naïve chain-rule ∂xᶠᶜᶜ(pHY′), which is exact for the flat
##### geopotential levels of those grids. On a `MultiEnvelopeGrid` (terrain-following / multi-envelope) the
##### levels tilt over topography and the naïve scheme leaves a large spurious horizontal pressure-gradient
##### error. There we use the **density-Jacobian** form (Shchepetkin & McWilliams 2003 style):
#####
#####     ∂φ/∂x|_z = ∂xᵣ(pHY′) − ℑx(b)·∂x_z
#####
##### which is well-balanced — exact for buoyancy linear in z — so a resting stratified ocean stays at rest
##### (verified: ~1000× smaller spurious force than the naïve scheme). `b` is the same buoyancy that was
##### integrated to form pHY′, and `∂x_z` is the coordinate-surface slope (from the fixed-up `znode`).

@inline x_hydrostatic_pressure_gradient(i, j, k, grid, pHY, buoyancy, tracers) = ∂xᶠᶜᶜ(i, j, k, grid, pHY)
@inline y_hydrostatic_pressure_gradient(i, j, k, grid, pHY, buoyancy, tracers) = ∂yᶜᶠᶜ(i, j, k, grid, pHY)

# No buoyancy ⇒ pHY′ = 0; fall back to the default (cheaper, and avoids touching `buoyancy.formulation`).
@inline x_hydrostatic_pressure_gradient(i, j, k, grid::MultiEnvelopeGrid, pHY, ::Nothing, tracers) = ∂xᶠᶜᶜ(i, j, k, grid, pHY)
@inline y_hydrostatic_pressure_gradient(i, j, k, grid::MultiEnvelopeGrid, pHY, ::Nothing, tracers) = ∂yᶜᶠᶜ(i, j, k, grid, pHY)

@inline x_hydrostatic_pressure_gradient(i, j, k, grid::MultiEnvelopeGrid, pHY, buoyancy, tracers) =
    ∂xᵣᶠᶜᶜ(i, j, k, grid, pHY) - ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.formulation, tracers) * ∂x_zᶠᶜᶜ(i, j, k, grid)

@inline y_hydrostatic_pressure_gradient(i, j, k, grid::MultiEnvelopeGrid, pHY, buoyancy, tracers) =
    ∂yᵣᶜᶠᶜ(i, j, k, grid, pHY) - ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.formulation, tracers) * ∂y_zᶜᶠᶜ(i, j, k, grid)

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
    return ( - U_dot_∇u(i, j, k, grid, advection, velocities)
             - U_dot_∇u_hydrostatic_metric(i, j, k, grid, advection, velocities, velocities)
             - explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface)
             - x_f_cross_U(i, j, k, grid, coriolis, velocities)
             - x_hydrostatic_pressure_gradient(i, j, k, grid, hydrostatic_pressure_anomaly, buoyancy, tracers)
             - ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, closure_fields, clock, model_fields, buoyancy)
             - immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid, velocities, u_immersed_bc, closure, closure_fields, clock, model_fields)
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
                                                              v_immersed_bc,
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
    return ( - U_dot_∇v(i, j, k, grid, advection, velocities)
             - U_dot_∇v_hydrostatic_metric(i, j, k, grid, advection, velocities, velocities)
             - explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface)
             - y_f_cross_U(i, j, k, grid, coriolis, velocities)
             - y_hydrostatic_pressure_gradient(i, j, k, grid, hydrostatic_pressure_anomaly, buoyancy, tracers)
             - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, closure_fields, clock, model_fields, buoyancy)
             - immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities, v_immersed_bc, closure, closure_fields, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
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

    biogeochemical_velocities = biogeochemical_drift_velocity(biogeochemistry, val_tracer_name)
    closure_velocities = closure_auxiliary_velocity(closure, closure_fields, val_tracer_name)

    total_velocities = sum_of_velocities(velocities, biogeochemical_velocities, closure_velocities)
    total_velocities = with_advective_forcing(forcing, total_velocities)

    return ( - div_Uc(i, j, k, grid, advection, total_velocities, c)
             - ∇_dot_qᶜ(i, j, k, grid, closure, closure_fields, val_tracer_index, c, clock, model_fields, buoyancy)
             - immersed_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, closure_fields, val_tracer_index, clock, model_fields)
             + biogeochemical_transition(i, j, k, grid, biogeochemistry, val_tracer_name, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end
