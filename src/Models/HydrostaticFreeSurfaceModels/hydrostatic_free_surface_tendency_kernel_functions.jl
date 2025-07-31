using Oceananigans.BuoyancyFormulations
using Oceananigans.Coriolis
using Oceananigans.Operators
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∇_dot_qᶜ
using Oceananigans.Biogeochemistry: biogeochemical_transition, biogeochemical_drift_velocity
using Oceananigans.TurbulenceClosures: immersed_∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₂ⱼ, immersed_∂ⱼ_τ₃ⱼ, immersed_∇_dot_qᶜ
using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v
using Oceananigans.Forcings: with_advective_forcing
using Oceananigans.TurbulenceClosures: shear_production, buoyancy_flux, dissipation, closure_turbulent_velocity
using Oceananigans.Utils: sum_of_velocities
using KernelAbstractions: @private

import Oceananigans.TurbulenceClosures: hydrostatic_turbulent_kinetic_energy_tendency

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
                                                              diffusivities,
                                                              hydrostatic_pressure_anomaly,
                                                              auxiliary_fields,
                                                              ztype,
                                                              clock,
                                                              forcing)

    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers), auxiliary_fields)

    return ( - U_dot_∇u(i, j, k, grid, advection, velocities)
             - explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface)
             - x_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂xᶠᶜᶜ(i, j, k, grid, hydrostatic_pressure_anomaly)
             - grid_slope_contribution_x(i, j, k, grid, buoyancy, ztype, model_fields)
             - ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy)
             - immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid, velocities, u_immersed_bc, closure, diffusivities, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
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

    return ( - U_dot_∇v(i, j, k, grid, advection, velocities)
             - explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface)
             - y_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂yᶜᶠᶜ(i, j, k, grid, hydrostatic_pressure_anomaly)
             - grid_slope_contribution_y(i, j, k, grid, buoyancy, ztype, model_fields)
             - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy)
             - immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields)
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
                                                          diffusivities,
                                                          auxiliary_fields,
                                                          clock,
                                                          forcing) where tracer_index

    @inbounds c = tracers[tracer_index]
    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers),
                         auxiliary_fields,
                         biogeochemical_auxiliary_fields(biogeochemistry))

    biogeochemical_velocities = biogeochemical_drift_velocity(biogeochemistry, val_tracer_name)
    closure_velocities = closure_turbulent_velocity(closure, diffusivities, val_tracer_name)

    total_velocities = sum_of_velocities(velocities, biogeochemical_velocities, closure_velocities)
    total_velocities = with_advective_forcing(forcing, total_velocities)

    return ( - div_Uc(i, j, k, grid, advection, total_velocities, c)
             - ∇_dot_qᶜ(i, j, k, grid, closure, diffusivities, val_tracer_index, c, clock, model_fields, buoyancy)
             - immersed_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, diffusivities, val_tracer_index, clock, model_fields)
             + biogeochemical_transition(i, j, k, grid, biogeochemistry, val_tracer_name, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end


@inline function hydrostatic_free_surface_TKE_tendency(i, j, k, grid,
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
                                                       diffusivities,
                                                       auxiliary_fields,
                                                       clock,
                                                       forcing) where tracer_index

    @inbounds c = tracers[tracer_index]
    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers),
                         auxiliary_fields,
                         biogeochemical_auxiliary_fields(biogeochemistry))

    biogeochemical_velocities = biogeochemical_drift_velocity(biogeochemistry, val_tracer_name)
    closure_velocities = closure_turbulent_velocity(closure, diffusivities, val_tracer_name)

    total_velocities = sum_of_velocities(velocities, biogeochemical_velocities, closure_velocities)
    total_velocities = with_advective_forcing(forcing, total_velocities)

    Gⁿ_closure = closure_dependent_forcing(i, j, k, grid, closure, diffusivities, val_tracer_name, c, clock, velocities, tracers, buoyancy, model_fields)

    return ( - div_Uc(i, j, k, grid, advection, total_velocities, c)
             - ∇_dot_qᶜ(i, j, k, grid, closure, diffusivities, val_tracer_index, c, clock, model_fields, buoyancy)
             - immersed_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, diffusivities, val_tracer_index, clock, model_fields)
             + biogeochemical_transition(i, j, k, grid, biogeochemistry, val_tracer_name, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields)
             + Gⁿ_closure)
end

@inline function closure_dependent_forcing(i, j, k, grid, closures::Tuple, diffusivities, val_tracer_name, c, clock, velocities, tracers, buoyancy, model_fields)

    Gⁿ = 0
    for n in eachindex(closures)
        @inbounds Gⁿ += closure_dependent_forcing(i, j, k, grid, closures[n], diffusivities, val_tracer_name, c, clock, velocities, tracers, buoyancy, model_fields)
    end

    return Gⁿ
end

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: FlavorOfCATKE, explicit_buoyancy_flux, bottommost_active_node, dissipation_rate

@inline function closure_dependent_forcing(i, j, k, grid, closure::FlavorOfCATKE, diffusivities, ::Val{:e}, e, clock, model_fields, buoyancy)

    closure_ij = getclosure(i, j, closure)

    # Compute additional diagonal component of the linear TKE operator
    wb = explicit_buoyancy_flux(i, j, k, grid, closure_ij, velocities, model_fields, buoyancy, diffusivities)
    wb⁻ = min(zero(grid), wb)
    wb⁺ = max(zero(grid), wb)

    κe = diffusivity_fields.κe
    Le = diffusivity_fields.Le

    eⁱʲᵏ = @inbounds e[i, j, k]
    eᵐⁱⁿ = closure_ij.minimum_tke
    wb⁻_e = wb⁻ / eⁱʲᵏ * (eⁱʲᵏ > eᵐⁱⁿ)

    on_bottom = bottommost_active_node(i, j, k, grid, c, c, c)
    active = !inactive_cell(i, j, k, grid)
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    Cᵂϵ = closure_ij.turbulent_kinetic_energy_equation.Cᵂϵ
    e⁺ = clip(eⁱʲᵏ)
    w★ = sqrt(e⁺)
    div_Jᵉ_e = - on_bottom * Cᵂϵ * w★ / Δz

    # Implicit TKE dissipation
    ω = dissipation_rate(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, diffusivities)

    @inbounds Le[i, j, k] = (wb⁻_e - ω + div_Jᵉ_e) * active

    # Compute fast TKE RHS
    u⁺ = velocities.u
    v⁺ = velocities.v
    uⁿ = velocities.u
    vⁿ = velocities.v
    κu = diffusivities.κu

    # TODO: correctly handle closure / diffusivity tuples
    # TODO: the shear_production is actually a slow term so we _could_ precompute.
    P = shear_production(i, j, k, grid, κu, uⁿ, u⁺, vⁿ, v⁺)
    ϵ = dissipation(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, diffusivities)
    fast_Gⁿe = P + wb⁺ - ϵ

    return fast_Gⁿe
end