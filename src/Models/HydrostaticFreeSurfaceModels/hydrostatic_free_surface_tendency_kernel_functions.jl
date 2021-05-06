using Oceananigans.BuoyancyModels
using Oceananigans.Coriolis
using Oceananigans.Operators
using Oceananigans.Operators: ∂xᶠᶜᵃ, ∂yᶜᶠᵃ
using Oceananigans.StokesDrift
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ, ∇_dot_qᶜ
using Oceananigans.Advection: div_Uc

"""
    hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)

Return the tendency for the horizontal velocity in the x-direction, or the east-west 
direction, ``u``, at grid point `i, j, k` for a HydrostaticFreeSurfaceModel.

The tendency for ``u`` is called ``G_u`` and defined via

    ``∂_t u = G_u - ∂_x ϕ_n``

where ϕ_n is the part of the barotropic pressure that's treated
implicitly during time-stepping.
"""
@inline function hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid,
                                                              advection,
                                                              coriolis,
                                                              closure,
                                                              velocities,
                                                              free_surface,
                                                              tracers,
                                                              diffusivities,
                                                              hydrostatic_pressure_anomaly,
                                                              auxiliary_fields,
                                                              forcings,
                                                              clock)

    model_fields = merge(hydrostatic_prognostic_fields(velocities, free_surface, tracers), auxiliary_fields)
 
    return ( - U_dot_∇u(i, j, k, grid, advection, velocities)
             - explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface)
             - x_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂xᶠᶜᵃ(i, j, k, grid, hydrostatic_pressure_anomaly)
             - ∂ⱼ_τ₁ⱼ(i, j, k, grid, clock, closure, velocities, diffusivities)
             + forcings.u(i, j, k, grid, clock, hydrostatic_prognostic_fields(velocities, free_surface, tracers)))
end

"""
    hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)

Return the tendency for the horizontal velocity in the y-direction, or the east-west 
direction, ``v``, at grid point `i, j, k` for a HydrostaticFreeSurfaceModel.

The tendency for ``v`` is called ``G_v`` and defined via

    ``∂_t v = G_v - ∂_y ϕ_n``

where ϕ_n is the part of the barotropic pressure that's treated
implicitly during time-stepping.
"""
@inline function hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid,
                                                              advection,
                                                              coriolis,
                                                              closure,
                                                              velocities,
                                                              free_surface,
                                                              tracers,
                                                              diffusivities,
                                                              hydrostatic_pressure_anomaly,
                                                              auxiliary_fields,
                                                              forcings,
                                                              clock)

    model_fields = merge(hydrostatic_prognostic_fields(velocities, free_surface, tracers), auxiliary_fields)

    return ( - U_dot_∇v(i, j, k, grid, advection, velocities)
             - explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface)
             - y_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂yᶜᶠᵃ(i, j, k, grid, hydrostatic_pressure_anomaly)
             - ∂ⱼ_τ₂ⱼ(i, j, k, grid, clock, closure, velocities, diffusivities)
             + forcings.v(i, j, k, grid, clock, hydrostatic_prognostic_fields(velocities, free_surface, tracers)))
end

"""
    hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)

Return the tendency for a tracer field with index `tracer_index` 
at grid point `i, j, k`.

The tendency is called ``G_c`` and defined via

    ``∂_t c = G_c``

where `c = C[tracer_index]`. 
"""
@inline function hydrostatic_free_surface_tracer_tendency(i, j, k, grid,
                                                          val_tracer_index::Val{tracer_index},
                                                          advection,
                                                          closure,
                                                          buoyancy,
                                                          velocities,
                                                          free_surface,
                                                          tracers,
                                                          diffusivities,
                                                          auxiliary_fields,
                                                          forcing,
                                                          clock) where tracer_index

    @inbounds c = tracers[tracer_index]

    model_fields = merge(hydrostatic_prognostic_fields(velocities, free_surface, tracers), auxiliary_fields)

    return ( - div_Uc(i, j, k, grid, advection, velocities, c)
             - ∇_dot_qᶜ(i, j, k, grid, clock, closure, c, val_tracer_index, diffusivities, tracers, buoyancy)
             + forcing(i, j, k, grid, clock, hydrostatic_prognostic_fields(velocities, free_surface, tracers)))
end

"""
     free_surface_tendency(i, j, grid, 
                           velocities,
                           free_surface,
                           tracers,
                           forcing,
                           clock)

Return the tendency for an explicit free surface at horizontal grid point `i, j`.

The tendency is called ``G_η`` and defined via

    ``∂_t η = G_η``
"""
@inline function free_surface_tendency(i, j, grid,
                                       velocities,
                                       free_surface,
                                       tracers,
                                       auxiliary_fields,
                                       forcings,
                                       clock)

    k_surface = grid.Nz + 1
    model_fields = merge(hydrostatic_prognostic_fields(velocities, free_surface, tracers), auxiliary_fields)

    return @inbounds (   velocities.w[i, j, k_surface]
                       + forcings.η(i, j, k_surface, grid, clock, model_fields))
end
