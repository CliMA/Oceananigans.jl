using Oceananigans.Advection

@inline regularize_diffusivities_tuple(diffusivities::Tuple) = (diffusivities=diffusivities,)
@inline regularize_diffusivities_tuple(diffusivities::NamedTuple) = diffusivities 
@inline regularize_diffusivities_tuple(::Nothing) = NamedTuple()

"""
    u_velocity_tendency(i, j, k, grid,
                        advection,
                        coriolis,
                        surface_waves,
                        closure,
                        background_fields,
                        velocities,
                        tracers,
                        diffusivities,
                        forcings,
                        hydrostatic_pressure,
                        clock)

Return the tendency for the horizontal velocity in the x-direction, or the east-west 
direction, ``u``, at grid point `i, j, k`.

The tendency for ``u`` is called ``G_u`` and defined via

    ``∂_t u = G_u - ∂_x ϕ_n``

where ∂_x ϕ_n is the non-hydrostatic pressure gradient in the x-direction.

`coriolis`, `surface_waves`, and `closure` are types encoding information about Coriolis
forces, surface waves, and the prescribed turbulence closure.

`background_fields` is a `NamedTuple` containing background velocity and tracer
`FunctionFields`.

The arguments `velocities`, `tracers`, and `diffusivities` are `NamedTuple`s with the three
velocity components, tracer fields, and precalculated diffusivities where applicable.
`forcings` is a named tuple of forcing functions. `hydrostatic_pressure` is the hydrostatic
pressure anomaly.

`clock` keeps track of `clock.time` and `clock.iteration`.
"""
@inline function u_velocity_tendency(i, j, k, grid,
                                     advection,
                                     coriolis,
                                     surface_waves,
                                     closure,
                                     background_fields,
                                     velocities,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     hydrostatic_pressure,
                                     clock)
 
    return ( - div_Uu(i, j, k, grid, advection, velocities, velocities.u)
             - div_Uu(i, j, k, grid, advection, background_fields.velocities, velocities.u)
             - div_Uu(i, j, k, grid, advection, velocities, background_fields.velocities.u)
             - x_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂xᶠᵃᵃ(i, j, k, grid, hydrostatic_pressure)
             + ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, clock, closure, velocities, diffusivities)
             + x_curl_Uˢ_cross_U(i, j, k, grid, surface_waves, velocities, clock.time)
             + ∂t_uˢ(i, j, k, grid, surface_waves, clock.time)
             + forcings.u(i, j, k, grid, clock,
                          merge(velocities, tracers, regularize_diffusivities_tuple(diffusivities))))
end

"""
    v_velocity_tendency(i, j, k, grid,
                        advection,
                        coriolis,
                        surface_waves,
                        closure,
                        background_fields,
                        velocities,
                        tracers,
                        diffusivities,
                        forcings,
                        hydrostatic_pressure,
                        clock)

Return the tendency for the horizontal velocity in the y-direction, or the north-south 
direction, ``v``, at grid point `i, j, k`.

The tendency for ``v`` is called ``G_v`` and defined via

    ``∂_t v = G_v - ∂_y ϕ_n``

where ∂_y ϕ_n is the non-hydrostatic pressure gradient in the y-direction.

`coriolis`, `surface_waves`, and `closure` are types encoding information about Coriolis
forces, surface waves, and the prescribed turbulence closure.

`background_fields` is a `NamedTuple` containing background velocity and tracer
`FunctionFields`.

The arguments `velocities`, `tracers`, and `diffusivities` are `NamedTuple`s with the three
velocity components, tracer fields, and precalculated diffusivities where applicable.
`forcings` is a named tuple of forcing functions. `hydrostatic_pressure` is the hydrostatic
pressure anomaly.

`clock` keeps track of `clock.time` and `clock.iteration`.
"""
@inline function v_velocity_tendency(i, j, k, grid,
                                     advection,
                                     coriolis,
                                     surface_waves,
                                     closure,
                                     background_fields,
                                     velocities,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     hydrostatic_pressure,
                                     clock)

    return ( - div_Uv(i, j, k, grid, advection, velocities, velocities.v)
             - div_Uv(i, j, k, grid, advection, background_fields.velocities, velocities.v)
             - div_Uv(i, j, k, grid, advection, velocities, background_fields.velocities.v)
             - y_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂yᵃᶠᵃ(i, j, k, grid, hydrostatic_pressure)
             + ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, clock, closure, velocities, diffusivities)
             + y_curl_Uˢ_cross_U(i, j, k, grid, surface_waves, velocities, clock.time)
             + ∂t_vˢ(i, j, k, grid, surface_waves, clock.time)
             + forcings.v(i, j, k, grid, clock,
                          merge(velocities, tracers, regularize_diffusivities_tuple(diffusivities))))
end

"""
    w_velocity_tendency(i, j, k, grid,
                        advection,
                        coriolis,
                        surface_waves, 
                        closure,
                        background_fields,
                        velocities,
                        tracers,
                        diffusivities,
                        forcings,
                        clock)
                        
Return the tendency for the vertical velocity ``w`` at grid point `i, j, k`.
The tendency for ``w`` is called ``G_w`` and defined via

    ``∂_t w = G_w - ∂_z ϕ_n``

where ∂_z ϕ_n is the non-hydrostatic pressure gradient in the z-direction.

`coriolis`, `surface_waves`, and `closure` are types encoding information about Coriolis
forces, surface waves, and the prescribed turbulence closure.

`background_fields` is a `NamedTuple` containing background velocity and tracer
`FunctionFields`.

The arguments `velocities`, `tracers`, and `diffusivities` are `NamedTuple`s with the three
velocity components, tracer fields, and precalculated diffusivities where applicable.
`forcings` is a named tuple of forcing functions.

`clock` keeps track of `clock.time` and `clock.iteration`.
"""
@inline function w_velocity_tendency(i, j, k, grid,
                                     advection,
                                     coriolis,
                                     surface_waves, 
                                     closure,
                                     background_fields,
                                     velocities,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     clock)

    return ( - div_Uw(i, j, k, grid, advection, velocities, velocities.w)
             - div_Uw(i, j, k, grid, advection, background_fields.velocities, velocities.w)
             - div_Uw(i, j, k, grid, advection, velocities, background_fields.velocities.w)
             - z_f_cross_U(i, j, k, grid, coriolis, velocities)
             + ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, clock, closure, velocities, diffusivities)
             + z_curl_Uˢ_cross_U(i, j, k, grid, surface_waves, velocities, clock.time)
             + ∂t_wˢ(i, j, k, grid, surface_waves, clock.time)
             + forcings.w(i, j, k, grid, clock,
                          merge(velocities, tracers, regularize_diffusivities_tuple(diffusivities))))
end

"""
    tracer_tendency(i, j, k, grid, 
                    val_tracer_index::Val{tracer_index},
                    advection,
                    closure,
                    buoyancy,
                    background_fields,
                    velocities,
                    tracers,
                    diffusivities,
                    forcing,
                    clock)

Return the tendency for a tracer field with index `tracer_index` 
at grid point `i, j, k`.

The tendency is called ``G_c`` and defined via

    ``∂_t c = G_c``

where `c = C[tracer_index]`. 

`closure` and `buoyancy` are types encoding information about the prescribed
turbulence closure and buoyancy model.

`background_fields` is a `NamedTuple` containing background velocity and tracer
`FunctionFields`.

The arguments `velocities`, `tracers`, and `diffusivities` are `NamedTuple`s with the three
velocity components, tracer fields, and precalculated diffusivities where applicable.
`forcings` is a named tuple of forcing functions.

`clock` keeps track of `clock.time` and `clock.iteration`.
"""
@inline function tracer_tendency(i, j, k, grid,
                                 val_tracer_index::Val{tracer_index},
                                 advection,
                                 closure,
                                 buoyancy,
                                 background_fields,
                                 velocities,
                                 tracers,
                                 diffusivities,
                                 forcing,
                                 clock) where tracer_index

    @inbounds c = tracers[tracer_index]
    @inbounds background_fields_c = background_fields.tracers[tracer_index]

    return ( - div_Uc(i, j, k, grid, advection, velocities, c)
             - div_Uc(i, j, k, grid, advection, background_fields.velocities, c)
             - div_Uc(i, j, k, grid, advection, velocities, background_fields_c)
             + ∇_κ_∇c(i, j, k, grid, clock, closure, c, val_tracer_index, diffusivities, tracers, buoyancy)
             + forcing(i, j, k, grid, clock,
                       merge(velocities, tracers, regularize_diffusivities_tuple(diffusivities))))
end
