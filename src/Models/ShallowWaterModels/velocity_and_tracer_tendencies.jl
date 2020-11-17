using Oceananigans.Advection
using Oceananigans.Coriolis
using Oceananigans.Operators

"""
    u_velocity_tendency(i, j, k, grid,
                        advection,
                        coriolis,
                        velocities,
                        tracers,
                        diffusivities,
                        forcings,
                        clock)

Return the tendency for the horizontal velocity in the x-direction, or the east-west 
direction, ``u``, at grid point `i, j, k`.

The tendency for ``u`` is called ``G_u`` and defined via

    ``∂_t u = G_u - ∂_x ϕ_n``

where ∂_x ϕ_n is the non-hydrostatic pressure gradient in the x-direction.

`coriolis`, `surface_waves`, and `closure` are types encoding information about Coriolis
forces, surface waves, and the prescribed turbulence closure.

The arguments `velocities`, `tracers`, and `diffusivities` are `NamedTuple`s with the three
velocity components, tracer fields, and precalculated diffusivities where applicable.
`forcings` is a named tuple of forcing functions. `hydrostatic_pressure` is the hydrostatic
pressure anomaly.

`clock` keeps track of `clock.time` and `clock.iteration`.
"""
@inline function u_velocity_tendency(i, j, k, grid,
                                     advection,
                                     coriolis,
                                     velocities,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     clock)

# Put in h for hydrostatic pressure
    return ( - ∂xᶠᵃᵃ(i, j, k, grid, hydrostatic_pressure) )
end

"""
    v_velocity_tendency(i, j, k, grid,
                        advection,
                        coriolis,
                        velocities,
                        tracers,
                        diffusivities,
                        forcings,
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
                                     velocities,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     clock)

# Put in h for hydrostatic pressure
    return ( - ∂yᵃᶠᵃ(i, j, k, grid, hydrostatic_pressure) )

end

"""
    height_tendency(i, j, k, grid, 
                    advection,
                    velocities,
                    tracers,
                    diffusivities,
                    forcing,
                    clock)

Return the tendency for a tracer field with index `tracer_index` 
at grid point `i, j, k`.

`clock` keeps track of `clock.time` and `clock.iteration`.

"""
@inline function height_tendency(i, j, k, grid,
                                 advection,
                                 velocities,
                                 tracers,
                                 diffusivities,
                                 forcing,
                                 clock) where tracer_index

# Put in u for hydrostatic pressure
    return ( - ∂xᶠᵃᵃ(i, j, k, grid, hydrostatic_pressure) )

end

"""
    tracer_tendency(i, j, k, grid, 
                    val_tracer_index::Val{tracer_index},
                    advection,
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
                                 velocities,
                                 tracers,
                                 diffusivities,
                                 forcing,
                                 clock) where tracer_index

    @inbounds c = tracers[tracer_index]

    return ( - div_Uc(i, j, k, grid, advection, velocities, c) )

end
