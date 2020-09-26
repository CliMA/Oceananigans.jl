using Oceananigans.Advection

@inline tuple_nothing(tup::NamedTuple) = tup
@inline tuple_nothing(::Nothing) = NamedTuple()

"""
    u_velocity_tendency(i, j, k, grid, advection, coriolis, surface_waves, 
                        closure, U, C, K, F, pHY′, clock)

Return the tendency for the horizontal velocity in the x-direction, or the east-west 
direction, ``u``, at grid point `i, j, k`.

The tendency for ``u`` is called ``G_u`` and defined via

    ``∂_t u = G_u - ∂_x ϕ_n``

where ∂_x ϕ_n is the non-hydrostatic pressure gradient in the x-direction.

`coriolis`, `surface_waves`, and `closure` are types encoding information about Coriolis
forces, surface waves, and the prescribed turbulence closure.

The arguments `U`, `C`, and `K` are `NamedTuple`s with the three velocity components,
tracer fields, and precalculated diffusivities where applicable. `F` is a named tuple of 
forcing functions, `pHY′` is the hydrostatic pressure anomaly.

`parameters` is a `NamedTuple` of scalar parameters for user-defined forcing functions 
and `clock` is the physical clock of the model.
"""
@inline function u_velocity_tendency(i, j, k, grid, advection, coriolis, surface_waves, 
                                     closure, U, C, K, F, pHY′, clock)

    return ( - div_ũu(i, j, k, grid, advection, U)
             - x_f_cross_U(i, j, k, grid, coriolis, U)
             - ∂xᶠᵃᵃ(i, j, k, grid, pHY′)
             + ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, clock, closure, U, K)
             + x_curl_Uˢ_cross_U(i, j, k, grid, surface_waves, U, clock.time)
             + ∂t_uˢ(i, j, k, grid, surface_waves, clock.time)
             + F.u(i, j, k, grid, clock, merge(U, C, tuple_nothing(K))))
end

"""
    v_velocity_tendency(i, j, k, grid, advection, coriolis, surface_waves, 
                        closure, U, C, K, F, pHY′, clock)

Return the tendency for the horizontal velocity in the y-direction, or the north-south 
direction, ``v``, at grid point `i, j, k`.

The tendency for ``v`` is called ``G_v`` and defined via

    ``∂_t v = G_v - ∂_y ϕ_n``

where ∂_y ϕ_n is the non-hydrostatic pressure gradient in the y-direction.

`coriolis`, `surface_waves`, and `closure` are types encoding information about Coriolis
forces, surface waves, and the prescribed turbulence closure.

The arguments `U`, `C`, and `K` are `NamedTuple`s with the three velocity components,
tracer fields, and precalculated diffusivities where applicable. `F` is a named tuple of 
forcing functions, `pHY′` is the hydrostatic pressure anomaly.

`parameters` is a `NamedTuple` of scalar parameters for user-defined forcing functions 
and `clock` is the physical clock of the model.
"""
@inline function v_velocity_tendency(i, j, k, grid, advection, coriolis, surface_waves, 
                                     closure, U, C, K, F, pHY′, clock)

    return ( - div_ũv(i, j, k, grid, advection, U)
             - y_f_cross_U(i, j, k, grid, coriolis, U)
             - ∂yᵃᶠᵃ(i, j, k, grid, pHY′)
             + ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, clock, closure, U, K)
             + y_curl_Uˢ_cross_U(i, j, k, grid, surface_waves, U, clock.time)
             + ∂t_vˢ(i, j, k, grid, surface_waves, clock.time)
             + F.v(i, j, k, grid, clock, merge(U, C, tuple_nothing(K))))
end

"""
    w_velocity_tendency(i, j, k, grid, advection, coriolis, surface_waves, 
                        closure, U, C, K, F, clock)
                        
Return the tendency for the vertical velocity ``w`` at grid point `i, j, k`.
The tendency for ``w`` is called ``G_w`` and defined via

    ``∂_t w = G_w - ∂_z ϕ_n``

where ∂_z ϕ_n is the non-hydrostatic pressure gradient in the z-direction.

`coriolis`, `surface_waves`, and `closure` are types encoding information about Coriolis
forces, surface waves, and the prescribed turbulence closure.

The arguments `U`, `C`, and `K` are `NamedTuple`s with the three velocity components,
tracer fields, and precalculated diffusivities where applicable. `F` is a named tuple of 
forcing functions, `pHY′` is the hydrostatic pressure anomaly.

`parameters` is a `NamedTuple` of scalar parameters for user-defined forcing functions 
and `clock` is the physical clock of the model.
"""
@inline function w_velocity_tendency(i, j, k, grid, advection, coriolis, surface_waves, 
                                     closure, U, C, K, F, clock)

    return ( - div_ũw(i, j, k, grid, advection, U)
             - z_f_cross_U(i, j, k, grid, coriolis, U)
             + ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, clock, closure, U, K)
             + z_curl_Uˢ_cross_U(i, j, k, grid, surface_waves, U, clock.time)
             + ∂t_wˢ(i, j, k, grid, surface_waves, clock.time)
             + F.w(i, j, k, grid, clock, merge(U, C, tuple_nothing(K))))
end

"""
    tracer_tendency(i, j, k, grid, c, tracer_index, 
                    advection, closure, buoyancy, U, C, K, Fc, clock)

Return the tendency for a tracer field `c` with index `tracer_index` 
at grid point `i, j, k`.

The tendency for ``c`` is called ``G_c`` and defined via

    ``∂_t c = G_c``

`closure` and `buoyancy` are types encoding information about the prescribed
turbulence closure and buoyancy model.

The arguments `U`, `C`, and `K` are `NamedTuple`s with the three velocity components, 
tracer fields, and  precalculated diffusivities where applicable. 
`Fc` is the user-defined forcing function for tracer `c`.

`clock` keeps track of `clock.time` and `clock.iteration`.
"""
@inline function tracer_tendency(i, j, k, grid, c, tracer_index, 
                                 advection, closure, buoyancy, U, C, K, Fc, clock)

    return ( - div_uc(i, j, k, grid, advection, U, c)
             + ∇_κ_∇c(i, j, k, grid, clock, closure, c, tracer_index, K, C, buoyancy)
             + Fc(i, j, k, grid, clock, merge(U, C, tuple_nothing(K))))
end
