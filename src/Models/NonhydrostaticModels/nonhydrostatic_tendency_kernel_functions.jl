using Oceananigans.Advection
using Oceananigans.BuoyancyModels
using Oceananigans.Coriolis
using Oceananigans.Operators
using Oceananigans.StokesDrifts

using Oceananigans.Biogeochemistry: biogeochemical_transition, biogeochemical_drift_velocity
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ, ∇_dot_qᶜ
using Oceananigans.TurbulenceClosures: immersed_∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₂ⱼ, immersed_∂ⱼ_τ₃ⱼ, immersed_∇_dot_qᶜ
using Oceananigans.Forcings: with_advective_forcing
using Oceananigans.Fields: ZeroField

"return the ``x``-gradient of hydrostatic pressure"
hydrostatic_pressure_gradient_x(i, j, k, grid, hydrostatic_pressure) = ∂xᶠᶜᶜ(i, j, k, grid, hydrostatic_pressure)
hydrostatic_pressure_gradient_x(i, j, k, grid, ::Nothing) = zero(grid)

"return the ``y``-gradient of hydrostatic pressure"
hydrostatic_pressure_gradient_y(i, j, k, grid, hydrostatic_pressure) = ∂yᶜᶠᶜ(i, j, k, grid, hydrostatic_pressure)
hydrostatic_pressure_gradient_y(i, j, k, grid, ::Nothing) = zero(grid)

# Compiler shortcut, for the paranoid about ZeroField
@inline sum_fields(a, b::ZeroField) = a
@inline sum_fields(a, b)            = SumOfArrays{2}(a, b)

@inline sum_fields(a, b::ZeroField, c::ZeroField) = a
@inline sum_fields(a, b, c::ZeroField)            = SumOfArrays{2}(a, b)
@inline sum_fields(a, b::ZeroField, c)            = SumOfArrays{2}(a, c)
@inline sum_fields(a, b, c)                       = SumOfArrays{3}(a, b, c)

@inline assemble_closure_velocities(velocities, background_fields) = velocities

@inline function assemble_closure_velocities(velocities,
                                             background_fields::BackgroundFieldsWithClosureFluxes)

    u = sum_fields(velocities.u, background_fields.u)
    v = sum_fields(velocities.v, background_fields.v)
    w = sum_fields(velocities.w, background_fields.w)

    return (; u, v, w)
end


"""
    $(SIGNATURES)

Return the tendency for the horizontal velocity in the ``x``-direction, or the east-west
direction, ``u``, at grid point `i, j, k`.

The tendency for ``u`` is called ``G_u`` and defined via

```math
∂_t u = G_u - ∂_x p_n ,
```

where ``∂_x p_n`` is the non-hydrostatic kinematic pressure gradient in the ``x``-direction.

`coriolis`, `stokes_drift`, and `closure` are types encoding information about Coriolis
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
                                     stokes_drift,
                                     closure,
                                     u_immersed_bc,
                                     buoyancy,
                                     background_fields,
                                     velocities,
                                     tracers,
                                     auxiliary_fields,
                                     diffusivities,
                                     forcings,
                                     hydrostatic_pressure,
                                     clock)

    total_velocities = (u = sum_fields(velocities.u, background_fields.velocities.u),
                        v = sum_fields(velocities.v, background_fields.velocities.v),
                        w = sum_fields(velocities.w, background_fields.velocities.w))

    total_velocities = with_advective_forcing(forcings.u, total_velocities)

    closure_velocities = assemble_closure_velocities(velocities, background_fields)    
    closure_model_fields = merge(closure_velocities, tracers, auxiliary_fields)
    model_fields = merge(velocities, tracers, auxiliary_fields)

    return ( - div_𝐯u(i, j, k, grid, advection, total_velocities, velocities.u)
             - div_𝐯u(i, j, k, grid, advection, velocities, background_fields.velocities.u)
             + x_dot_g_bᶠᶜᶜ(i, j, k, grid, buoyancy, tracers)
             - x_f_cross_U(i, j, k, grid, coriolis, velocities)
             - hydrostatic_pressure_gradient_x(i, j, k, grid, hydrostatic_pressure)
             - ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, diffusivities, clock, closure_model_fields, buoyancy)
             - immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid, velocities, u_immersed_bc, closure, diffusivities, clock, model_fields)
             + x_curl_Uˢ_cross_U(i, j, k, grid, stokes_drift, velocities, clock.time)
             + ∂t_uˢ(i, j, k, grid, stokes_drift, clock.time)
             + forcings.u(i, j, k, grid, clock, model_fields))
end

"""
    $(SIGNATURES)

Return the tendency for the horizontal velocity in the ``y``-direction, or the north-south
direction, ``v``, at grid point `i, j, k`.

The tendency for ``v`` is called ``G_v`` and defined via

```math
∂_t v = G_v - ∂_y p_n ,
```

where ``∂_y p_n`` is the non-hydrostatic kinematic pressure gradient in the ``y``-direction.

`coriolis`, `stokes_drift`, and `closure` are types encoding information about Coriolis
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
                                     stokes_drift,
                                     closure,
                                     v_immersed_bc,
                                     buoyancy,
                                     background_fields,
                                     velocities,
                                     tracers,
                                     auxiliary_fields,
                                     diffusivities,
                                     forcings,
                                     hydrostatic_pressure,
                                     clock)

    total_velocities = (u = sum_fields(velocities.u, background_fields.velocities.u),
                        v = sum_fields(velocities.v, background_fields.velocities.v),
                        w = sum_fields(velocities.w, background_fields.velocities.w))

    total_velocities = with_advective_forcing(forcings.v, total_velocities)

    closure_velocities = assemble_closure_velocities(velocities, background_fields)    
    closure_model_fields = merge(closure_velocities, tracers, auxiliary_fields)
    model_fields = merge(velocities, tracers, auxiliary_fields)

    return ( - div_𝐯v(i, j, k, grid, advection, total_velocities, velocities.v)
             - div_𝐯v(i, j, k, grid, advection, velocities, background_fields.velocities.v)
             + y_dot_g_bᶜᶠᶜ(i, j, k, grid, buoyancy, tracers)
             - y_f_cross_U(i, j, k, grid, coriolis, velocities)
             - hydrostatic_pressure_gradient_y(i, j, k, grid, hydrostatic_pressure)
             - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, diffusivities, clock, closure_model_fields, buoyancy)
             - immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields)
             + y_curl_Uˢ_cross_U(i, j, k, grid, stokes_drift, velocities, clock.time)
             + ∂t_vˢ(i, j, k, grid, stokes_drift, clock.time)
             + forcings.v(i, j, k, grid, clock, model_fields))
end

# Only add buoyancy if the hydrostatic pressure isa Nothing
@inline maybe_z_dot_g_bᶜᶜᶠ(i, j, k, grid, hydrostatic_pressure, buoyancy, tracers) = zero(grid)
@inline maybe_z_dot_g_bᶜᶜᶠ(i, j, k, grid, ::Nothing, buoyancy, tracers) =
    z_dot_g_bᶜᶜᶠ(i, j, k, grid, buoyancy, tracers)

"""
    $(SIGNATURES)

Return the tendency for the vertical velocity ``w`` at grid point `i, j, k`.

The tendency for ``w`` is called ``G_w`` and defined via

```math
∂_t w = G_w - ∂_z p_n ,
```

where ``∂_z p_n`` is the non-hydrostatic kinematic pressure gradient in the ``z``-direction.

`coriolis`, `stokes_drift`, and `closure` are types encoding information about Coriolis
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
                                     stokes_drift,
                                     closure,
                                     w_immersed_bc,
                                     buoyancy,
                                     background_fields,
                                     velocities,
                                     tracers,
                                     auxiliary_fields,
                                     diffusivities,
                                     forcings,
                                     hydrostatic_pressure,
                                     clock)


    total_velocities = (u = sum_fields(velocities.u, background_fields.velocities.u),
                        v = sum_fields(velocities.v, background_fields.velocities.v),
                        w = sum_fields(velocities.w, background_fields.velocities.w))

    total_velocities = with_advective_forcing(forcings.w, total_velocities)

    closure_velocities = assemble_closure_velocities(velocities, background_fields)    
    closure_model_fields = merge(closure_velocities, tracers, auxiliary_fields)
    model_fields = merge(velocities, tracers, auxiliary_fields)

    return ( - div_𝐯w(i, j, k, grid, advection, total_velocities, velocities.w)
             - div_𝐯w(i, j, k, grid, advection, velocities, background_fields.velocities.w)
             + maybe_z_dot_g_bᶜᶜᶠ(i, j, k, grid, hydrostatic_pressure, buoyancy, tracers)
             - z_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂ⱼ_τ₃ⱼ(i, j, k, grid, closure, diffusivities, clock, closure_model_fields, buoyancy)
             - immersed_∂ⱼ_τ₃ⱼ(i, j, k, grid, velocities, w_immersed_bc, closure, diffusivities, clock, model_fields)
             + z_curl_Uˢ_cross_U(i, j, k, grid, stokes_drift, velocities, clock.time)
             + ∂t_wˢ(i, j, k, grid, stokes_drift, clock.time)
             + forcings.w(i, j, k, grid, clock, model_fields))
end

"""
    $(SIGNATURES)

Return the tendency for a tracer field with index `tracer_index`
at grid point `i, j, k`.

The tendency is called ``G_c`` and defined via

```math
∂_t c = G_c ,
```

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
                                 val_index::Val{tracer_index},
                                 val_tracer_name,
                                 advection,
                                 closure,
                                 c_immersed_bc,
                                 buoyancy,
                                 biogeochemistry,
                                 background_fields,
                                 velocities,
                                 tracers,
                                 auxiliary_fields,
                                 diffusivities,
                                 forcing,
                                 clock) where tracer_index


    biogeochemical_velocities = biogeochemical_drift_velocity(biogeochemistry, val_tracer_name)

    total_velocities = (u = sum_fields(velocities.u, background_fields.velocities.u, biogeochemical_velocities.u),
                        v = sum_fields(velocities.v, background_fields.velocities.v, biogeochemical_velocities.v),
                        w = sum_fields(velocities.w, background_fields.velocities.w, biogeochemical_velocities.w))

    total_velocities = with_advective_forcing(forcing, total_velocities)

    @inbounds c = tracers[tracer_index]
    @inbounds background_fields_c = background_fields.tracers[tracer_index]

    closure_c = if background_fields isa BackgroundFieldsWithClosureFluxes
        sum_fields(c, background_fields_c)
    else
        c
    end

    closure_velocities = assemble_closure_velocities(velocities, background_fields)
    closure_model_fields = merge(closure_velocities, tracers, auxiliary_fields)
    model_fields = merge(velocities, tracers, auxiliary_fields)

    return ( - div_Uc(i, j, k, grid, advection, total_velocities, c)
             - div_Uc(i, j, k, grid, advection, velocities, background_fields_c)
             - ∇_dot_qᶜ(i, j, k, grid, closure, diffusivities, val_index, closure_c, clock, closure_model_fields, buoyancy)
             - immersed_∇_dot_qᶜ(i, j, k, grid, closure_c, c_immersed_bc, closure, diffusivities, val_index, clock, model_fields)
             + biogeochemical_transition(i, j, k, grid, biogeochemistry, val_tracer_name, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end

