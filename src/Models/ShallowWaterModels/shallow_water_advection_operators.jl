using Oceananigans.Advection:
    FluxFormAdvection,
    CenteredScheme,
    UpwindScheme,
    _advective_momentum_flux_Uu,
    _advective_momentum_flux_Uv,
    _advective_momentum_flux_Vu,
    _advective_momentum_flux_Vv,
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    horizontal_advection_U,
    horizontal_advection_V,
    bernoulli_head_U,
    bernoulli_head_V,
    spherical_shell_horizontal_volume_flux_velocities

using Oceananigans.Grids: AbstractGrid, SphericalShellGrid
using Oceananigans.Operators: horizontal_volume_flux_div_xyᶜᶜᶜ,
                              covariant_to_volume_flux_uᶠᶜᶜ,
                              covariant_to_volume_flux_vᶜᶠᶜ

#####
##### Momentum flux operators
#####

@inline shallow_water_momentum_flux_u(solution) = solution.uh
@inline shallow_water_momentum_flux_v(solution) = solution.vh

@inline momentum_flux_huu(i, j, k, grid, advection, solution) =
    @inbounds _advective_momentum_flux_Uu(i, j, k, grid, advection,
                                          shallow_water_momentum_flux_u(solution),
                                          shallow_water_momentum_flux_u(solution)) / solution.h[i, j, k]

@inline function momentum_flux_huu(i, j, k, grid::SphericalShellGrid, advection, solution)
    converted_transport =
        spherical_shell_horizontal_volume_flux_velocities(grid,
                                                          (shallow_water_momentum_flux_u(solution),
                                                           shallow_water_momentum_flux_v(solution)))
    return @inbounds _advective_momentum_flux_Uu(i, j, k, grid, advection,
                                                 converted_transport,
                                                 shallow_water_momentum_flux_u(solution)) / solution.h[i, j, k]
end

@inline momentum_flux_hvu(i, j, k, grid, advection, solution) =
    @inbounds _advective_momentum_flux_Vu(i, j, k, grid, advection,
                                          shallow_water_momentum_flux_v(solution),
                                          shallow_water_momentum_flux_u(solution)) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline function momentum_flux_hvu(i, j, k, grid::SphericalShellGrid, advection, solution)
    converted_transport =
        spherical_shell_horizontal_volume_flux_velocities(grid,
                                                          (shallow_water_momentum_flux_u(solution),
                                                           shallow_water_momentum_flux_v(solution)))
    return @inbounds _advective_momentum_flux_Vu(i, j, k, grid, advection,
                                                 converted_transport,
                                                 shallow_water_momentum_flux_u(solution)) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)
end

@inline momentum_flux_huv(i, j, k, grid, advection, solution) =
    @inbounds _advective_momentum_flux_Uv(i, j, k, grid, advection,
                                          shallow_water_momentum_flux_u(solution),
                                          shallow_water_momentum_flux_v(solution)) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline function momentum_flux_huv(i, j, k, grid::SphericalShellGrid, advection, solution)
    converted_transport =
        spherical_shell_horizontal_volume_flux_velocities(grid,
                                                          (shallow_water_momentum_flux_u(solution),
                                                           shallow_water_momentum_flux_v(solution)))
    return @inbounds _advective_momentum_flux_Uv(i, j, k, grid, advection,
                                                 converted_transport,
                                                 shallow_water_momentum_flux_v(solution)) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)
end

@inline momentum_flux_hvv(i, j, k, grid, advection, solution) =
    @inbounds _advective_momentum_flux_Vv(i, j, k, grid, advection,
                                          shallow_water_momentum_flux_v(solution),
                                          shallow_water_momentum_flux_v(solution)) / solution.h[i, j, k]

@inline function momentum_flux_hvv(i, j, k, grid::SphericalShellGrid, advection, solution)
    converted_transport =
        spherical_shell_horizontal_volume_flux_velocities(grid,
                                                          (shallow_water_momentum_flux_u(solution),
                                                           shallow_water_momentum_flux_v(solution)))
    return @inbounds _advective_momentum_flux_Vv(i, j, k, grid, advection,
                                                 converted_transport,
                                                 shallow_water_momentum_flux_v(solution)) / solution.h[i, j, k]
end

#####
##### Momentum flux divergence operators
#####

@inline div_mom_u(i, j, k, grid, advection, solution, formulation) =
    1 / Azᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_huu, advection, solution) +
                                δyᵃᶜᵃ(i, j, k, grid, momentum_flux_hvu, advection, solution))

@inline div_mom_v(i, j, k, grid, advection, solution, formulation) =
    1 / Azᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_huv, advection, solution) +
                                δyᵃᶠᵃ(i, j, k, grid, momentum_flux_hvv, advection, solution))

@inline function div_mom_u(i, j, k, grid, advection, solution, formulation::VectorInvariantFormulation)
    transport_u = shallow_water_transport_u(solution, formulation)
    transport_v = shallow_water_transport_v(solution, formulation)
    return (+ horizontal_advection_U(i, j, k, grid, advection, transport_u, transport_v)
            + bernoulli_head_U(i, j, k, grid, advection, transport_u, transport_v))
end

@inline function div_mom_v(i, j, k, grid, advection, solution, formulation::VectorInvariantFormulation)
    transport_u = shallow_water_transport_u(solution, formulation)
    transport_v = shallow_water_transport_v(solution, formulation)
    return (+ horizontal_advection_V(i, j, k, grid, advection, transport_u, transport_v)
            + bernoulli_head_V(i, j, k, grid, advection, transport_u, transport_v))
end

# Support for no advection
@inline div_mom_u(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution, formulation) where FT = zero(FT)
@inline div_mom_v(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution, formulation) where FT = zero(FT)
@inline div_mom_u(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution, ::VectorInvariantFormulation) where FT = zero(FT)
@inline div_mom_v(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution, ::VectorInvariantFormulation) where FT = zero(FT)

#####
##### Mass transport divergence operator
#####

"""
    div_Uh(i, j, k, grid, advection, solution, formulation)

Calculate the divergence of the mass flux into a cell,

```
1/Az * [δxᶜᵃᵃ(Δy * uh) + δyᵃᶜᵃ(Δx * vh)]
```

which ends up at the location `ccc`.
"""
@inline shallow_water_transport_u(solution, formulation) = solution.uh
@inline shallow_water_transport_v(solution, formulation) = solution.vh
@inline shallow_water_transport_u(solution, ::VectorInvariantFormulation) = solution.u
@inline shallow_water_transport_v(solution, ::VectorInvariantFormulation) = solution.v

struct ShallowWaterConvertedTransportU{G, S}
    grid :: G
    solution :: S
end

struct ShallowWaterConvertedTransportV{G, S}
    grid :: G
    solution :: S
end

@inline Base.getindex(U::ShallowWaterConvertedTransportU, i, j, k) =
    covariant_to_volume_flux_uᶠᶜᶜ(i, j, k, U.grid, U.solution.u, U.solution.v)

@inline Base.getindex(V::ShallowWaterConvertedTransportV, i, j, k) =
    covariant_to_volume_flux_vᶜᶠᶜ(i, j, k, V.grid, V.solution.u, V.solution.v)

struct ConservativeShallowWaterVelocityU{G, S}
    grid :: G
    solution :: S
end

struct ConservativeShallowWaterVelocityV{G, S}
    grid :: G
    solution :: S
end

@inline Base.getindex(U::ConservativeShallowWaterVelocityU, i, j, k) =
    @inbounds U.solution.uh[i, j, k] / ℑxᶠᵃᵃ(i, j, k, U.grid, U.solution.h)

@inline Base.getindex(V::ConservativeShallowWaterVelocityV, i, j, k) =
    @inbounds V.solution.vh[i, j, k] / ℑyᵃᶠᵃ(i, j, k, V.grid, V.solution.h)

@inline function div_Uh(i, j, k, grid, advection, solution, formulation)
    transport_u = shallow_water_transport_u(solution, formulation)
    transport_v = shallow_water_transport_v(solution, formulation)
    return 1/Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, transport_u) +
                                     δyᵃᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, transport_v))
end

@inline function div_Uh(i, j, k, grid::SphericalShellGrid, advection, solution, formulation)
    transport_u = shallow_water_transport_u(solution, formulation)
    transport_v = shallow_water_transport_v(solution, formulation)
    return 1 / Azᶜᶜᶜ(i, j, k, grid) *
           horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, k, grid, transport_u, transport_v)
end

@inline div_Uh(i, j, k, grid::SphericalShellGrid, advection, solution, formulation::VectorInvariantFormulation) =
        div_Uc(i, j, k, grid, advection, solution, solution.h, formulation)

@inline div_Uh(i, j, k, grid, advection, solution, formulation::VectorInvariantFormulation) =
        div_Uc(i, j, k, grid, advection, solution, solution.h, formulation)

#####
##### Tracer advection operator
#####

@inline transport_tracer_flux_x(i, j, k, grid, advection, uh, h, c) =
    @inbounds _advective_tracer_flux_x(i, j, k, grid, advection, uh, c) / ℑxᶠᵃᵃ(i, j, k, grid, h)

@inline transport_tracer_flux_y(i, j, k, grid, advection, vh, h, c) =
    @inbounds _advective_tracer_flux_y(i, j, k, grid, advection, vh, c) / ℑyᵃᶠᵃ(i, j, k, grid, h)

"""
    div_Uc(i, j, k, grid, advection, solution, c, formulation)

Calculate the divergence of the flux of a tracer quantity ``c`` being advected by
a velocity field ``𝐔 = (u, v)``, ``𝛁·(𝐔c)``,

```
1/Az * [δxᶜᵃᵃ(Δy * uh * ℑxᶠᵃᵃ(c) / h) + δyᵃᶜᵃ(Δx * vh * ℑyᵃᶠᵃ(c) / h)]
```

which ends up at the location `ccc`.
"""

@inline function div_Uc(i, j, k, grid, advection, solution, c, formulation)
    transport_u = shallow_water_transport_u(solution, formulation)
    transport_v = shallow_water_transport_v(solution, formulation)
    return 1/Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, transport_tracer_flux_x, advection, transport_u, solution.h, c) +
                                     δyᵃᶜᵃ(i, j, k, grid, transport_tracer_flux_y, advection, transport_v, solution.h, c))
end

@inline function div_Uc(i, j, k, grid::SphericalShellGrid, advection, solution, c, formulation)
    transport_u = shallow_water_transport_u(solution, formulation)
    transport_v = shallow_water_transport_v(solution, formulation)
    converted_velocities =
        spherical_shell_horizontal_volume_flux_velocities(grid, (transport_u, transport_v))

    return 1 / Azᶜᶜᶜ(i, j, k, grid) *
           (δxᶜᵃᵃ(i, j, k, grid, transport_tracer_flux_x, advection, converted_velocities, solution.h, c) +
            δyᵃᶜᵃ(i, j, k, grid, transport_tracer_flux_y, advection, converted_velocities, solution.h, c))
end

@inline function div_Uc(i, j, k, grid, advection, solution, c, formulation::VectorInvariantFormulation)
    transport_u = shallow_water_transport_u(solution, formulation)
    transport_v = shallow_water_transport_v(solution, formulation)
    return 1/Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection, transport_u, c) +
                                     δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection, transport_v, c))
end

@inline function div_Uc(i, j, k, grid::SphericalShellGrid, advection, solution, c, formulation::VectorInvariantFormulation)
    transport_u = shallow_water_transport_u(solution, formulation)
    transport_v = shallow_water_transport_v(solution, formulation)
    converted_velocities =
        Oceananigans.Advection.spherical_shell_horizontal_volume_flux_velocities(grid, (transport_u, transport_v))

    return 1 / Azᶜᶜᶜ(i, j, k, grid) *
           (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection, converted_velocities, c) +
            δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection, converted_velocities, c))
end

# Support for no advection
@inline div_Uc(i, j, k, grid::AbstractGrid, ::Nothing, solution, c, formulation) = zero(grid)
@inline div_Uh(i, j, k, grid::AbstractGrid, ::Nothing, solution, formulation)    = zero(grid)
@inline div_Uc(i, j, k, grid::SphericalShellGrid, ::Nothing, solution, c, formulation) = zero(grid)
@inline div_Uh(i, j, k, grid::SphericalShellGrid, ::Nothing, solution, formulation)    = zero(grid)

# Disambiguation
@inline div_Uc(i, j, k, grid::AbstractGrid, ::Nothing, solution, c, ::VectorInvariantFormulation) = zero(grid)
@inline div_Uh(i, j, k, grid::AbstractGrid, ::Nothing, solution, ::VectorInvariantFormulation)    = zero(grid)
@inline div_Uc(i, j, k, grid::SphericalShellGrid, ::Nothing, solution, c, ::VectorInvariantFormulation) = zero(grid)
@inline div_Uh(i, j, k, grid::SphericalShellGrid, ::Nothing, solution, ::VectorInvariantFormulation)    = zero(grid)

@inline u(i, j, k, grid, solution) = @inbounds solution.uh[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, solution.h)
@inline v(i, j, k, grid, solution) = @inbounds solution.vh[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, solution.h)

"""
    c_div_U(i, j, k, grid, solution, c, formulation)

Calculate the product of the tracer concentration ``c`` with
the horizontal divergence of the velocity field ``𝐔 = (u, v)``, ``c ∇·𝐔``,

```
c * 1/Az * [δxᶜᵃᵃ(Δy * uh / h) + δyᵃᶜᵃ(Δx * vh / h)]
```

which ends up at the location `ccc`.
"""
@inline c_div_U(i, j, k, grid, solution, c, formulation) =
    @inbounds c[i, j, k] * 1/Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u, solution) + δyᵃᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v, solution))

@inline function c_div_U(i, j, k, grid::SphericalShellGrid, solution, c, ::ConservativeFormulation)
    velocity_u = ConservativeShallowWaterVelocityU(grid, solution)
    velocity_v = ConservativeShallowWaterVelocityV(grid, solution)

    return @inbounds c[i, j, k] * 1 / Azᶜᶜᶜ(i, j, k, grid) *
           horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, k, grid, velocity_u, velocity_v)
end

@inline function c_div_U(i, j, k, grid::SphericalShellGrid, solution, c, formulation)
    transport_u = shallow_water_transport_u(solution, formulation)
    transport_v = shallow_water_transport_v(solution, formulation)
    return @inbounds c[i, j, k] * 1 / Azᶜᶜᶜ(i, j, k, grid) *
           horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, k, grid, transport_u, transport_v)
end

@inline function c_div_U(i, j, k, grid, solution, c, formulation::VectorInvariantFormulation)
    transport_u = shallow_water_transport_u(solution, formulation)
    transport_v = shallow_water_transport_v(solution, formulation)
    return @inbounds c[i, j, k] * 1/Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, transport_u) +
                                                            δyᵃᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, transport_v))
end

@inline function c_div_U(i, j, k, grid::SphericalShellGrid, solution, c, formulation::VectorInvariantFormulation)
    transport_u = shallow_water_transport_u(solution, formulation)
    transport_v = shallow_water_transport_v(solution, formulation)
    return @inbounds c[i, j, k] * 1 / Azᶜᶜᶜ(i, j, k, grid) * horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, k, grid, transport_u, transport_v)
end

# Support for no advection
@inline c_div_Uc(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution, c, formulation) where FT = zero(FT)
