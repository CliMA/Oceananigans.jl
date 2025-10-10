# [Hydrostatic model with a free surface](@id hydrostatic_free_surface_model)

The [`HydrostaticFreeSurfaceModel`](@ref) solves the incompressible Navier-Stokes equations under
the Boussinesq and hydrostatic approximations and with an arbitrary number of tracer conservation
equations. Physics associated with individual terms in the momentum and tracer conservation
equations -- the background rotation rate of the equation's reference frame,
gravitational effects associated with buoyant tracers under the Boussinesq
approximation, generalized stresses and tracer fluxes associated with viscous and
diffusive physics, and arbitrary "forcing functions" -- are determined by the whims of the
user.

## [Mass conservation and free surface evolution equation](@id hydrostatic_mass_conservation_free_surface)

The mass conservation equation is
```math
    0 = \boldsymbol{\nabla}_h \boldsymbol{\cdot} \boldsymbol{u} + \partial_z w \, .
```

Given the horizontal flow ``\boldsymbol{u}`` we use the above to diagnose the vertical velocity ``w``.
We integrate the mass conservation equation from the bottom of the fluid (where ``w = 0``) up to
depth ``z`` and recover ``w(x, y, z, t)``.

The free surface displacement ``\eta(x, y, t)`` satisfies the linearized kinematic boundary
condition at the surface
```math
    \partial_t \eta = w(x, y, z=0, t) \, .
```

## The momentum conservation equation

The equations governing the conservation of momentum in a rotating fluid, including buoyancy
via the Boussinesq approximation are
```math
    \begin{align}
    \partial_t \boldsymbol{u} & = - \left ( \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{u}
                        - \boldsymbol{f} \times \boldsymbol{u}
                        - \boldsymbol{\nabla}_h (p + g \eta)
                        - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau}
                        + \boldsymbol{F_u} \, , \label{eq:momentum}\\
    0 & = b - \partial_z p \, , \label{eq:hydrostatic}
    \end{align}
```
where ``b`` the is buoyancy, ``\boldsymbol{\tau}`` is the hydrostatic kinematic stress tensor,
``\boldsymbol{F_u}`` denotes an internal forcing of the horizontal flow ``\boldsymbol{u}``,
``\boldsymbol{v} = \boldsymbol{u} + w \hat{\boldsymbol{z}}`` is the three-dimensional flow,
``p`` is kinematic pressure, ``\eta`` is the free-surface displacement, and ``\boldsymbol{f}``
is the *Coriolis parameter*, or the background vorticity associated with the specified rate of
rotation of the frame of reference.

We can recast the advection term ``(\boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla}) \boldsymbol{u}``
above in vector-invariant form as:

```math
\left ( \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{u} = \zeta \hat{\boldsymbol{z}} \times \boldsymbol{u} + \boldsymbol{\nabla}\left(\frac1{2} \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{u}\right) + w \partial_z \boldsymbol{u}
```
with ``\zeta(x, y, t) = \partial_x v - \partial_y u`` the vertical component of the relative vorticity.
The vector-invariant form is used with curvilinear grids, like [`LatitudeLongitudeGrid`](@ref LatitudeLongitudeGrid) or [`OrthogonalSphericalShellGrid`](@ref).

The hydrostatic approximation \eqref{eq:hydrostatic} above comes about as the dominant balance
of terms in the Navier-Stokes vertical momentum equation under the Boussinesq approximation.

The terms that appear on the right-hand side of the momentum conservation equation are (in order):

* momentum advection: ``\left ( \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} \right )
  \boldsymbol{u}``,
* Coriolis: ``\boldsymbol{f} \times \boldsymbol{u}``,
* baroclinic kinematic pressure gradient: ``\boldsymbol{\nabla} p``,
* barotropic kinematic pressure gradient: ``\boldsymbol{\nabla} (g \eta)``,
* molecular or turbulence viscous stress: ``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau}``, and
* an arbitrary internal source of momentum: ``\boldsymbol{F_u}``.

## The tracer conservation equation

The conservation law for tracers is
```math
    \begin{align}
    \partial_t c = - \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} c
                   - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c
                   + F_c \, ,
    \label{eq:tracer}
    \end{align}
```
where ``\boldsymbol{q}_c`` is the diffusive flux of ``c`` and ``F_c`` is an arbitrary source term.
An arbitrary tracers are permitted and thus an arbitrary number of tracer equations
can be solved simultaneously alongside with the momentum equations.

From left to right, the terms that appear on the right-hand side of the tracer conservation
equation are

* tracer advection: ``\boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} c``,
* molecular or turbulent diffusion: ``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c``, and
* an arbitrary internal source of tracer: ``F_c``.

## Vertical coordinates

We can use either `ZCoordinate`, that is height coordinate, or the
`ZStarCoordinate` [generalized vertical coordinate](@ref generalized_vertical_coordinates).

The `ZStarCoordinate` vertical coordinate conserves tracers and volume with the grid following
 the evolution of the free surface in the domain [adcroft2004rescaled](@citep).

In terms of the notation in the [Generalized vertical coordinates](@ref generalized_vertical_coordinates)
section, for a `ZCoordinate` we have that
```math
r(x, y, z, t) = z
```
and the specific thickness is ``\sigma = \partial z / \partial r = 1``.

The `ZStarCoordinate` generalized vertical coordinate is often denoted as ``z^*`` (zee-star), i.e.,
```math
\begin{equation}
    r(x, y, z, t) = z^*(x, y, z, t) = \frac{H(x, y)}{H(x, y) + \eta(x, y, t)}[z - \eta(x, y, t)] \label{zstardef}
\end{equation}
```
where ``\eta`` is the free surface and ``z = -H(x, y)`` is the bottom of the domain.

![Schematic of the quantities involved in the ZStarCoordinate generalized vertical coordinate](../assets/zstar_schematic.png)

Note that in both depth and ``z^*`` coordinates, the bottom boundary is the same ``z = z^* = - H(x, y)``.
On the other hand, while in depth coordinates the upper boundary ``z = \eta(x, y, t)`` changes with time,
but in ``z^*`` coordinates is fixed to ``z^* = 0``.

The `ZStarCoordinate` definition \eqref{zstardef} implies a specific thickness

```math
\sigma = 1 + \frac{\eta}{H}
```

All the equations transformed in ``r``-coordinates are described in the
[Generalized vertical coordinates](@ref generalized_vertical_coordinates) section.

For the specific choice of `ZStarCoordinate` coordinate \eqref{zstardef}, the ``\partial \eta/\partial r``
identically vanishes and thus the horizontal gradient of the free surface remain unchanged under vertical
coordinate transformation, i.e.,
```math
\begin{align}
    \frac{\partial \eta}{\partial x} \bigg\rvert_z & = \frac{\partial \eta}{\partial x} \bigg\rvert_r \\
    \frac{\partial \eta}{\partial y} \bigg\rvert_z & = \frac{\partial \eta}{\partial y} \bigg\rvert_r
\end{align}
```

An example of how the vertical coordinate surfaces differ for `ZCoordinate` and the time-varying `ZStarCoordinate` coordinate is shown below.

```@example
using CairoMakie

Lx, Lz = 1e3, 25 # m

x = range(-Lx/2, stop=Lx/2, length=200)

σ = Lx/15 # a horizontal length scale

# bottom, H(x)
x₀, h₀ = -Lx/3,  15 # m
slope = @. h₀ * (1 + tanh(-(x - x₀) / σ)) / 2
x₀, h₀ = Lx/3, 6 # m
mountain = @. h₀ * sech((x - x₀) / σ)^2
H = @. Lz - slope - mountain

# free surface, η(x)
x₀ = -Lx/8
η₀ = 2.5 # m
t = Observable(0.0)
η = @lift @. -η₀ * ((x - x₀)^2 / σ^2 - 1) * exp(-(x - x₀)^2 / 2σ^2) * cos(2π * $t)

fig = Figure(size=(1000, 400))
axis_kwargs = (titlesize = 20, xlabel = "x", ygridvisible = false)
ax1 = Axis(fig[1, 1]; title="ZCoordinate", ylabel="z", axis_kwargs...)
ax2 = Axis(fig[1, 2]; title="ZStarCoordinate", axis_kwargs...)

for ax in (ax1, ax2)
    band!(ax, x, -H, η, color = (:dodgerblue, 0.5))
    band!(ax, x, -1.1 * Lz, -H, color = (:orange, 0.2))
    lines!(ax, x,  η, linewidth=5, color=:darkblue)
    lines!(ax, x, -H, linewidth=5, color=:darkgrey)
end

for r in range(-Lz, stop=0, length=6)
    # ZCoordinate
    z = r * ones(size(x))
    lines!(ax1, x, z, color=:crimson, linestyle=:dash)

    # ZStarCoordinate
    z = lift(η) do η_val
        @. r * (H + η_val) / H + η_val
    end
    lines!(ax2, x, z, color=:crimson, linestyle=:dash)
end

Nt = 50
times = 0:1/Nt:1-1/Nt # one period of cos(2πt)
CairoMakie.record(fig, "z-zstar.gif", times, framerate=12) do val
    t[] = val
end

nothing #hide
```

![](z-zstar.gif)

Near the top, the surfaces of `ZStarCoordinate` mimic the free surface.
Further away from the fluid's surface, the surfaces of `ZStarCoordinate` resemble more surfaces
of constant depth `ZCoordinate`.
