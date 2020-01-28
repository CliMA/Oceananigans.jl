# The Boussinesq approximation

In Oceananigans.jl the fluid density $\rho$ is, in general, decomposed into three
components:
```math
    \rho(\bm{x}, t) = \rho_0 + \rho_*(z) + \rho'(\bm{x}, t) \, ,
```
where $\rho_0$ is a constant 'reference' density, $\rho_*(z)$ is a background density
profile typically associated with the hydrostatic compression of seawater in the deep ocean,
and $\rho'(\bm{x}, t)$ is the dynamic component of density corresponding to inhomogeneous
distributions of a buoyant tracer such as temperature or salinity.
The fluid *buoyancy*, associated with the buoyant acceleration of fluid, is
defined in terms of $\rho'$ as
```math
    b = - \frac{g \rho'}{\rho_0} \, ,
```
where $g$ is gravitational acceleration.

The Boussinesq approximation is valid when $\rho_* + \rho' \ll \rho_0$, which implies the
fluid is approximately *incompressible*[^2]
In this case, the mass conservation equation reduces to the continuity equation
```math
    \bm{\nabla} \bm{\cdot} \bm{u} = \partial_x u + \partial_y v + \partial_z w = 0 \, .
    \tag{eq:continuity}
```

[^2]: Incompressible fluids do not support acoustic waves.

## The momentum conservation equation

The equations governing the conservation of momentum in a rotating fluid, including buoyancy
via the Boussinesq approximation and including the averaged effects of surface gravity waves
at the top of the domain via the Craik-Leibovich approximation are
```math
    \partial_t \bm{u} + \left ( \bm{u} \bm{\cdot} \bm{\nabla} \right ) \bm{u}
        + \left ( \bm{f} - \bm{\nabla} \times \bm{u}^S \right ) \times \bm{u} = - \bm{\nabla} \phi + b \bm{\hat z}
        - \bm{\nabla} \bm{\cdot} \bm{\tau} - \partial_t \bm{u}^S + \bm{F_u} \, ,
    \tag{eq:momentum}
```
where $b$ is buoyancy, $\bm{\tau}$ is the kinematic stress tensor, $\bm{F_u}$
denotes an internal forcing of the velocity field $\bm{u}$, $\phi$ is the potential
associated with kinematic and constant hydrostatic contributions to pressure,
$\bm{u}^S$ is the 'Stokes drift' velocity field associated with surface gravity waves,
and $\bm{f}$ is *Coriolis parameter*, or the background vorticity associated with the
specified rate of rotation of the frame of reference.

## The tracer conservation equation

The conservation law for tracers in Oceananigans.jl is
```math
    \partial_t c + \bm{u} \bm{\cdot} \bm{\nabla} c = - \bm{\nabla} \bm{\cdot} \bm{q}_c + F_c \, ,
    \tag{eq:tracer}
```
where $\bm{q}_c$ is the diffusive flux of $c$ and $F_c$ is an arbitrary source term.
Oceananigans.jl permits arbitrary tracers and thus an arbitrary number of tracer
equations to be solved simultaneously with the momentum equations.
