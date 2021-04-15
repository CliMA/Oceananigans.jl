# Coordinate system and notation

Oceananigans.jl is formulated in a Cartesian coordinate system
``\boldsymbol{x} = (x, y, z)`` with unit vectors ``\boldsymbol{\hat x}``, ``\boldsymbol{\hat y}``, and ``\boldsymbol{\hat z}``,
where ``\boldsymbol{\hat x}`` points east, ``\boldsymbol{\hat y}`` points north, and ``\boldsymbol{\hat z}`` points 'upward',
opposite the direction of gravitational acceleration.
We denote time with ``t``, partial derivatives with respect to time ``t`` or a coordinate ``x``
with ``\partial_t`` or ``\partial_x``, and denote the gradient operator
``\boldsymbol{\nabla} \equiv \partial_x \boldsymbol{\hat x} + \partial_y \boldsymbol{\hat y} + \partial_z \boldsymbol{\hat z}``.
We use ``u``, ``v``, and ``w`` to denote the east, north, and vertical velocity components,
such that ``\boldsymbol{u} = u \boldsymbol{\hat x} + v \boldsymbol{\hat y} + w \boldsymbol{\hat z}``.

# The Boussinesq Navier-Stokes equations and tracer conservation equations

Oceananigans.jl solves the incompressible Navier-Stokes equations under the Boussinesq
approximation[^1] and an arbitrary number of tracer conservation equations.
Physics associated with individual terms in the momentum and tracer conservation
equations --- the background rotation rate of the equation's reference frame,
gravitational effects associated with buoyant tracers under the Boussinesq
approximation, generalized stresses and tracer fluxes associated with viscous and
diffusive physics, and arbitrary "forcing functions" --- are determined by the whims of the
user.

[^1]: Named after Boussinesq (1903) although used earlier by Oberbeck (1879), the Boussinesq
      approximation neglects density differences in the momentum equation except when associated
      with the gravitational term. It is an accurate approximation for many flows, and especially
      so for oceanic flows where density differences are very small. See Vallis (2017, section 2.4)
      for an oceanographic introduction to the Boussinesq equations and Vallis (2017, Section 2.A)
      for an asymptotic derivation. See Kundu (2015, Section 4.9) for an engineering
      introduction.

The fluid density ``\rho`` in Oceananigans.jl is, in general, decomposed into three
components:
```math
    \rho(\boldsymbol{x}, t) = \rho_0 + \rho_*(z) + \rho'(\boldsymbol{x}, t) \, ,
```
where ``\rho_0`` is a constant 'reference' density, ``\rho_*(z)`` is a background density
profile which, when non-zero, is typically associated with the hydrostatic compression
of seawater in the deep ocean, and ``\rho'(\boldsymbol{x}, t)`` is the dynamic component of density
corresponding to inhomogeneous distributions of a buoyant tracer such as temperature or salinity.
The fluid *buoyancy*, associated with the buoyant acceleration of fluid, is
defined in terms of ``\rho'`` as
```math
    b = - \frac{g \rho'}{\rho_0} \, ,
```
where ``g`` is gravitational acceleration.

The Boussinesq approximation is valid when ``\rho_* + \rho' \ll \rho_0``, which implies the
fluid is _approximately_ incompressible, and thus does not support acoustic waves.
In this case, the mass conservation equation reduces to the continuity equation
```math
    \begin{equation}
    \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{u} = \partial_x u + \partial_y v + \partial_z w = 0 \, .
    \label{eq:continuity}
    \end{equation}
```

## The momentum conservation equation

The equations governing the conservation of momentum in a rotating fluid, including buoyancy
via the Boussinesq approximation and including the averaged effects of surface gravity waves
at the top of the domain via the Craik-Leibovich approximation are
```math
    \begin{align}
    \partial_t \boldsymbol{u} & = - \left ( \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{u}
                        - \left ( \boldsymbol{U} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{u}
                        - \left ( \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{U} \nonumber \\
                        & \qquad
                        - \left ( \boldsymbol{f} - \boldsymbol{\nabla} \times \boldsymbol{u}^S \right ) \times \boldsymbol{u} 
                        - \boldsymbol{\nabla} \phi
                        + b \boldsymbol{\hat z}
                        - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau}
                        + \partial_t \boldsymbol{u}^S
                        + \boldsymbol{F_u} \, ,
    \label{eq:momentum}
    \end{align}
```
where ``b`` the is buoyancy, ``\boldsymbol{\tau}`` is the kinematic stress tensor, ``\boldsymbol{F_u}``
denotes an internal forcing of the velocity field ``\boldsymbol{u}``, ``\phi`` is the potential
associated with kinematic and constant hydrostatic contributions to pressure, ``\boldsymbol{u}^S`` 
is the 'Stokes drift' velocity field associated with surface gravity waves, and ``\boldsymbol{f}`` 
is the *Coriolis parameter*, or the background vorticity associated with the specified rate of 
rotation of the frame of reference.

The terms that appear on the right-hand side of the momentum conservation equation are (in order):

* momentum advection, ``\left ( \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) 
  \boldsymbol{u}``,
* advection of resolved momentum by the background velocity field ``\boldsymbol{U}``, 
  ``\left ( \boldsymbol{U} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{u}``,
* advection of background momentum by resolved velocity, ``\left ( \boldsymbol{u} \boldsymbol{\cdot} 
  \boldsymbol{\nabla} \right ) \boldsymbol{U}``,
* coriolis, ``\boldsymbol{f} \times \boldsymbol{u}``,
* the effective background rotation rate due to surface waves, ``\left ( \boldsymbol{\nabla} \times 
  \boldsymbol{u}^S \right ) \times \boldsymbol{u}``,
* pressure gradient, ``\boldsymbol{\nabla} \phi``,
* buoyant acceleration, ``b \boldsymbol{\hat z}``,
* molecular or turbulence viscous stress, ``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau}``,
* a source of momentum due to forcing or damping of surface waves, ``\partial_t \boldsymbol{u}^S``, and
* an arbitrary internal source of momentum, ``\boldsymbol{F_u}``.

## The tracer conservation equation

The conservation law for tracers in Oceananigans.jl is
```math
    \begin{align}
    \partial_t c = - \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} c
                   - \boldsymbol{U} \boldsymbol{\cdot} \boldsymbol{\nabla} c
                   - \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} C
                   - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c
                   + F_c \, ,
    \label{eq:tracer}
    \end{align}
```
where ``\boldsymbol{q}_c`` is the diffusive flux of ``c`` and ``F_c`` is an arbitrary source term.
Oceananigans.jl permits arbitrary tracers and thus an arbitrary number of tracer equations to 
be solved simultaneously with the momentum equations.

From left to right, the terms that appear on the right-hand side of the tracer conservation equation are

* tracer advection, ``\boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} c``,
* tracer advection by the background velocity field, ``U``, ``\boldsymbol{U} \boldsymbol{\cdot} \boldsymbol{\nabla} c``,
* advection of the background tracer field, ``C``, by the resolved velocity field, ``\boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} C``,
* molecular or turbulent diffusion, ``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c``, and
* an arbitrary internal source of tracer, ``F_c``.

The following subsections provide more details on the possible forms that each individual term 
in the momentum and tracer equations can take in Oceananigans.jl.
