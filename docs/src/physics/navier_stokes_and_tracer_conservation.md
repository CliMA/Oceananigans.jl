# The Boussinesq Navier-Stokes equations and tracer conservation equations

The `IncompressibleModel` solves the incompressible Navier-Stokes equations under the Boussinesq
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

The fluid density ``\rho`` is, in general, decomposed into three
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
    \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v} = \partial_x u + \partial_y v + \partial_z w = 0 \, .
    \label{eq:continuity}
    \end{equation}
```

## The momentum conservation equation

The equations governing the conservation of momentum in a rotating fluid, including buoyancy
via the Boussinesq approximation and including the averaged effects of surface gravity waves
at the top of the domain via the Craik-Leibovich approximation are
```math
    \begin{align}
    \partial_t \boldsymbol{v} & = - \left ( \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{v}
                        - \left ( \boldsymbol{V} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{v}
                        - \left ( \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{V} \nonumber \\
                        & \qquad
                        - \left ( \boldsymbol{f} - \boldsymbol{\nabla} \times \boldsymbol{v}^S \right ) \times \boldsymbol{v} 
                        - \boldsymbol{\nabla} \phi
                        + b \boldsymbol{\hat z}
                        - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau}
                        + \partial_t \boldsymbol{v}^S
                        + \boldsymbol{F_v} \, ,
    \label{eq:momentum}
    \end{align}
```
where ``b`` the is buoyancy, ``\boldsymbol{\tau}`` is the kinematic stress tensor, ``\boldsymbol{F_v}``
denotes an internal forcing of the velocity field ``\boldsymbol{v}``, ``\phi`` is the potential
associated with kinematic and constant hydrostatic contributions to pressure, ``\boldsymbol{v}^S`` 
is the 'Stokes drift' velocity field associated with surface gravity waves, and ``\boldsymbol{f}`` 
is the *Coriolis parameter*, or the background vorticity associated with the specified rate of 
rotation of the frame of reference.

The terms that appear on the right-hand side of the momentum conservation equation are (in order):

* momentum advection: ``\left ( \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) 
  \boldsymbol{v}``,
* advection of resolved momentum by the background velocity field ``\boldsymbol{V}``: 
  ``\left ( \boldsymbol{V} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{v}``,
* advection of background momentum by resolved velocity: ``\left ( \boldsymbol{v} \boldsymbol{\cdot} 
  \boldsymbol{\nabla} \right ) \boldsymbol{V}``,
* coriolis: ``\boldsymbol{f} \times \boldsymbol{v}``,
* the effective background rotation rate due to surface waves: ``\left ( \boldsymbol{\nabla} \times 
  \boldsymbol{v}^S \right ) \times \boldsymbol{v}``,
* pressure gradient: ``\boldsymbol{\nabla} \phi``,
* buoyant acceleration: ``b \boldsymbol{\hat z}``,
* molecular or turbulence viscous stress: ``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau}``,
* a source of momentum due to forcing or damping of surface waves: ``\partial_t \boldsymbol{v}^S``, and
* an arbitrary internal source of momentum: ``\boldsymbol{F_v}``.

## The tracer conservation equation

The conservation law for tracers in Oceananigans.jl is
```math
    \begin{align}
    \partial_t c = - \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} c
                   - \boldsymbol{V} \boldsymbol{\cdot} \boldsymbol{\nabla} c
                   - \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} C
                   - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c
                   + F_c \, ,
    \label{eq:tracer}
    \end{align}
```
where ``\boldsymbol{q}_c`` is the diffusive flux of ``c`` and ``F_c`` is an arbitrary source term.
Oceananigans.jl permits arbitrary tracers and thus an arbitrary number of tracer equations to 
be solved simultaneously with the momentum equations.

From left to right, the terms that appear on the right-hand side of the tracer conservation equation are

* tracer advection: ``\boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} c``,
* tracer advection by the background velocity field, ``\boldsymbol{V}``: ``\boldsymbol{V} \boldsymbol{\cdot} \boldsymbol{\nabla} c``,
* advection of the background tracer field, ``C``, by the resolved velocity field: ``\boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} C``,
* molecular or turbulent diffusion: ``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c``, and
* an arbitrary internal source of tracer: ``F_c``.

The following subsections provide more details on the possible forms that each individual term 
in the momentum and tracer equations can take in Oceananigans.jl.
