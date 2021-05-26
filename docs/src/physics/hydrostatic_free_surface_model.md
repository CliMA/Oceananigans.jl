# Hydrostatic Model with a Free Surface

`HydrostaticFreeSurfaceModel` solves the incompressible Navier-Stokes equations under the 
Boussinesq and hydrostatic approximations and with an arbitrary number of tracer conservation 
equations. Physics associated with individual terms in the momentum and tracer conservation
equations --- the background rotation rate of the equation's reference frame,
gravitational effects associated with buoyant tracers under the Boussinesq
approximation, generalized stresses and tracer fluxes associated with viscous and
diffusive physics, and arbitrary "forcing functions" --- are determined by the whims of the
user.

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
                        - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau}
                        + \partial_t \boldsymbol{u}^S
                        + \boldsymbol{F_u} \, , \label{eq:momentum}\\
                        0 & = b - \partial_z p \, , \label{eq:hydrostatic}
    \end{align}
```
where ``b`` the is buoyancy, ``\boldsymbol{\tau}`` is the kinematic stress tensor, ``\boldsymbol{F_u}``
denotes an internal forcing of the velocity field ``\boldsymbol{u}``, ``\phi`` is the potential
associated with kinematic and constant hydrostatic contributions to pressure, ``\boldsymbol{u}^S`` 
is the 'Stokes drift' velocity field associated with surface gravity waves, and ``\boldsymbol{f}`` 
is the *Coriolis parameter*, or the background vorticity associated with the specified rate of 
rotation of the frame of reference.

Equation~\eqref{eq:hydrostatic} is the hydrostatic approximation and comes about as the dominant 
balance of the Navier-Stokes vertical momentum equation under the Boussinesq approximation.

The terms that appear on the right-hand side of the momentum conservation equation are (in order):

* momentum advection: ``\left ( \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) 
  \boldsymbol{u}``,
* advection of resolved momentum by the background velocity field ``\boldsymbol{U}``: 
  ``\left ( \boldsymbol{U} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{u}``,
* advection of background momentum by resolved velocity: ``\left ( \boldsymbol{u} \boldsymbol{\cdot} 
  \boldsymbol{\nabla} \right ) \boldsymbol{U}``,
* Coriolis: ``\boldsymbol{f} \times \boldsymbol{u}``,
* the effective background rotation rate due to surface waves: ``\left ( \boldsymbol{\nabla} \times 
  \boldsymbol{u}^S \right ) \times \boldsymbol{u}``,
* pressure gradient: ``\boldsymbol{\nabla} \phi``,
* buoyant acceleration: ``b \boldsymbol{\hat z}``,
* molecular or turbulence viscous stress: ``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau}``,
* a source of momentum due to forcing or damping of surface waves: ``\partial_t \boldsymbol{u}^S``, and
* an arbitrary internal source of momentum: ``\boldsymbol{F_v}``.

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

* tracer advection: ``\boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} c``,
* tracer advection by the background velocity field, ``\boldsymbol{U}``: ``\boldsymbol{U} \boldsymbol{\cdot} \boldsymbol{\nabla} c``,
* advection of the background tracer field, ``C``, by the resolved velocity field: ``\boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} C``,
* molecular or turbulent diffusion: ``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c``, and
* an arbitrary internal source of tracer: ``F_c``.

The following subsections provide more details on the possible forms that each individual term 
in the momentum and tracer equations can take in Oceananigans.jl.
