# The Boussinesq approximation

Oceananigans.jl employ often the Boussinesq approximation[^1]. In the Boussinesq approximation
the fluid density ``\rho`` is, in general, decomposed into three components:
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
fluid is _approximately_ incompressible, and thus does not support acoustic waves. In this case, 
the mass conservation equation reduces to the continuity equation
```math
    \begin{equation}
    \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v} = \partial_x u + \partial_y v + \partial_z w = 0 \, .
    \label{eq:continuity}
    \end{equation}
```

[^1]: Named after Boussinesq (1903) although used earlier by Oberbeck (1879), the Boussinesq
      approximation neglects density differences in the momentum equation except when associated
      with the gravitational term. It is an accurate approximation for many flows, and especially
      so for oceanic flows where density differences are very small. See Vallis (2017, section 2.4)
      for an oceanographic introduction to the Boussinesq equations and Vallis (2017, Section 2.A)
      for an asymptotic derivation. See Kundu (2015, Section 4.9) for an engineering
      introduction.

