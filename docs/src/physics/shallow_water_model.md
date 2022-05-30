# Shallow water model

The [`ShallowWaterModel`](@ref) simulates the shallow water dynamics for a fluid of constant density
but with varying fluid depth ``h(x, y, t)``. The dynamics for the evolution of the two-dimensional
flow ``\boldsymbol{u}(x, y, t) = u(x, y, t) \boldsymbol{\hat x} + v(x, y, t) \boldsymbol{\hat y}``
and the fluid's height ``h(x, y, t)``:
```math
  \begin{align}
    \partial_t \boldsymbol{u} + \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \boldsymbol{u}
    + \boldsymbol{f} \times \boldsymbol{u} & = - g \boldsymbol{\nabla} h \, ,\\
    \partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} h \right ) & = 0 \, .
  \end{align}
```

[`ShallowWaterModel`](@ref) allow users to prescribe two different formulations for the dynamics:
`ConservativeFormulation()` and `VectorInvariantFormulation()`.

The `ConservativeFormulation()` uses the transport along each direction ``\boldsymbol{u} h = (uh, vh)`` and the total
depth of the fluid, ``h(x, y, t)``, as our dynamical variables.  The shallow water dynamics in conservative form:
```math
  \begin{align}
    \partial_t (\boldsymbol{u} h) + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} \boldsymbol{u} h \right ) + \boldsymbol{f} \times (\boldsymbol{u} h) & = - g \boldsymbol{\nabla} \left ( \frac1{2} h^2 \right ) \, ,\\
    \partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{u} h) & = 0 \, ,
  \end{align}
```
where ``\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} \boldsymbol{u} h \right )``
denotes a vector whose components are ``[\boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{u} \boldsymbol{u} h)]_i = \boldsymbol{\nabla} \boldsymbol{\cdot} (u_i \boldsymbol{u} h)``. We can retrieve the flow velocities
by dividing the corresponding transport by the fluid's height, e.g., `v = vh / h`.

The `VectorInvariantFormulation` uses the velocity ``\boldsymbol{u}=(u, v)`` and the total depth of the fluid, ``h``,
as our dynamical variables.  The shallow water dynamics in conservative form:

```math
\begin{align}
\partial_t \boldsymbol{u} + (\zeta + f) \boldsymbol{\hat k} \times\boldsymbol{u} & = 
- \boldsymbol{\nabla} \left( g h + \frac12 \boldsymbol{u} \cdot \boldsymbol{u} + g b \right) \, , \\
\partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} (h \boldsymbol{u}) & = 0 \, ,
\end{align}
```
where ``\zeta(x, y, t) =  \partial_x v - \partial_y u`` is the vertical component of the relative vorticity.

The elevation of the bottom bathymetry, measured with respect to the free-surface at rest, is ``b(x, y)``.
The free-surface elevation ``\eta`` is then:

```math
\eta(x, y, t) = h(x, y, t) + b(x, y) \, .
```
