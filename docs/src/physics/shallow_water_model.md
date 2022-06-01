# Shallow water model

The [`ShallowWaterModel`](@ref) simulates the shallow water dynamics for a fluid of constant density
but with varying fluid depth ``h(x, y, t)`` and whose velocity only varies in the horizontal,
``\boldsymbol{u}(x, y, t) = u(x, y, t) \boldsymbol{\hat x} + v(x, y, t) \boldsymbol{\hat y}``.

[`ShallowWaterModel`](@ref) allows users to prescribe the shallow water dynamics using two different formulations:
`VectorInvariantFormulation()` and `ConservativeFormulation()`.

The `VectorInvariantFormulation` uses the velocity ``\boldsymbol{u}=(u, v)`` and the total depth of the fluid, ``h``,
as the dynamical variables. Furthermore, the advective terms are rewritten via the vector identity:
```math
\boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \boldsymbol{u} = (\boldsymbol{\nabla} \times \boldsymbol{u}) \times \boldsymbol{u} + \boldsymbol{\nabla} \left( \frac1{2} \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{u} \right ) \, .
```
Thus, the shallow water dynamics in vector-invariant form become:

```math
\begin{align}
  \partial_t \boldsymbol{u} + (\zeta \boldsymbol{\hat k} + \boldsymbol{f})  \times\boldsymbol{u} & = 
  - \boldsymbol{\nabla} \left [ g (h +b) + \frac12 \boldsymbol{u} \cdot \boldsymbol{u} \right ] \, , \\
  \partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{u} h) & = 0 \, ,
\end{align}
```
where ``\zeta(x, y, t) =  \partial_x v - \partial_y u`` is the vertical component of the relative vorticity.

The elevation of the bottom bathymetry, measured with respect to the free-surface at rest, is ``b(x, y)``.
The free-surface elevation ``\eta`` is then:

```math
\eta(x, y, t) = h(x, y, t) + b(x, y) \, .
```

The `ConservativeFormulation()` uses the volume transport along each direction ``\boldsymbol{u} h = (u h, v h)`` and the total
depth of the fluid ``h`` as the dynamical variables.  The shallow water dynamics in conservative form is:
```math
\begin{align}
  \partial_t (\boldsymbol{u} h) + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} \boldsymbol{u} h \right ) + \boldsymbol{f} \times (\boldsymbol{u} h) & = - g h \boldsymbol{\nabla} \left ( h + b \right ) \, ,\\
  \partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{u} h) & = 0 \, ,
\end{align}
```
where ``\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} \boldsymbol{u} h \right )``
denotes a vector whose components are ``[\boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{u} \boldsymbol{u} h)]_i = \boldsymbol{\nabla} \boldsymbol{\cdot} (u_i \boldsymbol{u} h)``. We can retrieve the flow velocities
by dividing the corresponding transport by the fluid's height, e.g., `v = vh / h`.
