# Shallow water model

The `ShallowWaterModel` simulates the shallow water dynamics for a fluid of constant density but
with varying fluid depth ``h(x, y, t)``. The dynamics for the evolution of the two-dimensional
flow ``\boldsymbol{u}(x, y, t) = u(x, y, t) \boldsymbol{\hat x} + v(x, y, t) \boldsymbol{\hat y}``
and the fluid's height ``h(x, y, t)`` is:
```math
  \begin{align}
    \partial_t \boldsymbol{u} + \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \boldsymbol{u}
    + \boldsymbol{f} \times \boldsymbol{u} & = - g \boldsymbol{\nabla} h \, ,\\
    \partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} h \right ) & = 0 \, .
  \end{align}
```

There are two different formulations that you can use: `ConservativeFormulation` and `VectorInvariantFormulation`

The `ConservativeFormulation` uses the transport along each direction ``\boldsymbol{u} h = (uh, vh)`` and the total
depth of the fluid, ``h``, as our dynamical variables.  The shallow water dynamics in conservative form:
```math
  \begin{align}
    \partial_t (\boldsymbol{u} h) + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} \boldsymbol{u} h \right ) + \boldsymbol{f} \times (\boldsymbol{u} h) & = - g \boldsymbol{\nabla} \left ( \frac1{2} h^2 \right ) \, ,\\
    \partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{u} h) & = 0 \, ,
  \end{align}
```
where ``\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} \boldsymbol{u} h \right )``
denotes a vector whose components are ``[\boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{u} \boldsymbol{u} h)]_i = \boldsymbol{\nabla} \boldsymbol{\cdot} (u_i \boldsymbol{u} h)``.  
We can retrieve the flow velocities by dividing the corresponding transport by the fluid's height, e.g., `v = vh / h`.

The `VectorInvariantFormulation` uses the velocity ``\boldsymbol{u}=(u, v)`` and the total depth of the fluid, ``h``,
as our dynamical variables.  The shallow water dynamics in conservative form:

```math
\begin{eqnarray*}
\partial_t \boldsymbol{u} + (\zeta + f)\boldsymbol{\hat k}  \times\boldsymbol{u}
& = & -\boldsymbol{\nabla} \left( g h + k + g b\right), \\
\partial_t h + \boldsymbol{\nabla}\cdot (h \boldsymbol{\nabla}) &=& =0,  \\
\end{eqnarray*}

```

Note that the vertical component of the relative vortocity, ``\zeta(x, y, t) =  \partial_x v - \partial_y u``,
and the horizontal component of the speed, ``k(x, y, t) = \frac12 \boldsymbol{u} \cdot \boldsymbol{u}``,  
are used in the momentum equation.  Also, ``b(x,y)`` is the bottom topography measured with respect to the
free-surface in the case of no motion and is negative.  The free-surface can be computed as
``\eta(x,y,t) = h(x, y, t) + b(x, y)``.
