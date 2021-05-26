# Shallow Water Dynamics

The shallow water dynamics evolve the two-dimensional flow ``\boldsymbol{u}(x, y, t) = 
u(x, y, t) \boldsymbol{\hat x} + v(x, y, t) \boldsymbol{\hat y}`` together with the fluid's 
height ``h(x, y, t)`` according to:
```math
  \begin{align}
    \partial_t \boldsymbol{u} + \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \boldsymbol{u} 
    + \boldsymbol{f} \times \boldsymbol{u} & = - g \boldsymbol{\nabla} h \, ,\\
    \partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} h \right ) & = 0 \, .
  \end{align}
```

Using the transport along each direction ``\boldsymbol{u} h`` as our dynamical 
variables, we can express the shallow-water dynamics in conservative form:
```math
  \begin{align}
    \partial_t (\boldsymbol{u} h) + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} \boldsymbol{u} h \right ) + \boldsymbol{f} \times (\boldsymbol{u} h) & = - g \boldsymbol{\nabla} \left ( \frac1{2} h^2 \right ) \, ,\\
    \partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{u} h) & = 0 \, ,
  \end{align}
```
where ``\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} \boldsymbol{u} h \right )`` 
denotes a vector whose components are ``[\boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{u} \boldsymbol{u} h)]_i = \boldsymbol{\nabla} \boldsymbol{\cdot} (u_i \boldsymbol{u} h)``.

The `ShallowWaterModel` state variables are the transports, `uh` and `vh` and the fluid's 
height `h`. We can retrieve the flow velocities by dividing the corresponding transport by 
the fluid's height, e.g., `v = vh / h`.