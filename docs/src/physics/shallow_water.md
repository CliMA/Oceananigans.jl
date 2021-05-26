# Shallow Water Dynamics

The shallow water dynamics evolve the two-dimensional flow ``\boldsymbol{u}(x, y, t) = 
u \boldsymbol{\hat x} + v \boldsymbol{\hat y}`` together with the fluid height ``h(x, y, t)``
according to:

```math
  \begin{align}
    \partial_t \boldsymbol{u} + \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \boldsymbol{u} 
    + \boldsymbol{f} \times \boldsymbol{u} & = - g \boldsymbol{\nabla} h \, ,\\
    \partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} h \right ) & = 0 \, .
  \end{align}
```

Using the transport along each direction ``\boldsymbol{U} = \boldsymbol{u} h`` as our dynamical 
variables, we can express the shallow-water dynamics in conservative form:

```math
  \begin{align}
    \partial_t \boldsymbol{U} + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{U} \frac{\boldsymbol{U}}{h} \right ) + \boldsymbol{f} \times \boldsymbol{U} & = - \boldsymbol{\nabla} \left ( \frac1{2} g h^2 \right ) \, ,\\
    \partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{U} & = 0 \, ,
  \end{align}
```

where ``\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{U} \boldsymbol{U} / h \right )`` 
denotes a vector whose components are:

```math
  \left [ \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{U} \frac{\boldsymbol{U}}{h} \right ) \right ]_i 
  = \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{U} \frac{U_i}{h} \right ) \, .
```
