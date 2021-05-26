# Shallow Water Dynamics

The shallow water dynamics evolve the two-dimensional flow ``\boldsymbol{u}(x, y, t) = 
u \boldsymbol{\hat x} + v \boldsymbol{\hat y}`` together with the fluid height ``h(x, y, t)``. 

The shallow-water dynamics can be expressed in conservative form if we use the transport along 
each direction by ``\boldsymbol{U} = \boldsymbol{u} h``,  as our dynamical variables:

```math
  \begin{align}
    \partial_t \boldsymbol{U} + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{U} \frac{\boldsymbol{U}}{h} \right ) + \boldsymbol{f} \times \boldsymbol{U} = - \boldsymbol{\nabla} \left ( \frac1{2} g h^2 \right ) \, ,\\
    \partial_t h + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{U} & = 0 \, .
  \end{align}
```

Above, notation ``\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{U} \frac{\boldsymbol{U}}{h} \right )`` denotes a vector whose components are:

```math
  [ \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{U} \frac{\boldsymbol{U}}{h} \right ) ]_i = \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{U} \frac{U_i}{h} \right ) \, .
```