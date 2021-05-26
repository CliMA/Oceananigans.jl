# Shallow Water Dynamics

The shallow water dynamics evolve the two-dimensional flow ``\boldsymbol{u}(x, y, t) = 
u \boldsymbol{\hat x} + v \boldsymbol{\hat y}`` together with the fluid height ``h(x, y, t)``. 

The shallow-water dynamics can be expressed in conservative form if we use the transport along 
each direction by ``U = u h`` and ``V = v h `` respectively,  as our dynamical variables:

```math
  \begin{align}
    \partial_t U + \partial_x \left ( \frac{U^2}{h} \right ) + \partial_y \left ( \frac{U V}{h} \right ) - f V & = -\partial_x \left ( \frac1{2} g h^2 \right ) \, ,\\
    \partial_t V + \partial_x \left ( \frac{U V}{h} \right ) + \partial_y \left ( \frac{V^2}{h} \right ) + f U & = -\partial_y \left ( \frac1{2} g h^2 \right ) \, ,\\
    \partial_t h + \partial_x U + \partial_y V & = 0 \, .
  \end{align}
```