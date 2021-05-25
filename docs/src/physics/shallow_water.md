# Shallow Water Dynamis

The shallow water dynamics evolve the two-dimensional flow ``\boldsymbol{u}_h(x, y, t) = 
u \boldsymbol{\hat x} + v \boldsymbol{\hat y}`` together with the fluid height ``h(x, y, t)``. 

By denoting the transport along each direction by ``U = u h`` and ``V = v h `` respectively, 
the shallow-water dyamics can be written in conservative form:

```math
  \begin{align}
    \partial_t U + \partial_x \left ( \frac{U^2}{h} \right ) + \partial_y \left ( \frac{U V}{h} \right ) - f V & = -\partial_x \left ( \frac1{2} g h^2 \right ) \, ,\\
    \partial_t V + \partial_x \left ( \frac{U V}{h} \right ) + \partial_y \left ( \frac{V^2}{h} \right ) + f U & = -\partial_y \left ( \frac1{2} g h^2 \right ) \, ,\\
    \partial_t h + \partial_x U + \partial_y V & = 0 \, .
  \end{align}
```