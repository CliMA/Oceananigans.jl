# Shallow Water Dynamis

The shallow water dynamics evolve:

```math
  \begin{align}
    \partial_t U + \partial_x \left ( \frac{U^2}{h}  \right ) + \partial_y \left ( \frac{U V}{h}  \right ) - f V & = -\partial_x \left ( \frac{g' h^2}{2} \right ) \, .\\
    \partial_t V + \partial_x \left ( \frac{U V}{h}  \right ) + \partial_y \left ( \frac{V^2}{h}  \right ) + f U & = -\partial_y \left ( \frac{g' h^2}{2} \right ) \, ,\\
    \partial_t h + \partial_x U + \partial_y V & = 0 \, .
```