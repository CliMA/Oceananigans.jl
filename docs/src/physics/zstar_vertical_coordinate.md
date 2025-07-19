# Vertical coordinates

For [`HydrostaticFreeSurfaceModel`](@ref), we have the choice between `ZCoordinate` and a `ZStar` [generalized vertical coordinate](@ref generalized_vertical_coordinates).

The `ZStar` vertical coordinate conserves tracers and volume with the grid following the evolution of the free surface in the
domain [adcroft2004rescaled](@citep).

In terms of the notation in the [Generalized vertical coordinates](@ref generalized_vertical_coordinates) section,
for a `ZCoordinate` we have that
```math
r(x, y, z, t) = z
```
and the specific thickness is ``\sigma = \partial z / \partial r = 1``.

For `ZStar` vertical coordinate we have
```math
\begin{equation}
    r(x, y, z, t) = \frac{H(x, y)}{H(x, y) + \eta(x, y, t)}[z - \eta(x, y, t)]  \label{zstardef}
\end{equation}
```
where ``\eta`` is the free surface and ``z = -H(x, y)`` is the bottom of the domain.

![Schematic of the quantities involved in the definition of `ZStar` generalized vertical coordinate](assets/zstar-schematic.png)

The `ZStar` coordinate definition \eqref{zstardef} implies a specific thickness

```math
\sigma = 1 + \frac{\eta}{H}
```

All the equations transformed in ``r``-coordinates are described in the [Generalized vertical coordinates](@ref generalized_vertical_coordinates)
section. However, for the specific choice of `ZStar` coordinate \eqref{zstardef}, the ``\partial \eta/\partial r`` identically vanishes and
thus the horizontal gradient of the free surface remain unchanged under vertical coordinate transformation, i.e.,
```math
\begin{align}
    \frac{\partial \eta}{\partial x} \bigg\rvert_z & = \frac{\partial \eta}{\partial x} \bigg\rvert_r \\
    \frac{\partial \eta}{\partial y} \bigg\rvert_z & = \frac{\partial \eta}{\partial y} \bigg\rvert_r
\end{align}
```
