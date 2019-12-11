# Spatial operators

To calculate the various terms and perform the time-stepping, discrete difference and interpolation operators must be
designed from which all the terms, such as momentum advection and Laplacian diffusion, may be constructed. Much of the
material in this section is derived from \citet{Marshall97FV}.

## Differences

Difference operators act as the discrete form of the derivative operator. Care must be taken when calculating
differences on a staggered grid. For example, the the difference of a cell-centered variable such as temperature $T$
lies on the faces  in the direction of the difference, and vice versa. In principle, there are three difference
operators, one for each  direction
```math
  \delta_x f = f_E - f_W, \quad
  \delta_y f = f_N - f_S , \quad
  \delta_z f = f_T - f_B ,
```
where the $E$ and $W$ subscripts indicate that the value is evaluated the eastern or western wall of the cell, $N$ and
$S$ indicate the northern and southern walls, and $T$ and $B$ indicate the top and bottom walls.

Additionally, two $\delta$ operators must be defined for each direction to account for the staggered nature of the grid.
One for taking the difference of a cell-centered variable and projecting it onto the cell faces
```math
\begin{aligned}
    \delta_x^{faa} f_{i,j,k} &= f_{i,j,k} - f_{i-1,j,k} \\
    \delta_y^{afa} f_{i,j,k} &= f_{i,j,k} - f_{i,j-1,k} \\
    \delta_z^{aaf} f_{i,j,k} &= f_{i,j,k} - f_{i,j,k-1}
\end{aligned}
```
and another for taking the difference of a face-centered variable and projecting it onto the cell centers
```math
\begin{aligned}
    \delta_x^{caa} f_{i,j,k} &= f_{i+1,j,k} - f_{i,j,k} \\
    \delta_y^{aca} f_{i,j,k} &= f_{i,j+1,k} - f_{i,j,k} \\
    \delta_z^{aac} f_{i,j,k} &= f_{i,j,k+1} - f_{i,j,k}
\end{aligned}
```

## Interpolation

In order to add or multiply variables that are defined at different points they are interpolated. In our case, linear
interpolation or averaging is employed. Once again, there are two averaging operators, one for each direction,
\begin{equation}
  \overline{f}^x = \frac{f_E + f_W}{2} , \quad
  \overline{f}^y = \frac{f_N + f_S}{2} , \quad
  \overline{f}^z = \frac{f_T + f_B}{2}
\end{equation}

Additionally, three averaging operators must be defined for each direction. One for taking the average of a
cell-centered  variable and projecting it onto the cell faces
```math
\begin{aligned}
    \overline{f_{i,j,k}}^{faa} = \frac{f_{i,j,k} + f_{i-1,j,k}}{2} \\
    \overline{f_{i,j,k}}^{afa} = \frac{f_{i,j,k} + f_{i,j-1,k}}{2} \\
    \overline{f_{i,j,k}}^{aaf} = \frac{f_{i,j,k} + f_{i,j,k-1}}{2}
\end{aligned}
```
and another for taking the average of a face-centered variable and projecting it onto the cell centers
```math
\begin{aligned}
    \overline{f_{i,j,k}}^{caa} = \frac{f_{i+1,j,k} + f_{i,j,k}}{2} \\
    \overline{f_{i,j,k}}^{aca} = \frac{f_{i,j+1,k} + f_{i,j,k}}{2} \\
    \overline{f_{i,j,k}}^{aac} = \frac{f_{i,j,k+1} + f_{i,j,k}}{2}
\end{aligned}
```

## Divergence and flux divergence

The divergence of the flux of a cell-centered quantity over the cell can be calculated as
```math
\nabla \cdot \bm{f}
= \frac{1}{V} \left[ \delta_x^{faa} (A_x f_x)
                   + \delta_y^{afa} (A_y f_y)
                   + \delta_z^{aaf} (A_z f_z) \right]
```
where $\bm{f} = (f_x, f_y, f_z)$ is the flux with components defined normal to the faces, and $V$ is the volume of
the cell. The presence of a solid boundary is indicated by setting the appropriate flux normal to the boundary to zero.

A similar divergence operator can be defined for a face-centered quantity. The divergence of the flux of $T$ over a
cell,  $\nabla \cdot (\bm{u} T)$, required in the evaluation of $G_T$, for example, is then
```math
\renewcommand{\div}[1] {\nabla \cdot \left ( #1 \right )}
\div{\bm{u} T}
= \frac{1}{V} \left[ \delta_x^{caa} (A_x u \overline{T}^{faa})
                   + \delta_y^{aca} (A_y v \overline{T}^{afa})
                   + \delta_z^{aac} (A_z w \overline{T}^{aaf}) \right]
```
where $T$ is interpolated onto the cell faces where it can be multiplied by the velocities, which are then differenced
and  projected onto the cell centers where they added together and then added to $G_T$ which also lives at the cell
centers.

## Momentum advection

The advection terms that make up the $\mathbf{G}$ terms in equations \eqref{eq:horizontalMomentum} and
\eqref{eq:verticalMomentum} can be mathematically written as, e.g,
```math
\renewcommand{\div}[1] {\nabla \cdot \left ( #1 \right )}
\bm{u} \cdot \nabla u
    = \div{u\bm{u}} - u(\underbrace{\nabla\cdot\bm{u}}_{=0})
    = \div{u\bm{u}}
```
which can then be discretized similarly to the flux divergence operator, however, they must be discretized differently
for each direction.

For example, the $x$-momentum advection operator is discretized as
```math
\bm{u} \cdot \nabla u
= \frac{1}{\overline{V}^x} \left[
    \delta_x^{faa} \left( \overline{A_x u}^{caa} \overline{u}^{caa} \right)
  + \delta_y^{afa} \left( \overline{A_y v}^{aca} \overline{u}^{aca} \right)
  + \delta_z^{aaf} \left( \overline{A_z w}^{aac} \overline{u}^{aac} \right)
\right]
```
where $\overline{V}^x$ is the average of the volumes of the cells on either side of the face in question. Calculating
$\partial(uu)/\partial x$ can be performed by interpolating $A_x u$ and $u$ onto the cell centers then multiplying them
and differencing them back onto the faces. However, in the case of the the two other terms, $\partial(vu)/\partial y$
and $\partial(wu)/\partial z$, the two variables must be interpolated onto the cell edges to be multiplied then
differenced back onto the cell faces.

## Discretization of isotropic diffusion operators

An isotropic viscosity operator acting on vertical momentum is discretized via
```math
    \bm{\nabla} \left ( \nu_e \bm{\nabla} w \right )
    = \frac{1}{V} \left[
          \delta_x^{faa} \left( \nu_e \overline{A_x}^{caa} \delta_x^{caa} w \right)
        + \delta_y^{afa} \left( \nu_e \overline{A_y}^{aca} \delta_y^{aca} w \right)
        + \delta_z^{aaf} \left( \nu_e \overline{A_z}^{aac} \delta_z^{aac} w \right)
    \right ]
```
where $\nu$ is the kinematic viscosity.

An isotropic diffusion operator acting on a tracer $c$, on the other hand, is discretized via
```math
   \bm{\nabla} \bm{\cdot} \left ( \kappa_e \bm{\nabla} c \right ) =
    = \frac{1}{V} \left[
        \delta_x^{caa} \left( \kappa_e A_x \delta_x^{faa} c \right)
      + \delta_y^{aca} \left( \kappa_e A_y \delta_y^{afa} c \right)
      + \delta_z^{aac} \left( \kappa_e A_z \delta_z^{aaf} c \right)
    \right]
```
