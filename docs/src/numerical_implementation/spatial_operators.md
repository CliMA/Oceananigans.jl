# Spatial operators

To calculate the various terms and perform the time-stepping, discrete difference and interpolation 
operators must be designed from which all the terms, such as momentum advection and Laplacian 
diffusion, may be constructed. Much of the material in this section is derived from [Marshall97FV](@cite).

## Differences

Difference operators act as the discrete form of the derivative operator. Care must be taken 
when calculating differences on a staggered grid. For example, the the difference of a cell-centered 
variable such as temperature ``T`` lies on the faces  in the direction of the difference, and 
vice versa. In principle, there are three difference operators, one for each  direction
```math
  \delta_x f = f_E - f_W , \quad
  \delta_y f = f_N - f_S , \quad
  \delta_z f = f_T - f_B ,
```
where the ``E`` and ``W`` subscripts indicate that the value is evaluated the eastern or western 
wall of the cell, ``N`` and ``S`` indicate the northern and southern walls, and ``T`` and ``B`` 
indicate the top and bottom walls.

Additionally, two ``\delta`` operators must be defined for each direction to account for the 
staggered nature of the grid. One for taking the difference of a cell-centered variable and 
projecting it onto the cell faces
```math
\begin{align}
    \delta_x^{faa} f_{i, j, k} &= f_{i, j, k} - f_{i-1, j, k} \, , \\
    \delta_y^{afa} f_{i, j, k} &= f_{i, j, k} - f_{i, j-1, k} \, , \\
    \delta_z^{aaf} f_{i, j, k} &= f_{i, j, k} - f_{i, j, k-1} \, , 
\end{align}
```
and another for taking the difference of a face-centered variable and projecting it onto the cell centers
```math
\begin{align}
    \delta_x^{caa} f_{i, j, k} &= f_{i+1, j, k} - f_{i, j, k} \, , \\
    \delta_y^{aca} f_{i, j, k} &= f_{i, j+1, k} - f_{i, j, k} \, , \\
    \delta_z^{aac} f_{i, j, k} &= f_{i, j, k+1} - f_{i, j, k} \, .
\end{align}
```

## Interpolation

In order to add or multiply variables that are defined at different points they are interpolated. 
In our case, linear interpolation or averaging is employed. Once again, there are two averaging 
operators, one for each direction,
```math
\begin{equation}
  \overline{f}^x = \frac{f_E + f_W}{2} \, , \quad
  \overline{f}^y = \frac{f_N + f_S}{2} \, , \quad
  \overline{f}^z = \frac{f_T + f_B}{2} \, .
\end{equation}
```

Additionally, three averaging operators must be defined for each direction. One for taking the 
average of a cell-centered  variable and projecting it onto the cell faces
```math
\begin{align}
    \overline{f_{i, j, k}}^{faa} = \frac{f_{i, j, k} + f_{i-1, j, k}}{2} \, , \\
    \overline{f_{i, j, k}}^{afa} = \frac{f_{i, j, k} + f_{i, j-1, k}}{2} \, , \\
    \overline{f_{i, j, k}}^{aaf} = \frac{f_{i, j, k} + f_{i, j, k-1}}{2} \, ,
\end{align}
```
and another for taking the average of a face-centered variable and projecting it onto the cell centers
```math
\begin{align}
    \overline{f_{i, j, k}}^{caa} = \frac{f_{i+1, j, k} + f_{i, j, k}}{2} \, , \\
    \overline{f_{i, j, k}}^{aca} = \frac{f_{i, j+1, k} + f_{i, j, k}}{2} \, , \\
    \overline{f_{i, j, k}}^{aac} = \frac{f_{i, j, k+1} + f_{i, j, k}}{2} \, .
\end{align}
```

## Divergence and flux divergence

The divergence of the flux of a cell-centered quantity over the cell can be calculated as
```math
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{f}
= \frac{1}{V} \left[ \delta_x^{faa} (A_x f_x)
                   + \delta_y^{afa} (A_y f_y)
                   + \delta_z^{aaf} (A_z f_z) \right] \, ,
```
where ``\boldsymbol{f} = (f_x, f_y, f_z)`` is the flux with components defined normal to the 
faces, and ``V`` is the volume of the cell. The presence of a solid boundary is indicated by 
setting the appropriate flux normal to the boundary to zero.

A similar divergence operator can be defined for a face-centered quantity. The divergence of 
the flux of ``T`` over a cell,  ``\boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{v} T)``, 
required in the evaluation of ``G_T``, for example, is then
```math
\renewcommand{\div}[1] {\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( #1 \right )}
\div{\boldsymbol{v} T}
= \frac{1}{V} \left[ \delta_x^{caa} (A_x u \overline{T}^{faa})
                   + \delta_y^{aca} (A_y v \overline{T}^{afa})
                   + \delta_z^{aac} (A_z w \overline{T}^{aaf}) \right] \, ,
```
where ``T`` is interpolated onto the cell faces where it can be multiplied by the velocities, 
which are then differenced and  projected onto the cell centers where they added together and 
then added to ``G_T`` which also lives at the cell centers.

## Momentum advection

The advection terms that make up the ``\mathbf{G}`` terms in equations \eqref{eq:horizontalMomentum} and
\eqref{eq:verticalMomentum} can be rewritten using the incompressibility (``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v} = 0``) 
as, e.g,
```math
\renewcommand{\div}[1] {\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( #1 \right )}
\begin{align}
\boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} u & = \div{u \boldsymbol{v}} - u ( \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v} ) \nonumber \\
    & = \div{u \boldsymbol{v}} \, ,
\end{align}
```
which can then be discretized similarly to the flux divergence operator, however, they must 
be discretized differently for each direction.

For example, the ``x``-momentum advection operator is discretized as
```math
\boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} u
= \frac{1}{\overline{V}^x} \left[
    \delta_x^{faa} \left( \overline{A_x u}^{caa} \overline{u}^{caa} \right)
  + \delta_y^{afa} \left( \overline{A_y v}^{aca} \overline{u}^{aca} \right)
  + \delta_z^{aaf} \left( \overline{A_z w}^{aac} \overline{u}^{aac} \right)
\right] \, ,
```
where ``\overline{V}^x`` is the average of the volumes of the cells on either side of the face 
in question. Calculating ``\partial(uu)/\partial x`` can be performed by interpolating ``A_x u`` 
and ``u`` onto the cell centers then multiplying them and differencing them back onto the faces. 
However, in the case of the the two other terms, ``\partial(vu)/\partial y`` and ``\partial(wu)/\partial z``, 
the two variables must be interpolated onto the cell edges to be multiplied then differenced 
back onto the cell faces.

## Discretization of isotropic diffusion operators

An isotropic viscosity operator acting on vertical momentum is discretized via
```math
    \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \nu_e \boldsymbol{\nabla} w \right )
    = \frac{1}{V} \left[
          \delta_x^{faa} ( \nu_e \overline{A_x}^{caa} \partial_x^{caa} w )
        + \delta_y^{afa} ( \nu_e \overline{A_y}^{aca} \partial_y^{aca} w )
        + \delta_z^{aaf} ( \nu_e \overline{A_z}^{aac} \partial_z^{aac} w )
    \right ] \, ,
```
where ``\nu`` is the kinematic viscosity.

An isotropic diffusion operator acting on a tracer ``c``, on the other hand, is discretized via
```math
   \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \kappa_e \boldsymbol{\nabla} c \right )
    = \frac{1}{V} \left[ \phantom{\overline{A_x}^{caa}}
        \delta_x^{caa} ( \kappa_e A_x \partial_x^{faa} c )
      + \delta_y^{aca} ( \kappa_e A_y \partial_y^{afa} c )
      + \delta_z^{aac} ( \kappa_e A_z \partial_z^{aaf} c )
    \right] \, .
```

## Vertical integrals
Vertical integrals are converted into sums along each column. For example, the hydrostatic pressure 
anomaly is
```math
    p_{HY}^\prime = \int_{-L_z}^0 b^\prime \, \mathrm{d} z \, ,
```
where ``b^\prime`` is the buoyancy perturbation. Converting it into a sum that we compute from 
the top downwards we get
```math
    \begin{equation}
    p_{HY}^\prime(k) =
        \begin{cases}
            - \overline{b_{N_z}^\prime}^{aaf} \Delta z^F_{N_z},               & \quad k = N_z \, , \\
            p_{HY}^\prime(k+1) - \overline{b_{k+1}^\prime}^{aaf} \Delta z^F_k, & \quad 1 \le k \le N_z - 1 \, ,
        \end{cases}
    \end{equation}
```
where we converted the sum into a recursive definition for ``p_{HY}^\prime(k)`` in terms of 
``p_{HY}^\prime(k+1)`` so that the integral may be computed with ``\mathcal{O}(N_z)`` operations 
by a single thread.

The vertical velocity ``w`` may be computed from ``u`` and ``v`` via the continuity equation
```math
    w = - \int_{-L_z}^0 (\partial_x u + \partial_y v) \, \mathrm{d} z \, ,
```
to satisfy the incompressibility condition ``\nabla\cdot\boldsymbol{v} = 0`` to numerical precision. 
This also involves computing a vertical integral, in this case evaluated from the bottom up
```math
    \begin{equation}
    w_k =
        \begin{cases}
            0, & \quad k = 1 \, , \\
            w_{k-1} - \left( \partial_x^{caa} u + \partial_y^{aca} v \right) \Delta z^C_k, & \quad 2 \le k \le N_z \, .
        \end{cases}
    \end{equation}
```
