# [Numerical implementation of boundary conditions](@id numerical_bcs)

We adopt a mixed approach for implementing boundary conditions that uses both halo regions and "direct"
imposition of boundary conditions, depending on the condition prescribed.

We illustrate how boundary conditions are implemented by considering the tracer equation
```math
    \begin{align}
    \partial_t c = - \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} c
                   - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c
                   + F_c \, ,
    \label{eq:tracer}
    \end{align}
```
where ``\boldsymbol{q}_c`` is the diffusive flux of ``c`` and ``F_c`` is an arbitrary source term.

See [Model setup: boundary conditions](@ref model_step_bcs) for how to create and use these
boundary conditions in Oceananigans.

## Gradient boundary conditions

Users impose gradient boundary conditions by prescribing the gradient ``\gamma`` of a field
``c`` across an *external boundary* ``\partial \Omega_b``. The prescribed gradient ``\gamma``
may be a constant, discrete array of values, or an arbitrary function. The gradient boundary
condition is enforced setting the value of halo points located outside the domain interior
such that
```math
    \begin{equation}
    \label{eq:gradient-bc}
    \hat{\boldsymbol{n}} \boldsymbol{\cdot} \boldsymbol{\nabla} c |_{\partial \Omega_b} = \gamma \, .
    \end{equation}
```
where ``\hat{\boldsymbol{n}}`` is the vector normal to ``\partial \Omega_b``.

Across the bottom boundary in ``z``, for example, this requires that
```math
    \begin{equation}
    \label{eq:linear-extrapolation}
    c_{i, j, 0} = c_{i, j, 1} + \gamma_{i, j, 1} \tfrac{1}{2} \left ( \Delta z_{i, j, 1} + \Delta z_{i, j, 0} \right ) \, ,
    \end{equation}
```
where ``\Delta z_{i, j, 1} = \Delta z_{i, j, 0}`` are the heights of the finite volume at ``i, j`` and ``k=1`` and ``k=0``.
This prescription implies that the ``z``-derivative of ``c`` across the boundary at ``k=1`` is
```math
    \begin{equation}
    \partial_z c \, |_{i, j, 1} \equiv
        \frac{c_{i, j, 1} - c_{i, j, 0}}{\tfrac{1}{2} \left ( \Delta z_{i, j, 1} + \Delta z_{i, j, 0} \right )}
            = \gamma_{i, j, 1} \, ,
    \end{equation}
```
as prescribed by the user.

Gradient boundary conditions are represented by the [`Gradient`](@ref) type.

## Value boundary conditions

Users impose value boundary conditions by prescribing ``c^b``, the value of ``c`` on the external
boundary ``\partial \Omega_b``.
The value ``c^b`` may be a constant, array of discrete values, or an arbitrary function.
To enforce a value boundary condition, the gradient associated with the difference between
``c^b`` and ``c`` at boundary-adjacent nodes is diagnosed and used to set the value of the ``c`` halo point
located outside the boundary.

At the bottom boundary in ``z``, for example, this means that the gradient of ``c`` is determined by
```math
    \begin{equation}
    \gamma = \frac{c_{i, j, 1} - c^b_{i, j, 1}}{\tfrac{1}{2} \Delta z_{i, j, 1}} \, ,
    \end{equation}
```
which is then used to set the halo point ``c_{i, j, 0}`` via linear extrapolation.

Value boundary conditions are represented by the [`Value`](@ref) type.

## Flux boundary conditions

Users impose flux boundary conditions by prescribing the flux ``q_c \, |_b`` of ``c`` across
the external boundary ``\partial \Omega_b``. The flux ``q_c \, |_b`` may be a constant, array
of discrete values, or arbitrary function. To explain how flux boundary conditions are imposed
in `Oceananigans.jl`, we note that the average of the tracer conservation equation over a finite
volume yields
```math
    \begin{equation}
    \label{eq:dc/dt}
    \partial_t c_{i, j, k} = - \frac{1}{V_{i, j, k}} \oint_{\partial \Omega_{i, j, k}} (\boldsymbol{v} c + \boldsymbol{q}_c)
                                                     \boldsymbol{\cdot} \hat{\boldsymbol{n}}  \, \mathrm{d} S
                             + \frac{1}{V_{i, j, k}} \int_{V_{i, j, k}} F_c \, \mathrm{d} V \, ,
    \end{equation}
```
where the surface integral over ``\partial \Omega_{i, j, k}`` averages the flux of ``c`` across
the six faces of the finite volume. The right-hand-side of \eqref{eq:dc/dt} above is denoted as
``G_c |_{i, j, k}``.


An external boundary of a finite volume is associated with a no-penetration condition such that
``\hat{\boldsymbol{n}} \boldsymbol{\cdot} \boldsymbol{v} \, |_{\partial \Omega_b} = 0``, where
``\hat{\boldsymbol{n}}`` is the vector normal to ``\partial \Omega_b``. Furthermore, the closures
currently available in `Oceananigans.jl` have the property that ``\boldsymbol{q}_c \propto \boldsymbol{\nabla} c``.
Thus setting ``\hat{\boldsymbol{n}} \boldsymbol{\cdot} \boldsymbol{\nabla} c \, |_{\partial \Omega_b} = 0``
on the external boundary implies that the total flux of ``c`` across the external boundary is
```math
    \begin{equation}
    \hat{\boldsymbol{n}} \boldsymbol{\cdot} \left ( \boldsymbol{v} c + \boldsymbol{q}_c \right ) |_{\partial \Omega_b} = 0 \, .
    \end{equation}
```
`Oceananigans.jl` exploits this fact to define algorithm that prescribe fluxes across external
boundaries ``\partial \Omega_b``:

1. Impose a constant gradient ``\hat{\boldsymbol{n}} \boldsymbol{\cdot} \boldsymbol{\nabla} c
   \, |_{\partial \Omega_b} = 0`` across external boundaries via using halo points (similar
   to \eqref{eq:gradient-bc}), which ensures that the evaluation of ``G_c`` in boundary-adjacent
   cells does not include fluxes across the external boundary, and;
2. Add the prescribed flux to the boundary-adjacent volumes prior to calculating ``G_c``:
   ``G_c \, |_b = G_c \, |_b - \frac{A_b}{V_b} q_c \, |_b \, \text{sign}(\hat{\boldsymbol{n}})``,
   where ``G_c \, |_b`` denotes values of ``G_c`` in boundary-adjacent volumes, ``q_c \, |_b``
   is the flux prescribed along the boundary, ``V_b`` is the volume of the boundary-adjacent
   cell, and ``A_b`` is the area of the external boundary of the boundary-adjacent cell.

   The factor ``\text{sign}(\hat{\boldsymbol{n}})`` is ``-``1 and ``+``1 on "left" and "right"
   boundaries, and accounts for the fact that a positive flux on a left boundary where
   ``\text{sign}(\hat{\boldsymbol{n}}) = -1`` implies an "inward" flux of ``c`` that increases
   interior values of ``c``, whereas a positive flux on a right boundary where
   ``\text{sign}(\hat{\boldsymbol{n}}) = 1`` implies an "outward" flux that decreases interior
   values of ``c``.

Flux boundary conditions are represented by the [`Flux`](@ref) type.

## Open boundary conditions

Open boundary conditions directly specify the value of the halo points. Typically this is used
to impose no penetration boundary conditions, i.e. setting wall normal velocity components on
to zero on the boundary.

The nuance here is that open boundaries behave differently for fields on face points in the
boundary direction due to the [staggered grid](@ref finite_volume). For example, the u-component
of velocity lies on `(Face, Center, Center)` points so for open `west` or `east` boundaries the
point specified by the boundary condition is the point lying on the boundary, where as for a
tracer on `(Center, Center, Center)` points the open boundary condition specifies a point outside
of the domain (hence the difference with `Value` boundary conditions).

The other important detail is that open (including no-penetration) boundary conditions are the
only conditions used on wall normal velocities when the domain is not periodic. This means that
their value affects the pressure calculation for nonhydrostatic models as it is involved in
calculating the divergence in the boundary adjacent center point (as described in the
[fractional step method](@ref time_stepping) documentation). Usually boundary points are filled
for the predictor velocity (i.e. before the pressure is calculated), and on the corrected field
(i.e. after the pressure correction is applied), but for open boundaries this would result in
the boundary adjacent center point becoming divergent so open boundaries are only filled for the
predictor velocity and stay the same after the pressure correction (so the boundary point is filled
with the final corrected velocity at the predictor step).

The restriction arrises as the boundary condition is specifying the wall normal velocity,
``\hat{\boldsymbol{n}}\cdot\boldsymbol{u}``, which leads to the pressure boundary condition
```math
    \begin{equation}
    \label{eq:pressure_boundary_condition}
    \Delta t \, \hat{\boldsymbol{n}}\cdot\boldsymbol{\nabla}p^{n+1}\big |_{\partial\Omega} = \left[\Delta t \, \hat{\boldsymbol{n}}\cdot\boldsymbol{u}^\star - \hat{\boldsymbol{n}}\cdot\boldsymbol{u}^{n+1}\right],
    \end{equation}
```
implying that there is a pressure gradient across the boundary. Since we solve the pressure poisson
equation (``\nabla^2p^{n+1}=\frac{\boldsymbol{\nabla}\cdot\boldsymbol{u}^\star}{\Delta t}``)
using the method described by [Schumann88](@citet) we have to move inhomogeneus boundary conditions
on the pressure to the right hand side. In order to do this we define a new field ``\phi`` where
```math
    \begin{equation}
    \label{eq:modified_pressure_field}
    \phi = p^{n+1} \quad \text{inside} \quad \Omega \quad \text{but} \quad \boldsymbol{\nabla} \cdot \boldsymbol{\nabla} \phi \, \big |_{\partial\Omega} = 0.
    \end{equation}
```
This moves the boundary condition to the right hand side as ``\phi`` becomes
```math
    \begin{equation}
    \label{eq:modified_pressure_poisson}
    \boldsymbol{\nabla}^2\phi^{n+1} = \boldsymbol{\nabla}\cdot\left[\frac{\boldsymbol{u}^\star}{\Delta t} - \delta\left(\boldsymbol{x} - \boldsymbol{x}_\Omega\right)\boldsymbol{\nabla}p\right].
    \end{equation}
```
Given the boundary condition on pressure given above, we can define a new modified predictor velocity
which is equal to the predictor velocity within the domain but shares boundary conditions with the
corrected field,
```math
    \begin{equation}
    \label{eq:quasi_predictor_velocity}
    \tilde{\boldsymbol{u}}^\star:=\boldsymbol{u}^\star + \delta\left(\boldsymbol{x} - \boldsymbol{x}_\Omega\right)(\boldsymbol{u}^{n+1} - \boldsymbol{u}^\star).
    \end{equation}
```
The modified pressure poisson equation becomes ``\nabla^2p^{n+1}=\frac{\boldsymbol{\nabla}\cdot\tilde{\boldsymbol{u}}^\star}{\Delta t}``
which can easily be solved.

Perhaps a more intuitive way to consider this is to recall that the corrector step projects ``\boldsymbol{u}^\star``
to the space of divergenece free velocity by applying
```math
    \begin{equation}
    \label{eq:pressure_correction_step}
    \boldsymbol{u}^{n+1} = \boldsymbol{u}^\star - \Delta t\boldsymbol{\nabla}p^{n+1},
    \end{equation}
```
but we have changed ``p^{n+1}`` to ``\phi`` and ``\boldsymbol{u}^\star`` to ``\tilde{\boldsymbol{u}}^\star``
so for ``\boldsymbol{\nabla}\phi \big |_{\partial\Omega} = 0`` the modified predictor velocity must
equal the corrected velocity on the boundary.

For simple open boundary conditions such as no penetration or a straight forward prescription of
a known velocity at ``t^{n+1}`` this is simple to implement as we just set the boundary condition
on the predictor velocity and don't change it after the correction. But some open boundary methods
calculate the boundary value based on the interior solution. As a simple example, if we wanted to
set the wall normal veloicty gradient to zero at the west boundary then we would set the boundary
point to
```math
    \begin{equation}
    \label{eq:zero_wall_normal_velocity_gradient}
    u^\star_{1jk} \approx u^\star_{3jk} + (u^\star_{2jk} - u^\star_{jk4}) / 2 + \mathcal{O}(\Delta x^2),
    \end{equation}
```
but we then pressure correct the interior so a new ``\mathcal{O}(\Delta t)`` error is introduced as
```math
    \begin{align}
    u^{n+1}_{1jk} &\approx u^{n+1}_{3jk} + (u^{n+1}_{2jk} - u^{n+1}_{jk4}) / 2 + \mathcal{O}(\Delta x^2),\\
    &= u^\star_{1jk} - \Delta t \left(\boldsymbol{\nabla}p^{n+1}_{3jk} + (\boldsymbol{\nabla}p^{n+1}_{2jk} - \boldsymbol{\nabla}p^{n+1}_{4jk}) / 2\right) + \mathcal{O}(\Delta x^2),\\
    &\approx u^\star_{1jk} + \mathcal{O}(\Delta x^2) + \mathcal{O}(\Delta t).
    \end{align}
```
This is prefered to a divergent interior solution as open boundary conditions (except no penetration)
are typlically already unphysical and only used in an attempt to allow information to enter or exit
the domain.

Open boundary conditions are represented by the [`Open`](@ref) type.
