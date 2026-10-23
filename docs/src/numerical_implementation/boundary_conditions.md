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
condition is enforced by setting the value of halo points located outside the domain interior
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
of discrete values, or an arbitrary function. To explain how flux boundary conditions are imposed
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
`Oceananigans.jl` exploits this fact to define an algorithm that prescribes fluxes across external
boundaries ``\partial \Omega_b``:

1. Impose a constant gradient ``\hat{\boldsymbol{n}} \boldsymbol{\cdot} \boldsymbol{\nabla} c
   \, |_{\partial \Omega_b} = 0`` across external boundaries by using halo points (similar
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

## Normal flow boundary conditions

Normal flow boundary conditions directly specify the velocity component normal to the boundary.
Typically this is used to impose no penetration boundary conditions, i.e. setting wall normal
velocity components to zero on the boundary, or to prescribe the velocity of an inflow or outflow.

On the [staggered grid](@ref finite_volume) the boundary-normal velocity lies on face points in the
boundary direction. For example, the u-component of velocity lies on `(Face, Center, Center)` points,
so on `west` or `east` boundaries the point specified by the boundary condition lies on the boundary
itself. A tracer on `(Center, Center, Center)` points has no point on the boundary, and its boundary
conditions set the halo points outside of the domain instead.

The other important detail is that normal flow boundary conditions, including no-penetration, are the
only conditions used on wall normal velocities when the domain is not periodic. This means that
their value affects the pressure calculation for nonhydrostatic models as it is involved in
calculating the divergence in the boundary adjacent center point (as described in the
[fractional step method](@ref time_stepping) documentation). Usually boundary points are filled
for the predictor velocity (i.e. before the pressure is calculated), and on the corrected field
(i.e. after the pressure correction is applied), but for normal flow boundary conditions this would
result in the boundary adjacent center point becoming divergent, so they are only filled for the
predictor velocity and stay the same after the pressure correction (so the boundary point is filled
with the final corrected velocity at the predictor step).

The restriction arises as the boundary condition is specifying the wall normal velocity,
``\hat{\boldsymbol{n}} \boldsymbol{\cdot} \boldsymbol{u}``, which leads to the pressure boundary condition
```math
    \begin{equation}
    \label{eq:pressure_boundary_condition}
    \Delta t \, \hat{\boldsymbol{n}} \boldsymbol{\cdot} \boldsymbol{\nabla} p^{n+1} \big |_{\partial\Omega} = \left[\Delta t \, \hat{\boldsymbol{n}} \boldsymbol{\cdot} \boldsymbol{u}^\star - \hat{\boldsymbol{n}} \boldsymbol{\cdot} \boldsymbol{u}^{n+1} \right],
    \end{equation}
```
implying that there is a pressure gradient across the boundary. Since we solve the pressure poisson
equation (``\nabla^2p^{n+1} = \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{u}^\star / \Delta t``)
using the method described by [Schumann88](@citet), we have to move inhomogeneous boundary conditions
on the pressure to the right hand side. In order to do this we define a new field ``\phi`` where
```math
    \begin{equation}
    \label{eq:modified_pressure_field}
    \phi = p^{n+1} \quad \text{inside} \quad \Omega \quad \text{but} \quad \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\nabla} \phi \, \big |_{\partial\Omega} = 0.
    \end{equation}
```
This moves the boundary condition to the right hand side as ``\phi`` becomes
```math
    \begin{equation}
    \label{eq:modified_pressure_poisson}
    \boldsymbol{\nabla}^2\phi^{n+1} = \boldsymbol{\nabla} \boldsymbol{\cdot} \left[\frac{\boldsymbol{u}^\star}{\Delta t} - \delta\left(\boldsymbol{x} - \boldsymbol{x}_\Omega\right)\boldsymbol{\nabla}p\right].
    \end{equation}
```
Given the boundary condition on pressure given above, we can define a new modified predictor velocity
which is equal to the predictor velocity within the domain but shares boundary conditions with the
corrected field,
```math
    \begin{equation}
    \label{eq:quasi_predictor_velocity}
    \tilde{\boldsymbol{u}}^\star \equiv \boldsymbol{u}^\star + \delta\left(\boldsymbol{x} - \boldsymbol{x}_\Omega\right)(\boldsymbol{u}^{n+1} - \boldsymbol{u}^\star).
    \end{equation}
```
The modified pressure Poisson equation becomes ``\nabla^2 p^{n+1} = \frac{\boldsymbol{\nabla} \boldsymbol{\cdot} \tilde{\boldsymbol{u}}^\star}{\Delta t}``
which can easily be solved.

Perhaps a more intuitive way to consider this is to recall that the corrector step projects ``\boldsymbol{u}^\star``
to the space of divergence free velocity by applying
```math
    \begin{equation}
    \label{eq:pressure_correction_step}
    \boldsymbol{u}^{n+1} = \boldsymbol{u}^\star - \Delta t\boldsymbol{\nabla}p^{n+1},
    \end{equation}
```
but we have changed ``p^{n+1}`` to ``\phi`` and ``\boldsymbol{u}^\star`` to ``\tilde{\boldsymbol{u}}^\star``
so for ``\boldsymbol{\nabla}\phi \big |_{\partial\Omega} = 0`` the modified predictor velocity must
equal the corrected velocity on the boundary.

For simple normal flow boundary conditions such as no penetration or a straight forward prescription of
a known velocity at ``t^{n+1}`` this is simple to implement as we just set the boundary condition
on the predictor velocity and don't change it after the correction. But open boundary conditions,
which are normal flow conditions with a `scheme`, calculate the boundary value based on the interior
solution. As a simple example, if we wanted to set the wall normal velocity gradient to zero at the
west boundary then we would set the boundary point to
```math
    \begin{equation}
    \label{eq:zero_wall_normal_velocity_gradient}
    u^\star_{1jk} \approx u^\star_{3jk} + \frac{1}{2}(u^\star_{2jk} - u^\star_{4jk}) + \mathcal{O}(\Delta x^2),
    \end{equation}
```
but we then pressure correct the interior so a new ``\mathcal{O}(\Delta t)`` error is introduced as
```math
    \begin{align}
    u^{n+1}_{1jk} &\approx u^{n+1}_{3jk} + (u^{n+1}_{2jk} - u^{n+1}_{4jk}) / 2 + \mathcal{O}(\Delta x^2),\\
    &= u^\star_{1jk} - \Delta t \left[ \boldsymbol{\nabla} p^{n+1}_{3jk} + \frac{1}{2}(\boldsymbol{\nabla} p^{n+1}_{2jk} - \boldsymbol{\nabla} p^{n+1}_{4jk}) \right] + \mathcal{O}(\Delta x^2),\\
    &\approx u^\star_{1jk} + \mathcal{O}(\Delta x^2) + \mathcal{O}(\Delta t).
    \end{align}
```
This is preferred to a divergent interior solution as open boundary conditions are typically already
unphysical and only used in an attempt to allow information to enter or exit the domain.

Normal flow boundary conditions are represented by the [`NormalFlow`](@ref) classification.

## Open boundary condition "schemes"

Except for trivial cases (i.e. no-penetration) the velocity on the boundary point has to be
approximated as it is outside the computed domain. There is insufficient information to step the
full equation of motion as gradients across the boundary cannot be computed and simply prescribing
a boundary normal velocity is unphysical and reflects energy leaving the domain [Orlanksi1976](@citep).

An open boundary condition is a `NormalFlowBoundaryCondition` or a `ValueBoundaryCondition` with a
scheme, passed through the `scheme` keyword argument. The scheme computes the boundary value from the
interior solution and is designed to let disturbances propagate out of the domain. In other literature
this is sometimes called radiation boundary conditions. In a `NormalFlowBoundaryCondition` the scheme
sets the boundary-normal velocity on the boundary face. In a `ValueBoundaryCondition` it sets the first
halo cell of a field that is `Center`-located in the boundary-normal direction, such as a tracer or a
boundary-tangential velocity component. The available schemes are

| Scheme                          | Boundary condition                                      | Fields                          |
|:--------------------------------|:--------------------------------------------------------|:--------------------------------|
| [`PerturbationAdvection`](@ref) | `NormalFlowBoundaryCondition`, `ValueBoundaryCondition` | velocities, tracers             |
| [`NormalRadiation`](@ref)       | `NormalFlowBoundaryCondition`, `ValueBoundaryCondition` | velocities, tracers             |
| [`ObliqueRadiation`](@ref)      | `NormalFlowBoundaryCondition`, `ValueBoundaryCondition` | velocities, tracers             |
| [`TracerReservoir`](@ref)       | `ValueBoundaryCondition`                                | tracers                         |
| [`GravityWaveRadiation`](@ref)  | `NormalFlowBoundaryCondition`                           | barotropic transports `U`, `V`  |
| [`SurfaceWaveRadiation`](@ref)  | `ValueBoundaryCondition`                                | free surface displacement `η`   |

`GravityWaveRadiation` and `SurfaceWaveRadiation` apply to the barotropic state of a
[`SplitExplicitFreeSurface`](@ref).

### Perturbation advection

A common method for allowing information to exit is to perform a one-sided advection operation.
For example, in the [Orlanksi1976](@citet) boundary condition fields are advected out of the domain by a
locally determined phase speed. We can show that this is the first-order approximation of the full equations
of motion in the predictor velocity step. Consider a right boundary normal to the `u` velocity
component (the east boundary):
```math
    \partial_t u + u \partial_x u + v \partial_y u + w \partial_z u = - (\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau})_x + F,
```
let ``\boldsymbol{u} = \boldsymbol{U} + \boldsymbol{u}'`` with ``\boldsymbol{U} = U(x, y, z, t) \hat{\boldsymbol{x}}``
where ``U`` is an externally determined "background" wall-normal flow in the proximity of the boundary,
and assume that the divergence of ``\boldsymbol{\tau}`` is small,
```math
    \partial_t u = -(U + u') \partial_x(U + u') - v \partial_y (U + u') - w \partial_z (U + u') + F,
```
then, taking only first-order terms:
```math
    \partial_t u = - U \partial_x u' - v \partial_y U - w \partial_z U + F.
```
Now consider the dominant forcing to be relaxation to the background state (we will explain why later) so
that ``F = (U - u) / \tau``, and recall that when we compute the boundary normal point the interior domain is
already stepped, we can take an upwind backward Euler step:
```math
    \frac{u^{n+1} - u^n}{\Delta t} = -\max(0, U)\frac{u^{n+1}_i - u^{n+1}_{i-1}}{\Delta x} - \min(0, U) \frac{u^{n+1}_{i+1} - u^{n+1}_{i}}{\Delta x} - v^{n+1}\partial_y U - w^{n+1} \partial_z U + \frac{U - u^{n+1}}{\tau},
```
where ``i`` is the boundary point and ``v`` and ``w`` are interpolated to the boundary point. Of course if ``U<0``
(i.e. it is directed into the domain), then we need ``u_{i+1}`` which is outside of the domain, so this scheme
cannot be used. We therefore disregard this term, taking ``\bar{U} = \max(0, U)\Delta t / \Delta x`` and
``\bar{\tau} = \Delta t / \tau``, and rearranging we recover:
```math
    u^{n+1} = \frac{u^{n} + \bar{U} u^{n+1}_{i-1} + U\bar{\tau} - v^{n+1} \Delta t \partial_y U - w^{n+1} \Delta t \partial_z U}{1 + \bar{\tau} + \bar{U}}.
```
For numerical stability, we also need to apply a CFL-like constraint and set ``\bar{U} = \min(1, \max(0, U) \Delta t / \Delta x)``.

Previously we considered the dominant forcing to be relaxation to ``U`` so that we can take the backward Euler
step. And because it is useful to prescribe relaxation on the boundary both to damp numerical oscillations,
and to prevent inconsistency when ``U`` is directed into the domain.

The implementation neglects the terms with ``\partial_y U`` and ``\partial_z U``. The relaxation time scale
``\tau`` is `outflow_timescale` when ``U`` points out of the domain and `inflow_timescale` when it points
into it, and a zero time scale sets the boundary value to ``U``. The keyword argument `gravity_wave_speed`
adds a constant outward speed to ``U`` in ``\bar{U}``, for fields whose signals travel faster than the
flow. In a `ValueBoundaryCondition` the same update applies to a tracer, with ``\bar{U}`` computed from the
boundary-normal velocity on the boundary face and the tracer relaxed to its exterior value.

### Normal radiation

[`NormalRadiation`](@ref) is the radiation condition of [Orlanksi1976](@citet) with the adaptive nudging
of [marchesiello.ea2001](@citet). A field ``\phi`` on the boundary obeys
```math
    \begin{equation}
    \label{eq:normal_radiation}
    \partial_t \phi + c_n \partial_n \phi = - \frac{\phi - \phi^{\mathrm{ext}}}{\tau} \, ,
    \end{equation}
```
where ``n`` is the outward boundary-normal coordinate, ``c_n`` is the speed at which signals cross the
boundary, ``\phi^{\mathrm{ext}}`` is the exterior value given by the boundary condition, and ``\tau`` is a
relaxation time scale. [Orlanksi1976](@citet) diagnoses ``c_n`` from the solution next to the boundary,
where \eqref{eq:normal_radiation} without relaxation gives ``c_n = - \partial_t \phi / \partial_n \phi``.

Let ``\phi^b`` be the value at the boundary point, which is the boundary face in a
`NormalFlowBoundaryCondition` and the first halo cell in a `ValueBoundaryCondition`, and let ``\phi_1`` and
``\phi_2`` be the values at the first and second interior points next to it. The interior is stepped before
the boundary is filled, so ``\phi_1^{n+1}`` and ``\phi_2^{n+1}`` are known when ``\phi^b`` is updated. The
phase speed is diagnosed as the Courant number
```math
    C = - \frac{\phi_1^{n+1} - \phi_1^n}{\phi_1^{n+1} - \phi_2^{n+1}} \, ,
```
with ``C = 0`` where the denominator vanishes. Inflow and outflow are distinguished by the sign of the
boundary-normal velocity ``u_n``. In a `NormalFlowBoundaryCondition`, ``u_n`` is the normal velocity on the
first interior face, ``\phi_1``. In a `ValueBoundaryCondition`, it is the normal velocity one face into the
interior, or on the boundary face when `use_boundary_velocity = true`. The sign of ``C`` is not used because
``C`` is ill-defined when an extremum of ``\phi`` crosses the boundary. On outflow the Courant number is
bounded below by the advective Courant number and above by one,
```math
    C_n = \min \left [ 1, \max \left ( C, \frac{|u_n| \Delta t}{\Delta n} \right ) \right ] \, ,
```
and ``\tau`` is `outflow_timescale`. On inflow ``C_n = 0`` and ``\tau`` is `inflow_timescale`. A backward
Euler step of \eqref{eq:normal_radiation} on the boundary point gives
```math
    \begin{equation}
    \label{eq:normal_radiation_update}
    \phi^{b, n+1} = \frac{\phi^{b, n} + C_n \phi_1^{n+1} + \tilde{\tau} \phi^{\mathrm{ext}}}{1 + C_n + \tilde{\tau}} \, ,
    \qquad \tilde{\tau} = \frac{\Delta t}{\tau} \, ,
    \end{equation}
```
with ``\phi^{b, n+1} = \phi^{\mathrm{ext}}`` when ``\tau = 0``. The update is a convex combination of
``\phi^{b, n}``, ``\phi_1^{n+1}`` and ``\phi^{\mathrm{ext}}``, so it creates no new extrema. With the
defaults `inflow_timescale = 0` and `outflow_timescale = Inf` the boundary value is prescribed on inflow
and radiates freely on outflow. The adaptive nudging of [marchesiello.ea2001](@citet) relaxes strongly on
inflow, where the exterior data set the boundary value, and weakly on outflow, where the interior solution
does.

The scheme stores ``\phi^{b, n}`` and ``\phi_1^n``, the values at the start of the time step, at every
boundary point. Every stage of a multi-stage time stepper updates ``\phi^b`` from these stored values with
the time step of that stage. Before the first time step, ``\phi^b`` is set to ``\phi_1``, or to
``\phi^{\mathrm{ext}}`` where ``\tau = 0``.

### Oblique radiation

[`ObliqueRadiation`](@ref) extends normal radiation to signals that reach the boundary at an angle,
following [raymond.kuo1984](@citet). With ``s`` the coordinate tangential to the boundary, the radiation
equation becomes
```math
    \partial_t \phi + c_n \partial_n \phi + c_s \partial_s \phi = - \frac{\phi - \phi^{\mathrm{ext}}}{\tau} \, .
```
A single equation does not determine both components of the phase velocity. [raymond.kuo1984](@citet) take
the phase velocity parallel to ``\boldsymbol{\nabla} \phi``, which gives
```math
    c_n = - \frac{\partial_t \phi \, \partial_n \phi}{(\partial_n \phi)^2 + (\partial_s \phi)^2} \, , \qquad
    c_s = - \frac{\partial_t \phi \, \partial_s \phi}{(\partial_n \phi)^2 + (\partial_s \phi)^2} \, .
```
The Courant numbers are diagnosed at the first interior point from the differences
``\delta_t \phi = \phi_1^{n+1} - \phi_1^n`` and ``\delta_n \phi = \phi_1^{n+1} - \phi_2^{n+1}``, and from a
tangential difference ``\delta_s \phi`` of ``\phi_1`` that is upwinded with respect to the sign of ``c_s``,
```math
    C_n = - \frac{\delta_t \phi \, \delta_n \phi}{(\delta_n \phi)^2 + (\delta_s \phi)^2} \, , \qquad
    C_s = - \frac{\delta_t \phi \, \delta_s \phi}{(\delta_n \phi)^2 + (\delta_s \phi)^2} \, .
```
On outflow ``C_n`` is bounded as in normal radiation and ``C_s`` is bounded to ``[-1, 1]``. On inflow both
are zero. The tangential term is discretized explicitly with upwind differences of the boundary values,
```math
    \phi^{b, n+1} = \frac{\phi^{b, n} + C_n \phi_1^{n+1} - \max(C_s, 0) \, \delta_s^- \phi^b - \min(C_s, 0) \, \delta_s^+ \phi^b + \tilde{\tau} \phi^{\mathrm{ext}}}{1 + C_n + \tilde{\tau}} \, ,
```
where ``\delta_s^-`` and ``\delta_s^+`` are backward and forward differences along the boundary. All
tangential differences use the values of the previous time step and are zero at the ends of the boundary.
Where ``\delta_s \phi = 0`` the scheme reduces to \eqref{eq:normal_radiation_update}. The tangential term
is applied on lateral boundaries only, so top and bottom boundaries use normal radiation. The treatment of
inflow and outflow, the advecting velocity, and the stored values are the same as in normal radiation.

### Tracer reservoir

A tracer boundary with a prescribed exterior value has no memory of the water that leaves the domain. When
the boundary-normal flow reverses, the water that re-enters carries the exterior value ``c^{\mathrm{ext}}``
regardless of what left through the same face, so oscillating flows such as tides exchange tracer
spuriously across the boundary. [`TracerReservoir`](@ref) follows the tracer reservoirs of MOM6 and keeps
a reservoir value ``c^r`` just outside each boundary point. The reservoir relaxes toward an upstream value
``c^\star`` over a distance ``L`` advected across the boundary,
```math
    \partial_t c^r = \frac{|u_n|}{L} \left ( c^\star - c^r \right ) \, ,
```
where ``u_n`` is the boundary-normal velocity on the boundary face. On outflow ``c^\star = c_1``, the
adjacent interior value, and ``L`` is `outflow_length_scale`. On inflow ``c^\star = c^{\mathrm{ext}}`` and
``L`` is `inflow_length_scale`. A backward Euler step gives
```math
    c^{r, n+1} = \frac{c^{r, n} + a \, c^\star}{1 + a} \, , \qquad a = \frac{|u_n| \Delta t}{L} \, ,
```
which is stable for any ``\Delta t``, and the halo cell is set to ``c^{r, n+1}``. Water that re-enters the
domain therefore carries back the tracer that recently left, and relaxes to ``c^{\mathrm{ext}}`` over an
advected distance of about `inflow_length_scale`. A zero length scale sets ``c^r = c^\star`` at every time
step and an infinite one freezes the reservoir. With both length scales zero, the default,
`TracerReservoir` reduces to a condition without memory: zero gradient on outflow and ``c^{\mathrm{ext}}``
on inflow. The reservoir starts from ``c^{\mathrm{ext}}`` and is advanced over the stages of a time step in
the same way as the stored values of normal radiation.

### Gravity wave radiation

A [`SplitExplicitFreeSurface`](@ref) advances the barotropic transports ``U = \int u \, \mathrm{d}z`` and
``V = \int v \, \mathrm{d}z`` and the free surface displacement ``\eta`` with many short substeps of the
barotropic equations. Their open boundaries must let surface gravity waves, which travel at
``c = \sqrt{g H}`` in a water column of depth ``H``, leave the domain, and let exterior signals such as
tides enter it. Linearized in the direction normal to an east boundary, the barotropic equations are
```math
    \partial_t \eta + \partial_x U = 0 \, , \qquad \partial_t U + g H \, \partial_x \eta = 0 \, ,
```
whose Riemann invariants ``U + c \eta`` and ``U - c \eta`` travel at the speeds ``+c`` and ``-c``. At an
east boundary the first leaves the domain and the second enters it. [`GravityWaveRadiation`](@ref) is the
condition of [flather1976tidal](@citet), which takes the incoming invariant from the exterior state,
``U^b - c \eta^b = U^{\mathrm{ext}} - c \eta^{\mathrm{ext}}``, so that
```math
    U^b = U^{\mathrm{ext}} + \sqrt{g H} \left ( \eta^b - \eta^{\mathrm{ext}} \right ) \, .
```
On west and south boundaries the incoming invariant is ``U + c \eta`` and the sign of the second term is
reversed. The exterior transport and free surface are the 2-tuple `(U, η)` value of the boundary condition,
see [`GravityWaveRadiationBoundaryCondition`](@ref). The free surface on the boundary face, ``\eta^b``, is
the average of the first interior and halo values of ``\eta``, ``H`` is the column depth on the boundary
face, and ``U^b = 0`` where the column is dry.

The condition is applied at every barotropic substep. A `target_transport` ``Q`` shifts the transports on
the boundary face after every application of the condition,
```math
    U^b \leftarrow U^b - \frac{1}{L} \left ( \int U^b \, \mathrm{d}l - Q \right ) \, ,
```
where ``\mathrm{d}l`` is the length element along the boundary and ``L`` is the length of its wet part. After
the substeps, the barotropic correction adjusts the three-dimensional velocity on the boundary face so that
its depth integral equals the barotropic transport. The boundary condition on the three-dimensional normal
velocity, for example a [`NormalRadiation`](@ref) scheme, therefore sets only the departure from the depth
average.

### Surface wave radiation

[`SurfaceWaveRadiation`](@ref) is the companion condition on ``\eta``, from [chapman1985numerical](@citet).
The free surface radiates at the gravity wave speed,
```math
    \partial_t \eta + \sqrt{g H} \, \partial_n \eta = 0 \, ,
```
which is discretized implicitly as in ROMS,
```math
    \eta^{h, n+1} = \frac{\eta^{h, n} + C \eta_1^{n+1}}{1 + C} \, , \qquad C = \frac{\sqrt{g H} \, \Delta t}{\Delta n} \, ,
```
where ``\eta^h`` is the halo value, ``\eta_1`` the adjacent interior value, ``H`` the column depth of the
interior cell and ``\Delta t`` the barotropic substep. The halo value evolves with the radiation equation
instead of mirroring the interior value, as the default zero-gradient condition on ``\eta`` does, and it
enters the [flather1976tidal](@citet) condition through the face average ``\eta^b``.

In each barotropic substep the ``\eta`` halos are filled first, then ``U`` and ``V`` are advanced and their
boundary values set by `GravityWaveRadiation`, and finally ``\eta`` is advanced with the new transports. A
`SplitExplicitFreeSurface` adds `SurfaceWaveRadiation` on ``\eta`` at every boundary where ``U`` or ``V``
has a `GravityWaveRadiation` condition, unless boundary conditions for ``\eta`` are given.
[`tidal_boundary_conditions`](@ref) builds both conditions on every lateral boundary from the harmonic
constants of a tidal atlas.

### Net transport through open boundaries

Before the pressure solve of a [`NonhydrostaticModel`](@ref), the boundary-normal velocities set by the
schemes are corrected so that the net volume transport into the domain vanishes, which the pressure
Poisson equation with a rigid lid requires. The keyword argument `target_transport` helps the user have
control over how this correction is applied.

The correction has two steps. First, every boundary whose scheme has a `target_transport` ``Q`` is shifted
uniformly so that its transport matches the target,
```math
    u^b \leftarrow u^b - \frac{1}{A} \left ( \int u^b \, \mathrm{d}A - Q \right ) \, ,
```
where ``A`` is the area of the boundary and the transport is positive in the positive coordinate
direction. Second, the remaining net inflow is removed by the same uniform shift of the normal velocity on
all boundaries that have a scheme and no target. If every such boundary has a target, the targets must add
up to zero net transport.

A [`HydrostaticFreeSurfaceModel`](@ref) or a [`NonhydrostaticFreeSurfaceModel`](@ref) with a free surface
do not need the second step because the free surface absorbs a net imbalance.
With an [`ImplicitFreeSurface`](@ref) the first step is applied to the predictor velocity
before the free surface solve, so the transport after the barotropic correction matches the target only
approximately. With a `SplitExplicitFreeSurface` the transport is prescribed through the
`target_transport` of `GravityWaveRadiation` on the barotropic transport.
