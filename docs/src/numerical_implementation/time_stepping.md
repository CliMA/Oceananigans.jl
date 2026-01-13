# [Time-stepping](@id time_stepping)

## [Available time steppers](@id available-time-steppers)

The `TimeSteppers` module provides three generic, explicit time-stepping schemes. Importantly, the module handles only the
time integration of prognostic fields, the tendency computation is the responsibility
of each model's implementation.

!!! note "Time stepper availability in Oceananigans models"
    - `NonhydrostaticModel`: `QuasiAdamsBashforth2`, `RungeKutta3TimeStepper` (default)
    - `ShallowWaterModel`: `QuasiAdamsBashforth2TimeStepper`, `RungeKutta3` (default)
    - `HydrostaticFreeSurfaceModel`: `QuasiAdamsBashforth2TimeStepper` (default), `SplitRungeKuttaTimeStepper`

### Quasi-Adams-Bashforth second order

The `QuasiAdamsBashforth2TimeStepper` approximates the time integral of tendencies via
```math
    \begin{equation}
    \label{eq:adams-bashforth}
    \int_{t_n}^{t_{n+1}} G \, \mathrm{d} t \approx
        \Delta t \left [ \left ( \tfrac{3}{2} + \chi \right ) G^n
        - \left ( \tfrac{1}{2} + \chi \right ) G^{n-1} \right ] \, ,
    \end{equation}
```
where ``\chi`` is a parameter. [Ascher et a. (1995)](@cite Ascher95) suggest that ``\chi = \tfrac{1}{8}`` is optimal.
With the additional ``\chi`` parameter, the scheme is formally first-order accurate but offers improved
stability properties. The default ``\chi = 0.1`` provides a reasonable balance between accuracy and stability;
``\chi = 0`` recovers the standard second-order Adams-Bashforth method.

The scheme requires storing tendencies from the previous time step, but only requires one tendency evalution, 
making it computationally-efficient compared to multi-stage methods. The first time step automatically uses forward 
Euler (``\chi = -1/2``) since no previous tendencies exist.

### Runge-Kutta third order

The `RungeKutta3TimeStepper` implements a low-storage, third-order Runge-Kutta scheme following
[Le and Moin (1991)](@cite LeMoin1991). The scheme advances the state through three substeps per time step:
```math
U^{m+1} = U^m + \Delta t \left( \gamma^m G^m + \zeta^m G^{m-1} \right)
```
with default coefficients ``\gamma^1 = 8/15``, ``\gamma^2 = 5/12``, ``\gamma^3 = 3/4``,
``\zeta^2 = -17/60``, and ``\zeta^3 = -5/12`` (``\zeta^1 = 0``).

This scheme requires three tendency evaluations per time step but provides higher accuracy and better
stability for problems with oscillatory dynamics.

### Split Runge-Kutta

The `SplitRungeKuttaTimeStepper` implements a runge-kutta scheme suitable for split-explicit computations, that follows the implmentation
detailed in [Wicker and Skamarock (2002)](@cite WickerSkamarock2002). At the beginning of each time step the
prognostic fields are cached, and subsequent substeps compute:
```math
U^{m+1} = U^0 + \frac{\Delta t}{\beta^m} G^m
```
where ``U^0`` is the cached initial state and ``\beta`` are stage coefficients. The default three-stage
scheme uses ``\beta = (3, 2, 1)``. This time stepper is used by `HydrostaticFreeSurfaceModel` for
split-explicit treatment of the barotropic and baroclinic modes.

## Extending time steppers for custom models

To use the existing time steppers with a new model type, implement the following methods.
For `QuasiAdamsBashforth2TimeStepper`:

```julia
ab2_step!(model::MyModel, Δt, callbacks)           # advance fields one AB2 step
cache_previous_tendencies!(model::MyModel)         # store Gⁿ → G⁻
```

For `RungeKutta3TimeStepper`:

```julia
rk3_substep!(model::MyModel, Δt, γⁿ, ζⁿ, callbacks) # advance one RK3 substep
cache_previous_tendencies!(model::MyModel)          # store Gⁿ → G⁻
```

For `SplitRungeKuttaTimeStepper`:

```julia
cache_current_fields!(model::MyModel)               # store U → Ψ⁻ at step start
rk_substep!(model::MyModel, Δτ, callbacks)          # advance one substep
```

All models must also implement `update_state!(model, callbacks)` to fill halo regions and compute
any diagnostic quantities after each step or substep.

## The fractional step method

With the [pressure decomposition](@ref pressure_decomposition) as discussed, the momentum evolves via:

```math
    \begin{equation}
    \label{eq:momentum-time-derivative}
    \partial_t \boldsymbol{v} = \boldsymbol{G}_{\boldsymbol{v}} - \boldsymbol{\nabla} p_{\rm{non}} \, ,
    \end{equation}
```

where, e.g., for the non-hydrostatic model (ignoring background velocities and surface-wave effects)

```math
\boldsymbol{G}_{\boldsymbol{v}} \equiv - \boldsymbol{\nabla}_h p_{\rm{hyd}}
                       - \left ( \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{v}
                       - \boldsymbol{f} \times \boldsymbol{v}
                       + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau}
                       + \boldsymbol{F}_{\boldsymbol{v}}
```

collects all terms on the right side of the momentum equation \eqref{eq:momentum-time-derivative}, *except* the
contribution of non-hydrostatic pressure ``\boldsymbol{\nabla} p_{\rm{non}}``.

The time-integral of the momentum equation \eqref{eq:momentum-time-derivative} from time step ``n`` at ``t = t_n``
to time step ``n+1`` at ``t_{n+1}`` is:
```math
    \begin{equation}
    \label{eq:momentum-time-integral}
    \boldsymbol{v}^{n+1} - \boldsymbol{v}^n =
        \int_{t_n}^{t_{n+1}} \Big [ - \boldsymbol{\nabla} p_{\rm{non}} + \boldsymbol{G}_{\boldsymbol{v}} \Big ] \, \mathrm{d} t \, ,
    \end{equation}
```
where the superscript ``n`` and ``n+1`` imply evaluation at ``t_n`` and ``t_{n+1}``, such that
``\boldsymbol{v}^n \equiv \boldsymbol{v}(t=t_n)``. The crux of the fractional step method is
to treat the pressure term ``\boldsymbol{\nabla} p_{\rm{non}}`` implicitly using the approximation
```math
    \begin{align}
    \label{eq:pnon_implicit}
    \int_{t_n}^{t_{n+1}} \boldsymbol{\nabla} p_{\rm{non}} \, \mathrm{d} t \approx
        \Delta t \boldsymbol{\nabla} p_{\rm{non}}^{n+1} \, ,
    \end{align}
```
while treating the rest of the terms on the right hand side of \eqref{eq:momentum-time-integral}
explicitly. The implicit treatment of pressure ensures that the velocity field obtained at
time step ``n+1`` is divergence-free.

To effect such a fractional step method, we define an intermediate velocity field ``\boldsymbol{v}^\star`` such that
```math
    \begin{equation}
    \label{eq:intermediate-velocity-field}
    \boldsymbol{v}^\star - \boldsymbol{v}^n = \int_{t_n}^{t_{n+1}} \boldsymbol{G}_{\boldsymbol{v}} \, \mathrm{d} t \, ,
    \end{equation}
```

The integral on the right of the equation for ``\boldsymbol{v}^\star`` may be approximated by any of the
time steppers described in [Available time steppers](@ref available-time-steppers).

Combining the equation \eqref{eq:intermediate-velocity-field} for ``\boldsymbol{v}^\star`` and the time integral
of the non-hydrostatic pressure \eqref{eq:pnon_implicit} yields
```math
    \begin{equation}
    \label{eq:fractional-step}
    \boldsymbol{v}^{n+1} - \boldsymbol{v}^\star = - \Delta t \boldsymbol{\nabla} p_{\rm{non}}^{n+1} \, .
    \end{equation}
```

Taking the divergence of fractional step equation \eqref{eq:fractional-step} and requiring that
``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v}^{n+1} = 0`` yields a Poisson equation
for the kinematic pressure ``p_{\rm{non}}`` at time-step ``n+1``:
```math
    \begin{equation}
    \label{eq:pressure-poisson}
    \nabla^2 p_{\rm{non}}^{n+1} = \frac{\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v}^{\star}}{\Delta t} \, .
    \end{equation}
```
With ``\boldsymbol{v}^\star`` in hand we can invert \eqref{eq:pressure-poisson} to get ``p_{\rm{non}}^{n+1}``
and then ``\boldsymbol{v}^{n+1}`` is computed via the fractional step equation \eqref{eq:fractional-step}.

Tracers are stepped forward explicitly via
```math
    \begin{equation}
    \label{eq:tracer-timestep}
    c^{n+1} - c^n = \int_{t_n}^{t_{n+1}} G_c \, \mathrm{d} t \, ,
    \end{equation}
```
where
```math
    \begin{equation}
    G_c \equiv - \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{v} c \right ) - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c + F_c \, ,
    \end{equation}
```
and the same time-stepping scheme used for the momentum tendencies is applied to evaluate the integral of ``G_c``.
