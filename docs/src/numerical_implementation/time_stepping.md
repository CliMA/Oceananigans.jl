# [Time-stepping and the fractional step method](@id time_stepping)

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

The integral on the right of the equation for ``\boldsymbol{v}^\star`` may be approximated by a variety of explicit
methods. For example, a forward Euler method approximates the integral via
```math
    \begin{equation}
    \int_{t_n}^{t_{n+1}} G \, \mathrm{d} t \approx \Delta t G^n \, ,
    \label{eq:forward-euler}
    \end{equation}
```
for any time-dependent function ``G(t)``, while a second-order Adams-Bashforth method uses the approximation
```math
    \begin{equation}
    \label{eq:adams-bashforth}
    \int_{t_n}^{t_{n+1}} G \, \mathrm{d} t \approx
        \Delta t \left [ \left ( \tfrac{3}{2} + \chi \right ) G^n 
        - \left ( \tfrac{1}{2} + \chi \right ) G^{n-1} \right ] \, ,
    \end{equation}
```
where ``\chi`` is a parameter. [Ascher95](@citet) claim that ``\chi = \tfrac{1}{8}`` is optimal; 
``\chi = -\tfrac{1}{2}`` yields the forward Euler scheme.

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
and the same forward Euler or Adams-Bashforth scheme as for the explicit evaluation of the time-integral of
``\boldsymbol{G}_u`` is used to evaluate the integral of ``G_c``.
