# Time-stepping and the fractional step method

The time-integral of the momentum equation with the pressure decomposition from time step ``n`` at ``t = t_n`` 
to time step ``n+1`` at ``t_{n+1}`` is
```math
    \begin{equation}
    \label{eq:momentum-time-integral}
    \boldsymbol{v}^{n+1} - \boldsymbol{v}^n = 
        \int_{t_n}^{t_{n+1}} \Big [ - \boldsymbol{\nabla} p_{\rm{non}} 
                                    - \boldsymbol{\nabla}_{h} p_{\rm{hyd}} 
                                    - \left ( \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{v} 
                                    - \boldsymbol{f} \times \boldsymbol{v} 
                                    + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau} 
                                    + \boldsymbol{F}_{\boldsymbol{v}} \Big ] \, \mathrm{d} t \, ,
    \end{equation}
```
where the superscript ``n`` and ``n+1`` imply evaluation at ``t_n`` and ``t_{n+1}``, such that 
``\boldsymbol{v}^n \equiv \boldsymbol{v}(t=t_n)``. The crux of the fractional step method is 
to treat the pressure term ``\boldsymbol{\nabla} p_{\rm{non}}`` implicitly using the approximation
```math
\int_{t_n}^{t_{n+1}} \boldsymbol{\nabla} p_{\rm{non}} \, \mathrm{d} t \approx 
    \Delta t \boldsymbol{\nabla} p_{\rm{non}}^{n+1} \, ,
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
where, e.g., for the incompressible model, 
```math
\boldsymbol{G}_{\boldsymbol{v}} \equiv - \boldsymbol{\nabla}_h p_{\rm{hyd}} 
                       - \left ( \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{v} 
                       - \boldsymbol{f} \times \boldsymbol{v} 
                       + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau} 
                       + \boldsymbol{F}_{\boldsymbol{v}}
```
collects all terms on the right side of the time-integral of the momentum equation except the 
contribution of non-hydrostatic pressure ``\boldsymbol{\nabla} p_n``. The integral on the right 
of the equation for ``\boldsymbol{v}^\star`` may be approximated by a variety of  explicit methods: 
for example, a forward Euler method uses
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
where ``\chi`` is a parameter. [Ascher95](@cite) claim that ``\chi = \tfrac{1}{8}`` is optimal; 
``\chi = -\tfrac{1}{2}`` yields the forward Euler scheme.

Combining the equations for ``\boldsymbol{v}^\star`` and the time integral of the momentum equation yields
```math
    \begin{equation}
    \label{eq:fractional-step}
    \boldsymbol{v}^{n+1} - \boldsymbol{v}^\star = - \Delta t \boldsymbol{\nabla} p_{\rm{non}}^{n+1} \, .
    \end{equation}
```
Taking the divergence of fractional step equation and requiring that 
``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v}^{n+1} = 0`` yields a Poisson equation 
for the kinematic pressure ``p_{\rm{non}}`` at time-step ``n+1``:
```math
    \nabla^2 p_{\rm{non}}^{n+1} = \frac{\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v}^{\star}}{\Delta t} \, .
```
With ``\boldsymbol{v}^\star`` and ``p_{\rm{non}}`` in hand, ``\boldsymbol{v}^{n+1}`` is then 
computed via the fractional step equation.

Tracers are stepped forward explicitly via
```math
    \begin{equation}
    \label{eq:tracer-timestep}
    c^{n+1} - c^n = \int_{t_n}^{t_{n+1}} G_c \, \mathrm{d} t \, ,
    \end{equation}
```
where 
```math
    G_c \equiv - \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{v} c \right ) - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c + F_c \, ,
```
and the same forward Euler or Adams-Bashforth scheme as for the explicit evaluation of the time-integral of
``\boldsymbol{G}_u`` is used to evaluate the integral of ``G_c``.
