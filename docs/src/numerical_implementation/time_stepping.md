# Time-stepping and the fractional step method

The time-integral of the momentum equation with the pressure decomposition from time step ``n`` at ``t = t_n`` 
to time step ``n+1`` at ``t_{n+1}`` is
```math
    \begin{equation}
    \label{eq:momentum-time-integral}
    \boldsymbol{u}^{n+1} - \boldsymbol{u}^n = 
        \int_{t_n}^{t_{n+1}} \Big [ - \boldsymbol{\nabla} \phi_{\rm{non}} 
                                    - \boldsymbol{\nabla}_{h} \phi_{\rm{hyd}} 
                                    - \left ( \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{u} 
                                    - \boldsymbol{f} \times \boldsymbol{u} 
                                    + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau} 
                                    + \boldsymbol{F}_{\boldsymbol{u}} \Big ] \, \mathrm{d} t \, ,
    \end{equation}
```
where the superscript ``n`` and ``n+1`` imply evaluation at ``t_n`` and ``t_{n+1}``, 
such that ``\boldsymbol{u}^n \equiv \boldsymbol{u}(t=t_n)``.
The crux of the fractional step method is to treat the pressure term 
``\boldsymbol{\nabla} \phi_{\rm{non}}`` implicitly using the approximation
```math
\int_{t_n}^{t_{n+1}} \boldsymbol{\nabla} \phi_{\rm{non}} \, \mathrm{d} t \approx 
    \Delta t \boldsymbol{\nabla} \phi_{\rm{non}}^{n+1} \, ,
```
while treating the rest of the terms on the right hand side of \eqref{eq:momentum-time-integral} explicitly.
The implicit treatment of pressure ensures that the velocity field obtained at time step ``n+1`` is divergence-free.

To effect such a fractional step method, we define an intermediate velocity field ``\boldsymbol{u}^\star`` such that
```math
    \begin{equation}
    \label{eq:intermediate-velocity-field}
    \boldsymbol{u}^\star - \boldsymbol{u}^n = \int_{t_n}^{t_{n+1}} \boldsymbol{G}_{\boldsymbol{u}} \, \mathrm{d} t \, ,
    \end{equation}
```
where
```math
\boldsymbol{G}_{\boldsymbol{u}} \equiv - \boldsymbol{\nabla}_h \phi_{\rm{hyd}} 
                       - \left ( \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{u} 
                       - \boldsymbol{f} \times \boldsymbol{u} 
                       + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau} 
                       + \boldsymbol{F}_{\boldsymbol{u}}
```
collects all terms on the right side of the time-integral of the momentum equation except the contribution 
of non-hydrostatic pressure ``\boldsymbol{\nabla} \phi_n``.
The integral on the right of the equation for ``\boldsymbol{u}^\star`` may be approximated by a variety of 
explicit methods: for example, a forward Euler method uses
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
where ``\chi`` is a parameter. Ascher et al. (1995) claim that ``\chi = \tfrac{1}{8}`` is optimal; 
``\chi=-\tfrac{1}{2}`` yields the forward Euler scheme.

Combining the equations for ``\boldsymbol{u}^\star`` and the time integral of the momentum equation yields
```math
    \begin{equation}
    \label{eq:fractional-step}
    \boldsymbol{u}^{n+1} - \boldsymbol{u}^\star = - \Delta t \boldsymbol{\nabla} \phi_{\rm{non}}^{n+1} \, .
    \end{equation}
```
Taking the divergence of fractional step equation and requiring that 
``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{u}^{n+1} = 0`` yields a Poisson equation for the potential 
``\phi_{\rm{non}}`` at time-step ``n+1``:
```math
    \nabla^2 \phi_{\rm{non}}^{n+1} = \frac{\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{u}^{\star}}{\Delta t} \, .
```
With ``\boldsymbol{u}^\star`` and ``\phi_{\rm{non}}``, ``\boldsymbol{u}^{n+1}`` is then computed via the fractional step equation.

Tracers are stepped forward explicitly via
```math
    \begin{equation}
    \label{eq:tracer-timestep}
    c^{n+1} - c^n = \int_{t_n}^{t_{n+1}} G_c \, \mathrm{d} t \, ,
    \end{equation}
```
where 
```math
    G_c \equiv - \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{u} c \right ) - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c + F_c \, ,
```
and the same forward Euler or Adams-Bashforth scheme as for the explicit evaluation of the time-integral of
``\boldsymbol{G}_u`` is used to evaluate the integral of ``G_c``.
