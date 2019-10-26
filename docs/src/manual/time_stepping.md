# Time-stepping and the fractional step method

The time-integral of the momentum equation with the pressure decomposition from time step $n$ at $t = t_n$ 
to time step $n+1$ at $t_{n+1}$ is
```math
    \tag{eq:momentum-time-integral}
    \bm{u}^{n+1} - \bm{u}^n = 
        \int_{t_n}^{t_{n+1}} \Big [ - \bm{\nabla} \phi_{\rm{non}} 
                                    - \bm{\nabla}_{\! h} \phi_{\rm{hyd}} 
                                    - \left ( \bm{u} \bm{\cdot} \bm{\nabla} \right ) \bm{u} 
                                    - \bm{f} \times \bm{u} 
                                    + \bm{\nabla} \bm{\cdot} \bm{\tau} 
                                    + \bm{F}_{\bm{u}} \Big ] \, \rm{d} t \, ,
```
where the superscript $n$ and $n+1$ imply evaluation at $t_n$ and $t_{n+1}$, 
such that $\bm{u}^n \equiv \bm{u}(t=t_n)$.
The crux of the fractional step method is to treat the pressure term 
$\bm{\nabla} \phi_{\rm{non}}$ implicitly using the approximation
```math
\int_{t_n}^{t_{n+1}} \bm{\nabla} \phi_{\rm{non}} \, \rm{d} t \approx 
    \Delta t \bm{\nabla} \phi_{\rm{non}}^{n+1} \, ,
```
while treating the rest of the terms on the right hand side of \eqref{eq:momentum-time-integral} explicitly.
The implicit treatment of pressure ensures that the velocity field obtained at time step $n+1$ is divergence-free.

To effect such a fractional step method, we define an intermediate velocity field $\bm{u}^\star$ such that
```math
    \tag{eq:intermediate-velocity-field}
    \bm{u}^\star - \bm{u}^n = \int_{t_n}^{t_{n+1}} \bm{G}_{\bm{u}} \, \rm{d} t \, ,
```
where
```math
\bm{G}_{\bm{u}} \equiv - \bm{\nabla}_h \phi_{\rm{hyd}} 
                       - \left ( \bm{u} \bm{\cdot} \bm{\nabla} \right ) \bm{u} 
                       - \bm{f} \times \bm{u} 
                       + \bm{\nabla} \bm{\cdot} \bm{\tau} 
                       + \bm{F}_{\bm{u}}
```
collects all terms on the right side of the time-integral of the momentum equation except the contribution 
of non-hydrostatic pressure $\bm{\nabla} \phi_n$.
The integral on the right of the equation for $\bm{u}^\star$ may be approximated by a variety of 
explicit methods: for example, a forward Euler method uses
```math
    \int_{t_n}^{t_{n+1}} G \, \rm{d} t \approx \Delta t G^n \, ,
    \tag{eq:forward-euler}
```
for any time-dependent function $G(t)$, while a second-order Adams-Bashforth method uses the approximation
```math
    \tag{eq:adams-bashforth}
    \int_{t_n}^{t_{n+1}} G \, \rm{d} t \approx 
        \Delta t \left [ \left ( \tfrac{3}{2} + \chi \right ) G^n 
        - \left ( \tfrac{1}{2} + \chi \right ) G^{n-1} \right ] \, ,
```
where $\chi$ is a parameter. Ascher et al. (1995) claim that $\chi = \tfrac{1}{8}$ is optimal; 
$\chi=-\tfrac{1}{2}$ yields the forward Euler scheme.

Combining the equations for $\bm{u}^\star$ and the time integral of the momnentum equation yields
```math
    \tag{eq:fractional-step}
    \bm{u}^{n+1} - \bm{u}^\star = - \Delta t \bm{\nabla} \phi_{\rm{non}}^{n+1} \, \rm{d} t \, .
```
Taking the divergence of fractional step equation and requiring that 
$\bm{\nabla} \bm{\cdot} \bm{u}^{n+1} = 0$ yields a Poisson equation for the potential 
$\phi_{\rm{non}}$ at time-step $n+1$:
```math
    \bm{\nabla}^2 \phi_{\rm{non}}^{n+1} = \frac{\bm{\nabla} \bm{\cdot} \bm{u}^{\star}}{\Delta t} \, .
```
With $\bm{u}^\star$ and $\phi_{\rm{non}}$, $\bm{u}^{n+1}$ is then computed via the fractional step equation.

Tracers are stepped forward explicitly via
```math
    \tag{eq:tracer-timestep}
    c^{n+1} - c^n = \int_{t_n}^{t_{n+1}} G_c \, \rm{d} t \, ,
```
where 
```math
    G_c \equiv - \bm{\nabla} \bm{\cdot} \left ( \bm{u} c \right ) - \bm{\nabla} \bm{\cdot} \bm{q}_c + F_c \, ,
```
and the same forward Euler or Adams-Bashforth scheme as for the explicit evaluation of the time-integral of
$\bm{G}_u$ is used to evaluate the integral of $G_c$.
