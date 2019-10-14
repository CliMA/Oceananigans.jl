# Time stepping

If we combine all the terms that must be evaluated at time step $n + \frac{1}{2}$ into a variable $G$, then we have
```math
\renewcommand{\div}[1] {\nabla \cdotp \left ( #1 \right )}
\bm{G}_{\bm{u}}^n
  = -\left[ \bm{u} \cdot \nabla \bm{u} \right]^n - 2\bm{\Omega}\times\bm{u}^n + \div{\nu\nabla\bm{u}^n} + \bm{F}^n
```
where $\bm{G}_{\bm{u}} = (G_u, G_v, G_w)$. Together with \eqref{eq:projection-step} allows us to write the discretized
momentum equation as
```math
\frac{\bm{u}^{n+1} - \bm{u}^n}{\Delta t}
  = \bm{G}_{\bm{u}}^{n+1/2} - \nabla (\phi_{HY}^\prime + \phi_{NH})^{n+1}
```
where we have brought back the hydrostatic pressure anomaly $\phi_{HY}^\prime$ and non-hydrostatic pressure $\phi_{NH}$.

Doing the same for tracer quantities yields
```math
\renewcommand{\div}[1] {\nabla \cdotp \left ( #1 \right )}
G_c^n = \bm{u}^n \cdot \nabla c^n + \div{\kappa_c \nabla c^n} + F_c^n
```
and
```math
\frac{c^{n+1} - c^n}{\Delta t} = G_c^{n + \frac{1}{2}}
```

We evaluate the $G^{n + \frac{1}{2}}$ terms explicitly using a weighted two-step Adams–Bashforth (AB2) method
```math
    G^{n+\frac{1}{2}} = \left( \frac{3}{2} + \chi \right) G^n - \left( \frac{1}{2} + \chi \right) G^{n-1} .
```
AB2 has the advantage of being quasi-second-order accurate in time and yet does not have a computational mode (???).
Furthermore, it can be implemented by evaluating the source terms $G$ only once and storing them for use on the next
time step, thus using less memory than higher-order time stepping schemes.

It turns out that for a second-order accurate approximation of $G^{n+\frac{1}{2}}$ we require $\chi = \frac{1}{8}$
\citep{Ascher95}. Note that $\chi = 0$ reproduces the unweighted Adams-Bashforth method which calculates a second-order
accurate approximation of $G^{n+1}$. Also note that $\chi = -\frac{1}{2}$ reproduces the first-order accurate forward
Euler method, useful for initializing the model when $G^{n-1}$ is not available, such as at the first time step.
