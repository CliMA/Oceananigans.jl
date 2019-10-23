# Fractional step method

Solving the momentum equation \eqref{eq:momentumFV} coupled with the continuity equation \eqref{eq:continuityFV} can be
cumbersome so instead we employ a fractional step method. To approximate the solution of the coupled system we first
solve an approximation to the discretized momentum equation \eqref{eq:momentumFV} for an intermediate velocity field
$\bm{u}^\star$ without worrying about satisfying the incompressibility constraint. We then project $\bm{u}^\star$ onto
the space of divergence-free velocity fields to obtain a value for $\bm{u}^{n+1}$ that satisfies
\eqref{eq:continuityFV}.

We thus discretize the momentum equation as
```math
\renewcommand{\div}[1] {\nabla \cdotp \left ( #1 \right )}
\frac{\bm{u}^\star - \bm{u}^n}{\Delta t}
  = - \left[ \bm{u} \cdot \nabla\bm{u} \right]^{n+\frac{1}{2}}
  - 2\bm{\Omega}\times\bm{u}^{n+\frac{1}{2}}
  + \div{\nu\nabla\bm{u}^{n+\frac{1}{2}}}
  + \bm{F}^{n+\frac{1}{2}}
```
where the superscript $n + \frac{1}{2}$ indicates that these terms are evaluated at time step $n + \frac{1}{2}$, which
we compute explicitly (see \S\ref{sec:time-stepping}).

The projection is then performed
```math
   \bm{u}^{n+1} = \bm{u}^\star - \Delta t \nabla \phi^{n+1}
```
to obtain a divergence-free velocity field $\bm{u}^{n+1}$. Here the projection is performed by solving an elliptic
problem for the pressure $\phi^{n+1}$ with the boundary condition
```math
\newcommand{\uvec}[1]{\boldsymbol{\hat{\textbf{#1}}}}
  \bm{\hat{n}} \cdotp \nabla\phi^{n+1} |_{\partial\Omega} = 0
```

\citet{Orszag86} and \citet{Brown01} raise an important issue regarding these fractional step methods, which is that
"while the velocity can be reliably computed to second-order accuracy in time and space, the pressure is typically only
first-order accurate in the $L_\infty$-norm." The numerical boundary conditions must be carefully accounted for to
ensure the second-order accuracy promised by the fractional step methods.

We are currently investigating whether our projection method is indeed second-order accurate in both velocity and
pressure (see \S\ref{sec:forced-flow}). However, it may not matter too much for simulating high Reynolds number
geophysical fluids as \citet{Brown01} conclude that "Quite often, semi-implicit projection methods are applied to
problems in which the viscosity is small. Since the predicted first-order errors in the pressure are scaled by $\nu$,
it is not clear whether the improved pressure-update formula is beneficial in such situations. ... Finally, in some
applications of projection methods, second-order accuracy in the pressure may not be relevant or in some cases even
possible due to the treatment of other terms in the equations."
