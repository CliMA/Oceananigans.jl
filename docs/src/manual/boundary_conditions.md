# [Numerical implementation of boundary conditions](@id numerical_bcs)

We adopt a mixed approach for implementing boundary conditions that uses both halo regions and "direct"
imposition of boundary conditions, depending on the condition prescribed.
We illustrate how boundary conditions are implemented by considering the tracer equation \eqref{eq:tracer}.

## Gradient boundary conditions

Users impose gradient boundary conditions by prescribing the gradient $\gamma$ of a field $c$ across a
*external boundary* $\partial \Omega_b$.
The prescribed gradient $\gamma$ may be a constant, discrete array of values, or an arbitrary function.
The gradient boundary condition is enforced setting the value of halo points located outside the domain interior such that
```math
    \tag{eq:gradient-bc}
    \hat{\bm{n}} \bm{\cdot} \bm{\nabla} c |_{\partial \Omega_b} = \gamma \, .
```
where $\hat{\bm{n}}$ is the vector normal to $\partial \Omega_b$.

Across the bottom boundary in $z$, for example, this requires that
```math
    \tag{eq:linear-extrapolation}
    c_{i, j, 0} = c_{i, j, 1} + \gamma_{i, j, 1} \tfrac{1}{2} \left ( \Delta z_{i, j, 1} + \Delta z_{i, j, 0} \right ) \, ,
```
where $\Delta z_{i, j, 1} = \Delta z_{i, j, 0}$ are the heights of the finite volume at $i, j$ and $k=1$ and $k=0$.
This prescription implies that the $z$-derivative of $c$ across the boundary at $k=1$ is
```math
    \partial_z c \, |_{i, j, 1} \equiv
        \frac{c_{i, j, 1} - c_{i, j, 0}}{\tfrac{1}{2} \left ( \Delta z_{i, j, 1} + \Delta z_{i, j, 0} \right )}
            = \gamma_{i, j, 1} , ,
```
as prescribed by the user.

## Value boundary conditions

Users impose value boundary conditions by prescribing $c^b$, the value of $c$ on the external
boundary $\partial \Omega_b$.
The value $c^b$ may be a constant, array of discrete values, or an arbitrary function.
To enforce a value boundary condition, the gradient associated with the difference between
$c^b$ and $c$ at boundary-adjacent nodes is diagnosed and used to set the value of the $c$ halo point
located outside the boundary.

At the bottom boundary in $z$, for example, this means that the gradient of $c$ is determined by
```math
\gamma = \frac{c_{i, j, 1} - c^b_{i, j, 1}}{\tfrac{1}{2} \Delta z_{i, j, 1}} \, ,
```
which is then used to set the halo point $c_{i, j, 0}$ via linear extrapolation.

## Flux boundary conditions

Users impose flux boundary conditions by prescribing the flux $q_c \, |_b$ of $c$ across
the external boundary $\partial \Omega_b$.
The flux $q_c \, |_b$ may be a constant, array of discrete values, or arbitrary function.
To explain how flux boundary conditions are imposed in `Oceananigans.jl`, we note that
the average of the tracer conservation equation over a finite volume yields
```math
    \partial_t c_{i, j, k} = \underbrace{ - \frac{1}{V_{i, j, k}} \oint_{\partial \Omega_{i, j, k}} \bm{u} c + \bm{q}_c \, \rm{d} S
                                          + \frac{1}{V_{i, j, k}} \int_{V_{i, j, k}} F_c \, \rm{d} V }_{\equiv G_c |_{i, j, k}} \, ,
```
where the surface integral over $\partial \Omega_{i, j, k}$ averages the flux of $c$ across the six faces of the finite volume.

An external boundary of a finite volume is associated with a no-penetration condition such that
$\hat{\bm{n}} \bm{\cdot} \bm{u} \, |_{\partial \Omega_b} = 0$, where $\hat{\bm{n}}$ is the vector normal to $\partial \Omega_b$.
Furthermore, the closures currently available in \texttt{Oceananigans.jl} have the property that $\bm{q}_c \propto \bm{\nabla} c$.
Thus setting $\hat{\bm{n}} \bm{\cdot} \bm{\nabla} c \, |_{\partial \Omega_b} = 0$ on the external boundary implies that the total
flux of $c$ across the external boundary is
```math
\hat{\bm{n}} \bm{\cdot} \left ( \bm{u} c + \bm{q}_c \right ) |_{\partial \Omega_b} = 0 \, .
```
`Oceananigans.jl` exploits this fact to define algorithm that prescribe fluxes across external boundaries $\partial \Omega_b$:

1. Impose a constant gradient $\hat{\bm{n}} \bm{\cdot} \bm{\nabla} c \, |_{\partial \Omega_b} = 0$ across external boundaries
    via using halo points (similar to \eqref{eq:gradient-bc}), which ensures that the evaluation of $G_c$ in boundary-adjacent
    cells does not include fluxes across the external boundary, and;
2. Add the prescribed flux to the boundary-adjacent volumes prior to calculating $G_c$

    $G_c \, |_b = G_c \, |_b - \frac{A_b}{V_b} q_c \, |_b \, \text{sign}(\hat{\bm{n}}) \, ,$

    where $G_c \, |_b$ denotes values of $G_c$ in boundary-adjacent volumes, $q_c \, |_b$ is the flux prescribed along the boundary,
    $V_b$ is the volume of the boundary-adjacent cell, and $A_b$ is the area of the external boundary of the boundary-adjacent cell.

    The factor $\text{sign}(\hat{\bm{n}})$ is -1 and +1 on "left" and "right" boundaries, and accounts for the fact that a positive
    flux on a left boundary where $\text{sign}(\hat{\bm{n}}) = -1$ implies an "inward" flux of $c$ that increases interior values of $c$,
    whereas a positive flux on a right boundary where $\text{sign}(\hat{\bm{n}}) = 1$ implies an "outward" flux that decreases interior
    values of $c$.
