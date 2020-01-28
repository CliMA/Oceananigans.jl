# Boundary conditions

In Oceananigans.jl the user may impose \textit{no-penetration}, \textit{flux},
\textit{gradient} (Neumann), and \textit{value} (Dirichlet) boundary conditions in bounded,
non-periodic directions.
Note that the only boundary condition available for a velocity field normal to the bounded
direction is \textit{no-penetration}.

## Flux boundary conditions

A flux boundary condition prescribes flux of a quantity normal to the boundary.
For a tracer $c$ this corresponds to prescribing
```math
q_c \, |_b \equiv \bm{q}_c \bm{\cdot} \hat{\bm{n}} \, |_{\partial \Omega_b} \, ,
```
where $\partial \Omega_b$ is an external boundary.

## Gradient (Neumann) boundary condition

A gradient boundary condition prescribes the gradient of a field normal to the boundary.
For a tracer $c$ this prescribes
```math
\gamma \equiv \bm{\nabla} c \bm{\cdot} \hat{\bm{n}} \, |_{\partial \Omega_b} \, .
```

## Value (Dirichlet) boundary condition

A value boundary condition prescribes the value of a field on a boundary; for a tracer this
prescribes
```math
c_b \equiv c \, |_{\partial \Omega_b} \, .
```

## No penetration boundary condition

A no penetration boundary condition prescribes the velocity component normal to a boundary to be 0,
so that
```math
\bm{\hat{n}} \bm{\cdot} \bm{u} \, |_{\partial \Omega_b} = 0 \, .
```
