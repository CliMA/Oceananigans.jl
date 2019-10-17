# Taylor-Green vortex
An exact solution to the two-dimensional incompressible Navier-Stokes equations is given by \citet{Taylor37} describing
the unsteady flow of a vortex decaying under viscous dissipation. The viscous terms balance the time derivatives while
the nonlinear advection terms balance the pressure gradient term. We use the doubly-periodic solution described by
\citet[p. 310]{Hesthaven07}

```math
\begin{aligned}
  u(x, y, t) &= -\sin(2\pi y) e^{-4\pi^2\nu t} \\
  v(x, y, t) &=  \sin(2\pi x) e^{-4\pi^2\nu t} \\
  p(x, y, t) &= -\cos(2\pi x) \cos(2\pi y) e^{-8\pi^2\nu t}
\end{aligned}
```
