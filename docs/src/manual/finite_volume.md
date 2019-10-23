# Finite volume method

The grid is defined by a Cartesian array of cuboids of horizontal dimensions $\Delta x, \Delta y$ and vertical dimension
$\Delta z$. The areas of the cell faces are given by
```math
    A_x = \Delta y \Delta z, \quad A_y = \Delta x \Delta z, \quad A_z = \Delta x \Delta y
```
so that each cell encloses a volume $V = \Delta x \Delta y \Delta z$.

The cells are indexed by $(i, j, k)$ where $i \in \{1, 2, \dots, N_x\}$, $j \in \{1, 2, \dots, N_y\}$, and
$k \in \{1, 2, \dots, N_z\}$ with $k=1$ corresponding to the top of the domain and $k=N_z$ corresponding to the bottom.
This has made a lot of people very angry and been widely regarded as a bad move.

In a finite volume method we work with the average quantity in the control volume, defined by
```math
  \overline{u} = \frac{1}{V} \int_\Omega u(\bm{x}) \; dV
```
where $\Omega$ denotes the control volume. We will always deal with the cell-averaged values so we will drop the
$\overline{\cdotp}$ notation from now on.

Integrating the momentum equations \eqref{eq:xMomentum}--\eqref{eq:zMomentum} over a control volume and approximating
the time derivative by a first-order forward Euler formula and evaluating the spatial derivatives at the current time
step we get an update formula for the momentum equation
```math
\renewcommand{\div}[1] {\nabla \cdotp \left ( #1 \right )}
\frac{\bm{u}^{n+1} - \bm{u}^n}{\Delta t}
  = -\bm{u}^n \cdotp \nabla\bm{u}^n - 2\bm{\Omega}\times\bm{u}^n - \nabla\phi^n + \div{\nu\nabla\bm{u}^n} + \bm{F}^n
```
and doing the same for the tracer equation \eqref{eq:tracer2} we get
```math
\renewcommand{\div}[1] {\nabla \cdot \left ( #1 \right )}
\frac{c^{n+1} - c^n}{\Delta t} = - \bm{u}^n \cdotp \nabla c^n + \div{\kappa_c \nabla c^n} + F_c^n
```
where $n$ denotes the current time step and $n+1$ is the next time step. Furthermore, the incompressibility condition
```math
\nabla \cdotp \bm{u}^{n+1} = 0
```
must be satisfied at all time steps.
