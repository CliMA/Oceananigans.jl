# Continuous equations

The governing equations solved by \texttt{Oceananigans.jl} are the rotating Navier-Stokes equations describing
viscous fluid flow with the Boussinesq approximation.[^1] The resulting mass conservation equation
imposes that the fluid flow is incompressible.[^2]

[^1]: Named after \citet{Boussinesq1903} although used earlier by \citet{Oberbeck1879}, the Boussinesq
    approximation neglects density differences in the momentum equation except when associated with the gravitational
    term. It is an accurate approximation for many flows, and especially so for oceanic flows where density differences 
    are very small. See \citet[\S2.4]{Vallis17} for an oceanographic introduction to the Boussinesq equations and
    \citet[\S2.A]{Vallis17} for an asymptotic derivation. See \citet[\S4.9]{Kundu15} for an engineering introduction.}

[^2]: Incompressibility rules out density and pressure waves like sound or shock waves.

## Rotating Boussinesq equations

The rotating Boussinesq equations can be written as
```math
  \D{\bm{u}}{t} + 2\bm{\Omega}\times\bm{u} + \frac{1}{\rho_0} \grad p - \div{\nu \grad \bm{u}}
    + g \frac{\rho\prime}{\rho_0} \hat{\bm{k}} - \bm{F} = 0 \label{eq:momentum}
```
along with the mass conservation equation
```math
    \grad \cdot \bm{u} = 0  \label{eq:continuity}
```
where $\uvec{i}, \uvec{j}, \uvec{k}$ are the standard Cartesian basis vectors, $\bm{u} = (u, v, w)$ is
the fluid flow velocity field, $\bm{\Omega}$ is the rotation vector, $p$ is the pressure, $\nu$ is the kinematic
viscosity, $g$ is the gravitational acceleration, $\rho_0$ is a reference density describing the base state of the
Boussinesq fluid whereas $\rho\prime = \rho - \rho_0$ denotes density variations, and $F = (F_u, F_v, F_w)$ includes
forcing terms (or rather, the sources and sinks). $\D{\bm{q}}{t} = \p{\bm{q}}{t} + \bm{u}\cdot\grad\bm{q}$ is the 
material derivative and $\grad = (\partial_x, \partial_y, \partial_z)$ is the del operator.

Tracer quantities $c$ such as temperature and salinity satisfy an advection-diffusion equation
```math
\D{c}{t} - \div{\kappa_c \grad c} + F_c = 0  \label{eq:tracer}
```
where $c$ is any tracer, $\kappa_c$ is the tracer diffusivity, and $F_c$ is a forcing term.

Tracer quantities, for example temperature $T$ and salinity $S$ in oceanographic applications, are related to the
density $\rho$ and pressure $p$ by an appropriate equation of state
```math
  \rho = \rho(T, S, p)
```
which may take multiple forms.

In order to discretize and discuss the equations, it will be helpful to expand out the material derivative and
write out the individual components of the momentum equation \eqref{eq:momentum} and tracer equation
\eqref{eq:tracer}.

We can write equations for the time derivatives[^3] of $u$, $v$, $w$, and $c$
```math
\begin{aligned}
  \partial_t u &= -\bm{u}\cdot\grad u + fv - \partial_x \phi + \div{\nu \grad u}      + F_u \label{eq:xMomentum} \\
  \partial_t v &= -\bm{u}\cdot\grad v - fu - \partial_y \phi + \div{\nu \grad v}      + F_v \label{eq:yMomentum} \\
  \partial_t w &= -\bm{u}\cdot\grad w      - \partial_z \phi + \div{\nu \grad w} + b  + F_w \label{eq:zMomentum} \\
  \partial_t c &= -\bm{u}\cdot\grad c                        + \div{\kappa_c \grad c} + F_c \label{eq:tracer2}
\end{aligned}
```
where we have rewritten the pressure gradient term as the gradient of the kinematic pressure $\phi = p/\rho_0$
and $b = -g\rho\prime/\rho_0$ is the buoyancy. We have also rewritten
$2\bm{\Omega}\times\bm{u} = -fv \bm{\hat{i}} + fu \bm{\hat{j}}$
where $f$ is the Coriolis parameter which on a rotating sphere can be expressed as $f = 2 \Omega \sin \varphi$
where $\Omega$ is the rotation rate of the sphere, and $\varphi$ is the latitude.[^4]

[^3]: In the geophysical sciences, the time derivatives are sometimes called the tendencies.

[^4]: It is important to note here that the full expression for the Coriolis force is given by 
    $$ 2\bm{\Omega}\times\bm{u} = (2\Omega w \cos\varphi - 2\Omega v \sin \varphi) \hat{\bm{i}}
       + 2\Omega u \sin\varphi \hat{\bm{j}} - 2\Omega u \cos\varphi \hat{\bm{k}} $$
    however the Coriolis terms involving the vertical velocity and $\cos\varphi$ term are neglected due to their small
    contribution in geophysical fluid dynamics on Earth. This is termed the \emph{traditional approximation} and must be
    taken with the \emph{shallow-fluid approximation}, which assumes the depth of the fluid is much shallower than the
    radius of the sphere on which it evolves, otherwise conservation of energy and angular momentum is not guaranteed.
    See \citet[\S2.2.4]{Vallis17} for an introductory discussion of these approximations, and
    \citet{Marshall97HY,White05} for a more detailed discussion.

## Hydrostatic and non-hydrostatic momentum equations

As a practical matter to allow the choice between evolving a hydrostatic and non-hydrostatic set of equations, we
split the kinematic pressure term into hydrostatic and non-hydrostatic parts,
```math
  \phi(x, y, z) = \phi_{HY}(x, y, z) + \phi_{NH}(x, y, z)
```
We then note that in the hydrostatic approximation, the pressure and buoyancy terms in the vertical momentum
equation are in balance 
```math
  \partial_z \phi_{HY} = -b
```
and so the $-\grad\phi + b\hat{\bm{k}}$ term in the momentum equation can be written as
```math
    -\grad\phi + b\hat{\bm{k}}
    = - \grad\phi_{NH} - \grad\phi_{HY} + b
    = - \grad\phi_{NH} - \partial_x \phi_{HY}^\prime \hat{\bm{i}} - \partial_y \phi_{HY}^\prime \hat{\bm{j}}
```
where ``\partial_x \phi_{HY} = \partial_x \phi_{HY}^\prime`` and ``\partial_y \phi_{HY} = \partial_y \phi_{HY}^\prime``
as ``\phi_{HY}^\prime`` denotes the *hydrostatic pressure anomaly*, which is the component of the pressure 
associated with buoyancy.

Thus the components of the momentum equation \eqref{eq:xMomentum}--\eqref{eq:zMomentum} can be written as
```math
\begin{aligned}
  \partial_t u &= -\bm{u}\cdot\grad u + fv - \partial_x (\phi_{NH} + \phi_{HY}^\prime) + \div{\nu \grad u} + F_u \\
  \partial_t v &= -\bm{u}\cdot\grad v - fu - \partial_y (\phi_{NH} + \phi_{HY}^\prime) + \div{\nu \grad v} + F_v \\
  \partial_t w &= -\bm{u}\cdot\grad w      - \partial_z  \phi_{NH}                  + \div{\nu \grad w} + F_w 
\end{aligned}
```

The non-hydrostatic pressure ``\phi_{NH}`` is associated with small-scale motions that are not in hydrostatic balance
and is numerically responsible for enforcing incompressibility, and thus mass conservation.

Note that we have not invoked the hydrostatic approximation. We are still dealing with a non-hydrostatic set of 
equations. We just made use of the hydrostatic approximation to manipulate the equations such that if we want to 
evolve a hydrostatic set of equations we can now just neglect the ``\grad\phi_{NH}`` terms (or multiply them by zero). 
So it becomes simple to switch between hydrostatic and non-hydrostatic modes.

