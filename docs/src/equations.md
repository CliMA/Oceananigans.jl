```math
\newcommand{\p}[2]  {\frac{\partial #1}{\partial #2}}
\newcommand{\D}[2]  {\frac{D #1}{D #2}}
\newcommand{\b}[1]  {\boldsymbol{#1}}

\newcommand{\grad}  {\b{\nabla}}

\newcommand{\v}    {\upsilon}
\newcommand{\bv}   {\b{\v}}
\newcommand{\bvh}  {\b{\v}_h}
\newcommand{\bnh}  {\b{\widehat{n}}}

\renewcommand{\div}[1]  {\bnabla \cdotp \left ( #1 \right )}
```

# Governing equations

The governing equations solved by \texttt{Oceananigans.jl} are the rotating Navier-Stokes equations describing
viscous fluid flow with the Boussinesq approximation.\footnotemark[1] The resulting mass conservation equation
imposes that the fluid flow is incompressible.\footnotemark[2]

\footnotetext[1]{Named after \citet{Boussinesq1903} although used earlier by \citet{Oberbeck1879}, the Boussinesq
approximation neglects density differences in the momentum equation except when associated with the gravitational
term. It is an accurate approximation for many flows, and especially so for oceanic flows where density differences
are very small. See \citet[\S2.4]{Vallis17} for an oceanographic introduction to the Boussinesq equations and
\citet[\S2.A]{Vallis17} for an asymptotic derivation. See \citet[\S4.9]{Kundu15} for an engineering introduction.}

\footnotetext[2]{Incompressibility rules out density and pressure waves like sound or shock waves.}

The rotating Boussinesq equations can be written as
\begin{equation}
  \D{\bm{u}}{t} + 2\bm{\Omega}\times\bm{u} + \frac{1}{\rho_0} \grad p - \div{\nu \grad \bm{u}}
    + g \frac{\rho\prime}{\rho_0} \hat{\bm{k}} + \bm{F} = 0 \label{eq:momentum}
\end{equation}
along with the mass conservation equation
\begin{equation}
    \grad \cdotp \bm{u} = 0  \label{eq:continuity}
\end{equation}
where $\hat{\bm{i}}, \hat{\bm{j}}, \hat{\bm{k}}$ are the standard Cartesian basis vectors, $\bm{u} = (u, v, w)$ is
the fluid flow velocity field, $\bm{\Omega}$ is the rotation vector, $p$ is the pressure, $\nu$ is the kinematic
viscosity, $g$ is the gravitational acceleration, $\rho_0$ is a reference density describing the base state of the
Boussinesq fluid whereas $\rho\prime = \rho - \rho_0$ denotes density variations, and $F = (F_u, F_v, F_w)$ includes
forcing terms (or rather, the sources and sinks). $\D{\bm{q}}{t} = \p{\bm{q}}{t} + \bm{u}\cdotp\grad\bm{q}$ is the
material derivative and $\grad = (\partial_x, \partial_y, \partial_z)$ is the del operator.

Tracer quantities $c$ such as temperature and salinity satisfy an advection-diffusion equation
\begin{equation}
  \D{c}{t} - \div{\kappa_c \grad c} + F_c = 0  \label{eq:tracer}
\end{equation}
where $c$ is any tracer, $\kappa_c$ is the tracer diffusivity, and $F_c$ is a forcing term.

Tracer quantities, for example temperature $T$ and salinity $S$ in oceanographic applications, are related to the
density $\rho$ and pressure $p$ by an appropriate equation of state

\begin{equation}
  \rho = \rho(T, S, p)
\end{equation}

which may take multiple forms.

## Manipulation of the governing equations
In order to discretize and discuss the equations, it will be helpful to expand out the material derivative and
write out the individual components of the momentum equation \eqref{eq:momentum}.

We can write equations for the time derivatives of $u$, $v$, and $w$\footnotemark[3]
\begin{align}
  \p{u}{t} &= -\bm{u}\cdotp\grad u + fv - \p{\phi}{x} + \div{\nu \grad u}     + F_u \label{eqn:xMomentum}  \\
  \p{v}{t} &= -\bm{u}\cdotp\grad v - fu - \p{\phi}{y} + \div{\nu \grad v}     + F_v \label{eqn:yMomentum}  \\
  \p{w}{t} &= -\bm{u}\cdotp\grad w      - \p{\phi}{z} + \div{\nu \grad w} + b + F_w \label{eqn:zMomentum}
\end{align}
where we have rewritten the pressure gradient term as the gradient of the kinematic pressure $\phi = p/\rho_0$
and $b = -g\rho\prime/\rho_0$ is the buoyancy. We have also rewritten
$2\bm{\Omega}\times\bm{u} = -fv \bm{\hat{i}} + fu \bm{\hat{j}}$
where $f$ is the Coriolis parameter which on a rotating sphere can be expressed as $f = 2 \Omega \sin \varphi$
where $\Omega$ is the rotation rate of the sphere, and $\varphi$ is the latitude.\footnotemark[4]

\footnotetext[3]{In the geophysical sciences, the time derivatives are sometimes called the tendencies.}

\footnotetext[3]{It is important to note here that the full expression for the Coriolis force is given by
$$ 2\bm{\Omega}\times\bm{u} = (2\Omega w \cos\varphi - 2\Omega v \sin \varphi) \hat{\bm{i}}
   + 2\Omega u \sin\varphi \hat{\bm{j}} - 2\Omega u \cos\varphi \hat{\bm{k}} $$
however the Coriolis terms involving the vertical velocity and $\cos\varphi$ term are neglected due to their small
contribution in geophysical fluid dynamics on Earth. This is termed the \emph{traditional approximation} and must be
taken with the \emph{shallow-fluid approximation}, which assumes the depth of the fluid is much shallower than the
radius of the sphere on which it evolves, otherwise conservation of energy and angular momentum is not guaranteed.
See \citet[\S2.2.4]{Vallis17} for an introductory discussion of these approximations, and
\citet{Marshall97HY,White05} for a more detailed discussion.}

As a practical matter to allow the choice between evolving a hydrostatic and non-hydrostatic set of equations, we
split the kinematic pressure term into hydrostatic and non-hydrostatic parts,
\begin{equation}
  \phi(x, y, z) = \phi_{HY}(x, y, z) + \phi_{NH}(x, y, z)
\end{equation}
We then note that in the hydrostatic approximation, the pressure and buoyancy terms in the vertical momentum
equation are in balance
\begin{equation}
  \p{\phi_{HY}}{z} = -b
\end{equation}
and so the $-\grad\phi + b\hat{\bm{k}}$ term in the momentum equation can be written as
\begin{equation}
    -\grad\phi + b\hat{\bm{k}}
    = - \grad\phi_{NH} - \grad\phi_{HY} + b
    = - \grad\phi_{NH} - \p{\phi_{HY}^\prime}{x} \hat{\bm{i}} - \p{\phi_{HY}^\prime}{y} \hat{\bm{j}}
\end{equation}
where $\partial_x \phi_{HY} = \partial_x \phi_{HY}^\prime$ and $\partial_y \phi_{HY} = \partial_y \phi_{HY}^\prime$
as $\phi_{HY}^\prime$ denotes the \emph{hydrostatic pressure anomaly}, which is the component of the pressure
associated with buoyancy.

Thus the components of the momentum equation \eqref{eqn:xMomentum}--\eqref{eqn:zMomentum} can be written as
\begin{align}
  \p{u}{t} &= -\bm{u}\cdotp\grad u + fv - \p{\phi_{NH}}{x} -\p{\phi_{HY}^\prime}{x} + \div{\nu \grad u} + F_u \\
  \p{v}{t} &= -\bm{u}\cdotp\grad v - fu - \p{\phi_{NH}}{y} -\p{\phi_{HY}^\prime}{x} + \div{\nu \grad v} + F_v \\
  \p{w}{t} &= -\bm{u}\cdotp\grad w      - \p{\phi_{NH}}{z}                          + \div{\nu \grad w} + F_w
\end{align}

The non-hydrostatic pressure is associated with small-scale motions that are not in hydrostatic balance and is
numerically responsible for enforcing incompressibility, and thus mass conservation.

Note that we have not invoked the hydrostatic approximation. We are still dealing with a non-hydrostatic set of
equations. We just made use of the hydrostatic approximation to manipulate the equations such that if we want to
evolve a hydrostatic set of equations we can now just neglect the $\grad\phi_{NH}$ terms (or multiply them by zero).
So it becomes simple to switch between hydrostatic and non-hydrostatic modes.
