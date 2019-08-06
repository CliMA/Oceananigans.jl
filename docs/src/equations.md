```math
\newcommand{\p}[2]      {\frac{\partial #1}{\partial #2}}
\newcommand{\D}[2]      {\frac{D #1}{D #2}}
\newcommand{\b}[1]      {\boldsymbol{#1}}

\newcommand{\beq}       {\begin{equation}}
\newcommand{\eeq}       {\end{equation}}

\newcommand{\bnabla}    {\b{\nabla}}
\newcommand{\bnablah}   {\bnabla_h}

\newcommand{\v}        {\upsilon}
\newcommand{\bv}        {\b{\v}}
\newcommand{\bvh}       {\b{\v}_h}

\newcommand{\bnh}       {\b{\widehat{n}}}

\renewcommand{\div}[1]  {\bnabla \cdotp \left ( #1 \right )}
```

# Governing equations

The governing equations solved by Oceananigans.jl are the rotating Navier-Stokes equations describing viscous fluid
flow with the Boussinesq approximation which neglects density differences in the momentum equation, except when
associated with the gravitational term. The resulting mass conservation equation imposes that the fluid flow is
incompressible.

The rotating Boussinesq equations along with the mass conservation equation can be written as

```math
\begin{gather}
  \D{u}{t} = f\v - \frac{1}{\rho_0} \p{p}{x} + \div{\nu \bnabla u} + F_u      \label{eqn:xMomentum}  \\
  \D{v}{t} = - f u - \frac{1}{\rho_0} \p{p}{y} + \div{\nu \bnabla \v} + F_\v  \label{eqn:yMomentum}  \\
  \D{w}{t} = - \frac{1}{\rho_o} \p{p}{z} + \div{\nu \bnabla w} - g + F_w      \label{eqn:zMomentum}  \\
  \bnabla \cdotp \bv = 0                                                      \label{eqn:continuity}
\end{gather}
```
where $\bv = (u, \v, w)$ is the fluid flow velocity field, $\bvh = (u, \v)$ is the horizontal velocity, $p$ is the
pressure, $\nu$ is the kinematic viscosity, $g$ is the gravitational acceleration, and $\rho_0$ is a reference density
describing the base state of the Boussinesq fluid. $f = 2 \Omega \sin \varphi$ is the Coriolis frequency where $\Omega$
is the rotation rate and $\varphi$ is the latitude. $\bnabla = (\partial_x, \partial_y, \partial_z)$ is the del
operator, and $\bnablah = (\partial_x, \partial_y)$ is the horizontal del operator. $F_u$, $F_v$, and $F_w$ are forcing
terms.

Equations \eqref{eqn:xMomentum}--\eqref{eqn:zMomentum} describe the $x$, $y$, and $z$ momentum equations respectively
while equation \eqref{eqn:continuity} is the mass conservation equation (sometimes called the continuity equation).

Tracer quantities $\phi$ such as temperature and salinity satisfy an advection-diffusion equation

```math
  \p{\phi}{t} = -\div{\bv \phi} + \div{\kappa_\phi \bnabla \phi} + F_\phi
```

where $\phi$ is any tracer, $\kappa_\phi$ is the diffusivity which is different for each tracer, and $F_\phi$ is a
forcing term.

The tracer quantities, typically temperature $T$ and salinity $S$, are related to the density $\rho$ and pressure $p$
by an appropriate equation of state

```math
  \rho = \rho(T, S, p)
```

which may take many forms.
