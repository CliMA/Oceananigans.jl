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

The governing equations solved by Oceananigans.jl are the rotating Navier-Stokes equations describing viscous fluid
flow with the Boussinesq approximation which neglects density differences in the momentum equation, except when
associated with the gravitational term. The resulting mass conservation equation imposes that the fluid flow is
incompressible.

The rotating Boussinesq equations along with the mass conservation equation can be written as

```math
\begin{gather}
  \D{\bv}{t} + 2\b{\Omega}\times\bv + \frac{1}{\rho_0} \grad p - \div{\nu \grad \bv} + g\hat{\b{k}} + \b{F}_\bv = 0 \label{eq:momentum} \\
  \bnabla \cdotp \bv = 0  \label{eq:continuity}
\end{gather}
```

where $\D{\bv}{t} = \p{\bv}{t} + \bv\cdotp\grad\bv$ is the material derivative and $\bnabla = (\partial_x, \partial_y,
\partial_z)$ is the del operator. $\bv = (u, \v, w)$ is the fluid flow velocity field, $\b{\Omega}$ is the rotation
vector, $p$ is the pressure, $\nu$ is the kinematic viscosity, $g$ is the gravitational acceleration, and $\rho_0$ is a
reference density describing the base state of the Boussinesq fluid. $F_u$, $F_v$, and $F_w$ are forcing terms.

Tracer quantities $Q$ such as temperature and salinity satisfy an advection-diffusion equation

```math
\begin{equation}
  \p{Q}{t} = -\div{\bv Q} + \div{\kappa_\phi \bnabla Q} + F_\phi  \label{eq:tracer}
\end{equation}
```

where $Q$ is any tracer, $\kappa_Q$ is the tracer diffusivity (different for each tracer), and $F_Q$ is a
forcing term.

The tracer quantities, typically temperature $T$ and salinity $S$, are related to the density $\rho$ and pressure $p$
by an appropriate equation of state

```math
  \rho = \rho(T, S, p)
```

which may take multiple forms.

## Manipulation of the governing equations
By expanding out the material derivative and writing out the individual components of the momentum equation
\eqref{eq:momentum}, we can write equations for the time derivatives of $u$, $\v$, and $w$, sometimes called their
tendencies

```math
\begin{align}
  \p{u}{t} &= -\bv\cdotp\grad u + f\v &- \p{\phi}{x} + \div{\nu \grad u}     &+ F_u  \label{eqn:xMomentum}  \\
  \p{v}{t} &= -\bv\cdotp\grad\v - f u &- \p{\phi}{y} + \div{\nu \grad \v}    &+ F_\v \label{eqn:yMomentum}  \\
  \p{w}{t} &= -\bv\cdotp\grad w       &- \p{\phi}{z} + \div{\nu \grad w} - g &+ F_w  \label{eqn:zMomentum}
\end{align}
```

where we have rewritten $2\b{\Omega}\times\bv = f\v \b{\hat{i}} - fu \b{\hat{j}}$ where $f$ is the Coriolis parameter
which on a rotating sphere can be expressed as $f = 2 \Omega \sin \varphi$ where $\Omega$ is the rotation rate of the
sphere, and $\varphi$ is the latitude. We have also rewritten the pressure gradient force as the gradient of the
kinematic pressure $\phi = p/\rho_0$.

###

```math
\begin{gather}
        \p{\bvh}{t} = \b{G}_{\bv h} - \frac{1}{\rho_0} \bnabla_h p ,  \label{eqn:horizontalMomentum} \\
           \p{w}{t} = G_w - \frac{1}{\rho_0} \p{p}{z} ,               \label{eqn:verticalMomentum} \\
 \bnabla \cdotp \bv = 0 ,                                             \label{eqn:continuity} \\
           \p{T}{t} = G_T ,                                           \label{eqn:TTendency} \\
           \p{S}{t} = G_S ,                                           \label{eqn:STendency} \\
               \rho = \rho(T,S,p) ,                                   \label{eqn:EOS}
\end{gather}
```


Equations \eqref{eqn:TTendency} and \eqref{eqn:STendency} prognostic equations describing the time evolution of temperature $T$ and salinity $S$.
Equation \eqref{eqn:EOS} is an equation of state for seawater giving the density $\rho$ in terms of $T$, $S$, and $p$.
The source terms $\b{G}_{\bv} = (\b{G}_{\bv h}, G_w) = (G_u, G_v, G_w)$ in \eqref{eqn:horizontalMomentum} and \eqref{eqn:verticalMomentum}
represent inertial, Coriolis, gravitational, forcing, and dissipation terms:

```math
\begin{align}
    G_u &= -\bv \cdotp \bnabla u + f\v - \frac{1}{\rho_0} \p{p'_{HY}}{x} + \div{\nu \bnabla u} + F_u  ,\\
    G_\v &= -\bv \cdotp \bnabla \v - f u - \frac{1}{\rho_0} \p{p'_{HY}}{y} + \div{\nu \bnabla \v} + F_\v  ,\\
    G_w &= -\bv \cdotp \bnabla w                                        + \div{\nu \bnabla w} + F_w ,
\end{align}
```

where $f = 2 \Omega \sin \phi$ is the Coriolis frequency, $\Omega$ is the rotation rate of the Earth, $\phi$ is the latitude, $p_{HY}$ is the hydrostatic pressure anomaly, and $\nu$ is the viscosity. $F_u$, $F_\v$, and $F_w$ represent other forcing terms that may be imposed.
Note that the buoyancy term $-g \delta \rho / \rho_0$ (with $g$ the acceleration due to gravity) that is usually present in the vertical momentum equation has been expressed in terms
of the hydrostatic pressure anomaly $p_{HY}$ which ends up in the horizontal momentum equations. (This step will be shown in an appendix.)

Similarly, the source terms for the tracer quantities can be written as

```math
\beq
  G_T = -\div{\bv T} + \kappa \nabla^2 T + F_T ,
  \label{eqn:G_T}
\eeq
```

```math
\beq
  G_S = -\div{\bv S} + \kappa \nabla^2 S + F_S ,
  \label{eqn:G_S}
\eeq
```

where $\kappa$ is the diffusivity while $F_T$ and $F_S$ represent forcing terms.

The associated boundary conditions for the embedded non-hydrostatic models is periodic in the horizontal direction and a
rigid boundary or "lid" at the top and bottom. The rigid lid approximation sets $w = 0$ at the vertical boundaries so
that it does not move but still allows a pressure to be exerted on the fluid by the lid.
