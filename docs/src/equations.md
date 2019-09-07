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
viscous fluid flow with the Boussinesq approximation\footnotemark. The resulting mass conservation equation imposes
that the fluid flow is incompressible.

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
