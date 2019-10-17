# Turbulence closures

To truly simulate and resolve turbulence at high Reynolds number (so basically all interesting flows) would require
you resolve all motions down to the \citet{Kolmogorov41} length scale $\eta = (\nu^3 / \varepsilon)^{1/4}$ where
$\nu$ is the kinematic viscosity and $\varepsilon$ the average rate of dissipation of turbulence kinetic energy per
unit mass.

As pointed out way back by \citet{Corrsin61}, to run a simulation on a horizontal domain about 10 times the size of an
"average eddy" with 100 vertical levels and where the grid spacing is given by $\eta$ would require the computer to
store on the order of $10^{14}$ variables.[^1] This is still impractical today, although may be within
reach in less than a decade. He ends by suggesting the use of an analog rather digital computer---a tank of water.

[^1]: And even then, $\eta$ gives the *maximum* allowable grid spacing. There is significant flow structure
    smaller than $\eta$.

To have any hope of simulating high Reynolds number flows we need some way of resolving the sub-grid scale motions.[^2]

[^2]: In reality there is no need to resolve all motions down to the Kolmogorov length scale to achieve
    acceptable accuracy. Perhaps good results can be achieved if 80\% of the kinetic energy is resolved
    \citep[\S13]{Pope00}.


## Reynolds-averaged Navier–Stokes equations

Following \citet{Reynolds1895} we can decompose flow variables such as velocity $\bm{u}$ into the mean component
$\overline{\bm{u}}$ and the fluctuating component $\bm{u}^\prime$ so that $\bm{u} = \overline{\bm{u}} + \bm{u}^\prime$
[see \citet[\S4]{Pope00} for a modern discussion].

Expressing the Navier-Stokes equations in tensor notation
```math
\begin{aligned}
    \partial_i u_i &= 0 \\
    \partial_t u_i + u_j \partial_j u_i &= f_i - \alpha\partial_i p + \nu \partial_j \partial_j u_i
\end{aligned}
```
where $\alpha = \rho^{-1}$ is the specific volume and $f_i$ represents external forces. We can plug in the Reynolds
decomposition for $\bm{u}$ and after some manipulation arrive at the following form for the *Reynolds-averaged
Navier-Stokes equations*
```math
\begin{aligned}
    \partial_i \overline{u}_i &= 0 \\
    \partial_t \overline{u}_i + \overline{u}_j \partial_j \overline{u}_i &= \overline{f}_i -
    \partial_j \left(-\alpha\overline{p}\delta_{ij} + 2\nu \overline{S}_{ij} - \overline{u_i^\prime u_j^\prime}\right)
\end{aligned}
```
where
```math
\overline{S}_{ij} = \frac{1}{2} ( \partial_j \overline{u}_i + \partial_i \overline{u}_j )
```
is the mean rate of strain tensor.

Thanks to the non-linearity of the Navier-Stokes equations, even when averaged we are left with pesky fluctuation
terms which form the components of the *Reynolds stress tensor*
```math
\tau_{ij} = \rho \overline{u_i^\prime u_j^\prime}
```
Attempting to close the equations leads to the *closure problem*: the time evolution of the Reynolds stresses
depends on  triple covariances $\overline{u_i^\prime u_j^\prime u_k^\prime}$ and covariances with pressure, which depend
on quadruple covariances and so on \citep{Chou45}.

This is kind of hopeless so we will have to find some way to model the Reynolds stresses.

## Gradient-diffusion hypothesis and eddy viscosity models

The *gradient-diffusion hypothesis*, due to \citet{Boussinesq1877}, assumes that the transport of scalar fluxes
such as $\overline{\bm{u}^\prime c^\prime}$ and $\overline{u_i^\prime u_j^\prime}$ occurs down the mean scalar gradient
$\grad c$ as if they are being diffused \citep[\S4.4]{Pope00}. This is in analogy with how momentum transfer by
molecular motion in a gas can be described by a molecular viscosity.

Taking this assumption we can express the Reynolds stresses and turbulent tracer fluxes in terms of the mean variables
and close the equations
```math
\overline{\bm{u}^\prime c^\prime} = -\kappa_e \nabla \overline{c}
\quad \text{and} \quad
\overline{u_i^\prime u_j^\prime} = -2\nu_e \overline{S}_{ij}
```
where $\nu_e = \nu_e(\bm{x}, t)$ is the turbulent or *eddy viscosity* and $\kappa_e = \kappa_e(\bm{x}, t)$
is the *eddy diffusivity*.

The effective diffusivity ends up being the sum of the molecular and eddy diffusivities. So just by using an elevated
value for the viscosity and diffusivity, you are already using an eddy viscosity model.

The eddy viscosity model is simple and for that reason is very popular. It can work well even with a constant eddy
diffusivity. However, it does assume that the flux is aligned down gradient, which is not true even in simple turbulent
flows as the physics of turbulence is quite different from that of colliding molecules leading to the viscous stress law
\citep[\S4.4,10.1]{Pope00}. So we might want something a little bit more sophisticated.
