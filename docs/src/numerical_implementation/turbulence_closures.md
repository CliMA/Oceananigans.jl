# [Turbulence closures](@id numerical_closures)

To truly simulate and resolve turbulence at high Reynolds number (so basically all interesting flows) would require
you resolve all motions down to the [Kolmogorov41](@cite) length scale ``\eta = (\nu^3 / \varepsilon)^{1/4}`` where
``\nu`` is the kinematic viscosity and ``\varepsilon`` the average rate of dissipation of turbulence kinetic energy per
unit mass.

As pointed out way back by [Corrsin61](@cite), to run a simulation on a horizontal domain about 10 times the size of an
"average eddy" with 100 vertical levels and where the grid spacing is given by ``\eta`` would require the computer to
store on the order of ``10^{14}`` variables.[^1] This is still impractical today, although may be within
reach in less than a decade. He ends by suggesting the use of an analog rather digital computer---a tank of water.

[^1]: And even then, ``\eta`` gives the *maximum* allowable grid spacing. There is significant flow structure
    smaller than ``\eta``.

To have any hope of simulating high Reynolds number flows we need some way of resolving the sub-grid scale motions.[^2]

[^2]: In reality there is no need to resolve all motions down to the Kolmogorov length scale to achieve
    acceptable accuracy. Perhaps good results can be achieved if 80\% of the kinetic energy is resolved
    (§13) [Pope00](@cite).


## Reynolds-averaged Navier–Stokes equations

Following [Reynolds1895](@cite) we can decompose flow variables such as velocity ``\boldsymbol{v}`` into the mean component
``\overline{\boldsymbol{v}}`` and the fluctuating component ``\boldsymbol{v}^\prime`` so that ``\boldsymbol{v} = \overline{\boldsymbol{v}} + \boldsymbol{v}^\prime``
[see §4 of [Pope00](@cite) for a modern discussion].

Expressing the Navier-Stokes equations in tensor notation
```math
\begin{align}
    \partial_i v_i &= 0  \, ,\\
    \partial_t v_i + v_j \partial_j v_i &= f_i - \alpha\partial_i p + \nu \partial_j \partial_j v_i \, ,
\end{align}
```
where ``\alpha = \rho^{-1}`` is the specific volume and ``f_i`` represents external forces. We can plug in the Reynolds
decomposition for ``\boldsymbol{v}`` and after some manipulation arrive at the following form for the *Reynolds-averaged
Navier-Stokes equations*
```math
\begin{align}
    \partial_i \overline{u}_i &= 0  \, ,\\
    \partial_t \overline{u}_i + \overline{u}_j \partial_j \overline{u}_i &= \overline{f}_i -
    \partial_j \left(-\alpha\overline{p}\delta_{ij} + 2\nu \overline{S}_{ij} - \overline{v_i^\prime v_j^\prime}\right) \, ,
\end{align}
```
where
```math
\overline{S}_{ij} = \frac{1}{2} ( \partial_j \overline{u}_i + \partial_i \overline{u}_j ) \, ,
```
is the mean rate of strain tensor.

Thanks to the non-linearity of the Navier-Stokes equations, even when averaged we are left with pesky fluctuation
terms which form the components of the *Reynolds stress tensor*
```math
\tau_{ij} = \rho \overline{v_i^\prime v_j^\prime} \, .
```
Attempting to close the equations leads to the *closure problem*: the time evolution of the Reynolds stresses
depends on  triple covariances ``\overline{v_i^\prime v_j^\prime v_k^\prime}`` and covariances with pressure, which depend
on quadruple covariances and so on [Chou45](@cite).

This is kind of hopeless so we will have to find some way to model the Reynolds stresses.

## Gradient-diffusion hypothesis and eddy viscosity models

The *gradient-diffusion hypothesis*, due to [Boussinesq1877](@cite), assumes that the transport of scalar fluxes
such as ``\overline{\boldsymbol{v}^\prime c^\prime}`` and ``\overline{v_i^\prime v_j^\prime}`` occurs down the mean scalar gradient
``\grad c`` as if they are being diffused (§4.4) [Pope00](@cite). This is in analogy with how momentum transfer by
molecular motion in a gas can be described by a molecular viscosity.

Taking this assumption we can express the Reynolds stresses and turbulent tracer fluxes in terms of the mean variables
and close the equations
```math
\overline{\boldsymbol{v}^\prime c^\prime} = -\kappa_e \boldsymbol{\nabla} \overline{c}
\quad \text{and} \quad
\overline{v_i^\prime v_j^\prime} = -2\nu_e \overline{S}_{ij} \, ,
```
where ``\nu_e = \nu_e(\boldsymbol{x}, t)`` is the turbulent or *eddy viscosity* and ``\kappa_e = \kappa_e(\boldsymbol{x}, t)``
is the *eddy diffusivity*.

The effective diffusivity ends up being the sum of the molecular and eddy diffusivities. So just by using an elevated
value for the viscosity and diffusivity, you are already using an eddy viscosity model.

The eddy viscosity model is simple and for that reason is very popular. It can work well even with a constant eddy
diffusivity. However, it does assume that the flux is aligned down gradient, which is not true even in simple turbulent
flows as the physics of turbulence is quite different from that of colliding molecules leading to the viscous stress law
(§4.4,10.1) [Pope00](@cite). So we might want something a little bit more sophisticated.
