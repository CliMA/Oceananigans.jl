# [Large eddy simulation](@id numerical_les)

The idea behind large eddy simulation (LES) is to resolve the "large eddies" while modeling the effect of unresolved
sub-grid scale motions. This is done usually be assuming eddy viscosity and eddy diffusivity models and providing an
estimate for the eddy viscosity ``\nu_e`` and diffusivity ``\kappa_e``.

Much of the early work on LES was motivated by the study of atmospheric boundary layer turbulence, being developed
by [Smagorinsky63](@cite) and [Lilly66](@cite), then first implemented by [Deardorff70](@cite) and [Deardorff74](@cite).

In the LES framework, the Navier-Stokes equations are averaged in the same way as [Reynolds1895](@cite) except that the
mean field ``\overline{\boldsymbol{u}}`` is obtained via convolution with a filter convolution kernel ``G``
```math
\overline{\boldsymbol{u}(\boldsymbol{x}, t)} = G \star \boldsymbol{u} =
  \int_{-\infty}^\infty \int_{-\infty}^\infty
  \boldsymbol{u}(\boldsymbol{x}^\prime, t) G(\boldsymbol{x} - \boldsymbol{x}^\prime, t - \tau) \, d\boldsymbol{x}^\prime \, \mathrm{d} \tau \, ,
```
as described by [Leonard75](@cite) who introduced the general filtering formalism.

The ``\overline{u_i^\prime u_j^\prime}`` terms are now components of what is called the sub-grid scale (SGS) stress
tensor ``\tau^\text{SGS}_{ij}``, which looks the same as the Reynolds stress tensor so we will drop the SGS superscript.

It is probably important to note that the large eddy simulation filtering operation does not satisfy the properties
of a Reynolds operator (§2.1)[sagaut06](@cite) and that in general, the filtered residual is not zero:
``\overline{\boldsymbol{u}^\prime(\boldsymbol{x}, t)} \ne 0``.

§13.2 of [Pope00](@cite) lists a number of popular choices for the filter function ``G``. For practical reasons we
simply employ the box kernel
```math
  \begin{equation}
  \label{eq:box-kernel}
  G_\Delta = G(\boldsymbol{x}, t) = \frac{1}{\Delta} H \left( \frac{1}{2}\Delta - |\boldsymbol{x}| \right) \delta(t - t_n) \, ,
  \end{equation}
```
where ``H`` is the Heaviside function, ``\Delta`` is the grid spacing, and ``t_n`` is the current time step. With
\eqref{eq:box-kernel} we get back the averaging operator originally used by [Deardorff70](@cite)
```math
\overline{\boldsymbol{u}(x, y, z, t)} =
  \frac{1}{\Delta x \Delta y \Delta z}
  \int_{x - \frac{1}{2}\Delta x}^{x + \frac{1}{2}\Delta x}
  \int_{y - \frac{1}{2}\Delta y}^{y + \frac{1}{2}\Delta y}
  \int_{z - \frac{1}{2}\Delta z}^{z + \frac{1}{2}\Delta z}
  \boldsymbol{u}(\xi, \eta, \zeta, t) \, \mathrm{d} \xi \, \mathrm{d} \eta \, \mathrm{d} \zeta \, ,
```
which if evaluated at the cell centers just returns the cell averages we already compute in the finite volume method.


## Smagorinsky-Lilly model

[Smagorinsky63](@cite) estimated the eddy viscosity ``\nu_e`` via a characteristic length scale ``\Delta`` times a velocity
scale given by ``\Delta |\overline{S}|`` where ``|\overline{S}| = \sqrt{2\overline{S}_{ij}\overline{S}_{ij}}``. Thus the
SGS stress tensor is given by
```math
\tau_{ij} = -2\nu_e \overline{S}_{ij} = -2 (C_s \Delta)^2 |\overline{S}| \overline{S}_{ij} \, ,
```
where ``C_s`` is a dimensionless constant. The grid spacing is usually used for the characteristic length scale ``\Delta``.
The eddy diffusivities are calculated via ``\kappa_e = \nu_e / \text{Pr}_t`` where the turbulent Prandtl number
``\text{Pr}_t`` is usually chosen to be ``\mathcal{O}(1)`` from experimental observations.

Assuming that the SGS energy cascade is equal to the overall dissipation rate ``\varepsilon`` from the
[Kolmogorov41](@cite) theory, [Lilly66](@cite) was able to derive a value of
```math
C_s = \left( \frac{3}{2}C_K\pi^\frac{4}{3} \right)^{-\frac{3}{4}} \approx 0.16 \, ,
```
using an empirical value of ``C_K \approx 1.6`` for the Kolmogorov constant. This seems reasonable for isotropic
turbulence if the grid spacing ``\Delta`` falls in the inertial range. In practice, ``C_s`` is a tunable parameter.

Due to the presence of the constant ``C_s``, the model is sometimes referred to as the *constant Smagorinsky* model
in contrast to *dynamic Smagorinsky* models that dynamically compute ``C_s`` to account for effects such as buoyant
convection.

## Anisotropic minimum dissipation models

Minimum-dissipation eddy-viscosity models are a class of LES closures that use the minimum eddy dissipation required to
dissipate the energy of sub-grid scale motion. [Rozema15](@cite) proposed the first minimum-dissipation model
appropriate for use on anisotropic grids, termed the *anisotropic minimum dissipation* (AMD) model.

It has a number of desirable properties over Smagorinsky-type closures: it is more cost-effective than dynamic
Smagorinsky, it appropriately switches off in laminar and transitional flows, and it is consistent with the exact SGS
stress tensor on both isotropic and anisotropic grids. [Abkar16](@cite) extended the AMD model to model SGS scalar
fluxes for tracer transport. [Abkar17](@cite) further extended the model to include a buoyancy term that accounts for
the contribution of buoyant forces to the production and suppression of turbulence.

[Vreugdenhil18](@cite) derive a modified AMD model by following the requirement suggested by [Verstappen18](@cite),
which entail normalising the displacement, the velocity, and the velocity gradient by the filter width to ensure that
the resulting eddy dissipation properly counteracts the spurious kinetic energy transferred by convective nonlinearity,
to derive a modified AMD model.

The eddy viscosity and diffusivity are defined in terms of eddy viscosity and diffusivity *predictors*
``\nu_e^\dagger`` and ``\kappa_e^\dagger``, such that
```math
\nu_e = \max \lbrace 0, \nu_e^\dagger \rbrace
\quad \text{and} \quad
\kappa_e = \max \lbrace 0, \kappa_e^\dagger \rbrace \, ,
```
to ensure that ``\nu_e \ge 0`` and ``\kappa_e \ge 0``. Leaving out the overlines and understanding that all variables
represent the resolved/filtered variables, the eddy viscosity predictor is given by
```math
    \begin{equation}
    \label{eq:nu-dagger}
    \nu_e^\dagger = -(C\Delta)^2
      \frac
        {\left( \hat{\partial}_k \hat{u}_i \right) \left( \hat{\partial}_k \hat{u}_j \right) \hat{S}_{ij}
        + C_b\hat{\delta}_{i3} \alpha g \left( \hat{\partial}_k \hat{u_i} \right) \hat{\partial}_k \theta}
        {\left( \hat{\partial}_l \hat{u}_m \right) \left( \hat{\partial}_l \hat{u}_m \right)} \, ,
    \end{equation}
```
and the eddy diffusivity predictor by
```math
    \begin{equation}
    \kappa_e^\dagger = -(C\Delta)^2
    \frac
        {\left( \hat{\partial}_k \hat{u}_i \right) \left( \hat{\partial}_k \hat{\theta} \right) \hat{\partial}_i \theta}
        {\left( \hat{\partial}_l \hat{\theta} \right) \left( \hat{\partial}_l \hat{\theta} \right)} \, ,
    \end{equation}
```
where
```math
  \begin{equation}
  \hat{x}_i = \frac{x_i}{\Delta_i}, \quad
  \hat{u}_i(\hat{x}, t) = \frac{u_i(x, t)}{\Delta_i}, \quad
  \hat{\partial}_i \hat{u}_j(\hat{x}, t) = \frac{\Delta_i}{\Delta_j} \partial_i u_j(x, t), \quad
  \hat{\delta}_{i3} = \frac{\delta_{i3}}{\Delta_3} \, ,
  \end{equation}
```
so that the normalized rate of strain tensor is
```math
    \begin{equation}
    \label{eq:S-hat}
    \hat{S}_{ij} =
      \frac{1}{2} \left[ \hat{\partial}_i \hat{u}_j(\hat{x}, t) + \hat{\partial}_j \hat{u}_i(\hat{x}, t) \right] \, .
    \end{equation}
```

In equations \eqref{eq:nu-dagger}--\eqref{eq:S-hat}, ``C`` is a modified Poincaré "constant" that is independent from
the filter width ``\Delta`` but does depend on the accuracy of the discretization method used. [Abkar16](@cite) cite
``C^2 = \frac{1}{12}`` for a spectral method and ``C^2 = \frac{1}{3}`` for a second-order accurate scheme. ``\Delta_i`` is
the filter width in the ``x_i``-direction, and ``\Delta`` is given by the square root of the harmonic mean of the squares
of the filter widths in each direction
```math
    \frac{1}{\Delta^2} = \frac{1}{3} \left( \frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2} \right) \, .
```
The term multiplying ``C_b`` is the buoyancy modification introduced by [Abkar17](@cite) and is small for weakly
stratified flows. We have introduced the ``C_b`` constant so that the buoyancy modification term may be turned on and off.
