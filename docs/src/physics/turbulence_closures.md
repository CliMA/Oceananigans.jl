# Turbulence closures

The turbulence closure selected by the user determines the form of stress divergence
``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau}`` and diffusive flux divergence
``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c`` in the momentum and tracer conservation equations.

## Constant isotropic diffusivity

In a constant isotropic diffusivity model, the kinematic stress tensor is defined
```math
\tau_{ij} = - \nu \Sigma_{ij} \, ,
```
where ``\nu`` is a constant viscosity and
``\Sigma_{ij} \equiv \tfrac{1}{2} \left ( v_{i, j} + v_{j, i} \right )`` is the strain-rate
tensor. The divergence of ``\boldsymbol{\tau}`` is then
```math
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau} = -\nu \nabla^2 \boldsymbol{v} \, .
```
Similarly, the diffusive tracer flux is ``\boldsymbol{q}_c = - \kappa \boldsymbol{\nabla} c`` for tracer
diffusivity ``\kappa``, and the diffusive tracer flux divergence is
```math
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c = - \kappa \nabla^2 c \, .
```
Each tracer may have a unique diffusivity ``\kappa``.

## Constant anisotropic diffusivity

In Oceananigans.jl, a constant anisotropic diffusivity implies a constant tensor
diffusivity ``\nu_{j k}`` and stress ``\boldsymbol{\tau}_{ij} = \nu_{j k} u_{i, k}`` with non-zero
components ``\nu_{11} = \nu_{22} = \nu_h`` and ``\nu_{33} = \nu_v``.
With this form the kinematic stress divergence becomes
```math
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau} = - \left [ \nu_h \left ( \partial_x^2 + \partial_y^2 \right )
                                    + \nu_v \partial_z^2 \right ] \boldsymbol{v} \, ,
```
and diffusive flux divergence
```math
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c = - \left [ \kappa_{h} \left ( \partial_x^2 + \partial_y^2 \right )
                                    + \kappa_{v} \partial_z^2 \right ] c \, ,
```
in terms of the horizontal viscosities and diffusivities ``\nu_h`` and ``\kappa_{h}`` and the
vertical viscosity and diffusivities ``\nu_v`` and ``\kappa_{v}``.
Each tracer may have a unique diffusivity components ``\kappa_h`` and ``\kappa_v``.

## Constant anisotropic biharmonic diffusivity

In Oceananigans.jl, a constant anisotropic biharmonic diffusivity implies a constant tensor
diffusivity ``\nu_{j k}`` and stress ``\boldsymbol{\tau}_{ij} = \nu_{j k} \partial_k^3 u_i`` with non-zero
components ``\nu_{11} = \nu_{22} = \nu_h`` and ``\nu_{33} = \nu_v``.
With this form the kinematic stress divergence becomes
```math
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau} = - \left [ \nu_h \left ( \partial_x^2 + \partial_y^2 \right )^2
                                    + \nu_v \partial_z^4 \right ] \boldsymbol{v} \, ,
```
and diffusive flux divergence
```math
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}_c = - \left [ \kappa_{h} \left ( \partial_x^2 + \partial_y^2 \right )^2
                                    + \kappa_{v} \partial_z^4 \right ] c \, ,
```
in terms of the horizontal biharmonic viscosities and diffusivities ``\nu_h`` and ``\kappa_{h}`` and the
vertical biharmonic viscosity and diffusivities ``\nu_v`` and ``\kappa_{v}``.
Each tracer may have a unique diffusivity components ``\kappa_h`` and ``\kappa_v``.

## Smagorinsky-Lilly turbulence closure

In the turbulence closure proposed by Lilly (1962) and [Smagorinsky63](@cite),
the subgrid stress associated with unresolved turbulent motions is modeled diffusively via
```math
\tau_{ij} = \nu_e \Sigma_{ij} \, ,
```
where ``\Sigma_{ij} = \tfrac{1}{2} \left ( v_{i, j} + v_{j, i} \right )`` is the resolved
strain rate.
The eddy viscosity is given by
```math
    \begin{align}
    \nu_e = \left ( C \Delta_f \right )^2 \sqrt{ \Sigma^2 } \, \varsigma(N^2 / \Sigma^2) + \nu \, ,
    \label{eq:smagorinsky-viscosity}
    \end{align}
```
where ``\Delta_f`` is the "filter width" associated with the finite volume grid spacing,
``C`` is a user-specified model constant, ``\Sigma^2 \equiv \Sigma_{ij} \Sigma_{ij}``, and
``\nu`` is a constant isotropic background viscosity.
The factor ``\varsigma(N^2 / \Sigma^2)`` reduces ``\nu_e`` in regions of
strong stratification via
```math
    \varsigma(N^2 / \Sigma^2) = \sqrt{1 - \min \left ( 1, C_b N^2 / \Sigma^2 \right )} \, ,
```
where ``N^2 = \max \left (0, \partial_z b \right )`` is the squared buoyancy frequency for stable
stratification with ``\partial_z b > 0`` and ``C_b`` is a user-specified constant.
Roughly speaking, the filter width for the Smagorinsky-Lilly closure is taken as
```math
\Delta_f(\boldsymbol{x}) = \left ( \Delta x \Delta y \Delta z \right)^{1/3} \, ,
```
where ``\Delta x``, ``\Delta y``, and ``\Delta z`` are the grid spacing in the
``\boldsymbol{\hat x}``, ``\boldsymbol{\hat y}``, and ``\boldsymbol{\hat z}`` directions at location ``\boldsymbol{x} = (x, y, z)``.

The effect of subgrid turbulence on tracer mixing is also modeled diffusively via
```math
\boldsymbol{q}_c = \kappa_e \boldsymbol{\nabla} c \, ,
```
where the eddy diffusivity ``\kappa_e`` is
```math
\kappa_e = \frac{\nu_e - \nu}{Pr} + \kappa \, ,
```
where ``Pr`` is a turbulent Prandtl number and ``\kappa`` is a constant isotropic background diffusivity.
Both ``Pr`` and ``\kappa`` may be set independently for each tracer.

## Anisotropic minimum dissipation (AMD) turbulence closure

Oceananigans.jl uses the anisotropic minimum dissipation (AMD) model proposed by
Verstappen18 and described and tested by Vreugdenhil18.
The AMD model uses an eddy diffusivity hypothesis similar the Smagorinsky-Lilly model.
In the AMD model, the eddy viscosity and diffusivity for each tracer are defined in terms
of eddy viscosity and diffusivity *predictors*
``\nu_e^\dagger`` and ``\kappa_e^\dagger``, such that
```math
    \nu_e = \max \left ( 0, \nu_e^\dagger \right ) + \nu
    \quad \text{and} \quad
    \kappa_e = \max \left ( 0, \kappa_e^\dagger \right ) + \kappa \, ,
```
to ensure that ``\nu_e \ge 0`` and ``\kappa_e \ge 0``, where ``\nu`` and ``\kappa`` are the
constant isotropic background viscosity and diffusivities for each tracer. The eddy viscosity 
predictor is
```math
    \begin{equation}
    \nu_e^\dagger = -C \Delta_f^2
    \frac
        {(\hat{\partial}_k \hat{v}_i) (\hat{\partial}_k \hat{v}_j) \hat{\Sigma}_{ij}
        + C_b \hat{\delta}_{i3} (\hat{\partial}_k \hat{v_i}) (\hat{\partial}_k b)}
        {(\hat{\partial}_l \hat{v}_m) (\hat{\partial}_l \hat{v}_m)} \, ,
    \label{eq:nu-dagger}
    \end{equation}
```
while the eddy diffusivity predictor for tracer ``c`` is
```math
    \begin{equation}
    \label{eq:kappa-dagger}
    \kappa_e^\dagger = -C \Delta_f^2
    \frac
        {(\hat{\partial}_k \hat{v}_i) (\hat{\partial}_k c) (\hat{\partial}_i c)}
        {(\hat{\partial}_l c) (\hat{\partial}_l c)} \, .
    \end{equation}
```
In the definitions of the eddy viscosity and eddy diffusivity predictor, ``C`` and ``C_b`` are
user-specified model constants, ``\Delta_f`` is a "filter width" associated with the finite volume
grid spacing, and the hat decorators on partial derivatives, velocities, and the Kronecker
delta ``\hat \delta_{i3}`` are defined such that
```math
    \hat \partial_i \equiv \Delta_i \partial_i, \qquad
    \hat{v}_i(x, t) \equiv \frac{v_i(x, t)}{\Delta_i}, \quad \text{and} \quad
    \hat{\delta}_{i3} \equiv \frac{\delta_{i3}}{\Delta_3} \, .
```
A velocity gradient, for example, is therefore
``\hat{\partial}_i \hat{v}_j(x, t) = \frac{\Delta_i}{\Delta_j} \partial_i v_j(x, t)``,
while the normalized strain tensor is
```math
    \hat{\Sigma}_{ij} =
        \frac{1}{2} \left[ \hat{\partial}_i \hat{v}_j(x, t) + \hat{\partial}_j \hat{v}_i(x, t) \right] \, .
```
The filter width ``\Delta_f`` in that appears in the viscosity and diffusivity predictors
is taken as the square root of the harmonic mean of the squares of the filter widths in
each direction:
```math
    \frac{1}{\Delta_f^2} = \frac{1}{3} \left(   \frac{1}{\Delta x^2}
                                              + \frac{1}{\Delta y^2}
                                              + \frac{1}{\Delta z^2} \right) \, .
```
The constant ``C_b`` permits the "buoyancy modification" term it multiplies to be omitted
from a calculation.
By default we use the model constants ``C=1/12`` and ``C_b=0``.
