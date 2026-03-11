# Coriolis forces

The Coriolis model controls the manifestation of the term ``\boldsymbol{f} \times \boldsymbol{v}``
in the momentum equation.

## ``f``-plane approximation

Under an ``f``-plane approximation[^3] the reference frame in which
the momentum and tracer equations are solved rotates at a constant rate.

### The traditional ``f``-plane approximation

In the *traditional* ``f``-plane approximation, the coordinate system rotates around
a vertical axis such that
```math
    \boldsymbol{f} = f \boldsymbol{\hat z} \, ,
```
where ``f`` is constant and determined by the user.

## The arbitrary-axis constant-Coriolis approximation

In this approximation, the coordinate system rotates around an axis in the ``x,y,z``-plane, such
that
```math
    \boldsymbol{f} = f_x \boldsymbol{\hat x} + f_y \boldsymbol{\hat y} + f_z \boldsymbol{\hat z} \, ,
```
where ``f_x``, ``f_y``, and ``f_z`` are constants determined by the user.

[^3]: The ``f``-plane approximation is used to model the effects of Earth's rotation on anisotropic
      fluid motion in a plane tangent to the Earth's surface. In this case, the projection of
      the Earth's rotation vector at latitude ``\varphi`` and onto a coordinate system in which
      ``x, y, z`` correspond to the directions east, north, and up is
      ``\boldsymbol{f} \approx \frac{4 \pi}{\text{day}} \left ( \cos \varphi \boldsymbol{\hat y} + \sin \varphi \boldsymbol{\hat z} \right ) \, ,``
      where the Earth's rotation rate is approximately ``2 \pi / \text{day}``. The *traditional*
      ``f``-plane approximation neglects the ``y``-component of this projection, which is appropriate
      for fluid motions with large horizontal-to-vertical aspect ratios.

## ``\beta``-plane approximation

### The traditional ``\beta``-plane approximation

Under the *traditional* ``\beta``-plane approximation, the rotation axis is vertical as for the
``f``-plane approximation, but ``f`` is expanded in a Taylor series around a central latitude
such that
```math
    \boldsymbol{f} = \left ( f_0 + \beta y \right ) \boldsymbol{\hat z} \, ,
```
where ``f_0`` is the planetary vorticity at some central latitude, and ``\beta`` is the
planetary vorticity gradient.
The ``\beta``-plane model is not periodic in ``y`` and thus can be used only in domains that
are bounded in the ``y``-direction.

### The non-traditional ``\beta``-plane approximation

The *non-traditional* ``\beta``-plane approximation accounts for the latitudinal variation of both
the locally vertical and the locally horizontal components of the rotation vector
```math
    \boldsymbol{f} = \left[ 2\Omega\cos\varphi_0 \left( 1 -  \frac{z}{R} \right) + \gamma y \right] \boldsymbol{\hat y}
           + \left[ 2\Omega\sin\varphi_0 \left( 1 + 2\frac{z}{R} \right) + \beta  y \right] \boldsymbol{\hat z} \, ,
```
as can be found in the paper by [Dellar2011](@citet), where
``\beta = 2 \Omega \cos \varphi_0 / R`` and ``\gamma = -4 \Omega \sin \varphi_0 / R``.

## Spherical Coriolis

On latitude-longitude grids, the Coriolis parameter varies with latitude according to
```math
    f(\varphi) = 2 \Omega \sin \varphi \, ,
```
where ``\Omega`` is the planetary rotation rate and ``\varphi`` is latitude.

For hydrostatic models, only the vertical component of the Coriolis force is retained
(the *traditional approximation*), contributing ``-fv`` and ``+fu`` to the zonal and
meridional momentum equations respectively.

For nonhydrostatic models, the full Coriolis force includes additional terms involving
the horizontal component ``\tilde{f} = 2\Omega \cos \varphi``, which couples the horizontal
and vertical momentum equations.

## Discretization of the Coriolis term

On the Arakawa C-grid, the two velocity components ``u`` and ``v`` are staggered: ``u`` is
defined at the west and east faces of each cell, while ``v`` is defined at the south and north
faces. Computing the Coriolis acceleration (e.g., ``-fv`` in the ``u``-equation) therefore requires
**interpolating** ``v`` to the ``u``-point, and vice versa.

The choice of interpolation scheme affects two important properties:

1. **Conservation**: whether the scheme conserves kinetic energy, potential enstrophy, or both.
2. **Boundary accuracy**: whether the scheme correctly handles masked (land) points near
   immersed boundaries.

### Enstrophy-conserving scheme

The enstrophy-conserving scheme [Sadourny1975](@citep) evaluates ``f`` at cell centers and
interpolates **velocity** directly:
```math
    \left( \boldsymbol{f} \times \boldsymbol{v} \right)_x
    \approx -\overline{f}^x \; \overline{v}^{xy} \, ,
```
where ``\overline{f}^x`` denotes the interpolation of ``f`` from cell centers to the ``u``-point,
and ``\overline{v}^{xy}`` is the 4-point average of ``v`` to the ``u``-point.

This scheme conserves **potential enstrophy** (``\tfrac{1}{2} q^2`` where ``q = (\zeta + f)/h``)
for horizontally non-divergent flow, but does not conserve kinetic energy.

### Energy-conserving scheme

The energy-conserving scheme [Sadourny1975](@citep) evaluates ``f`` at vorticity (corner)
points and interpolates **transport** (``V = v \Delta x``) rather than velocity:
```math
    \left( \boldsymbol{f} \times \boldsymbol{v} \right)_x
    \approx -\frac{1}{\Delta x} \overline{f \, \overline{V}^x}^y \, ,
```
where ``V = v \Delta x`` is the volume transport per unit depth. The product ``f \cdot V`` is computed
at each vorticity point **before** the spatial averaging, which ensures that the Coriolis
terms cancel when forming the kinetic energy equation [Dobricic2006](@citep).

This scheme conserves **kinetic energy** but not potential enstrophy.

### EEN (Energy- and Enstrophy-Conserving) scheme

The EEN scheme [ArakawaLamb1981](@citep) uses **triads** to achieve simultaneous conservation
of both kinetic energy and potential enstrophy. Each triad at a cell center sums 3 of the 4
surrounding vorticity values, paired with transports at diagonally adjacent velocity points.
The four triads at cell center ``(i,j)`` are:
```math
    \mathcal{T}^{++}_{i,j} = q_{i,j+1} + q_{i+1,j+1} + q_{i+1,j} \, , \\
    \mathcal{T}^{-+}_{i,j} = q_{i,j} + q_{i,j+1} + q_{i+1,j+1} \, , \\
    \mathcal{T}^{+-}_{i,j} = q_{i+1,j+1} + q_{i+1,j} + q_{i,j} \, , \\
    \mathcal{T}^{--}_{i,j} = q_{i+1,j} + q_{i,j} + q_{i,j+1} \, ,
```
where ``q`` is the potential vorticity at corner (vorticity) points. The Coriolis tendency
is then:
```math
    \left( \boldsymbol{f} \times \boldsymbol{v} \right)_x
    \approx -\frac{1}{12 \Delta x} \sum_{\sigma} \mathcal{T}^{\sigma}_{i,j} \; V^{\sigma}_{i,j} \, ,
```
where the sum is over the four triads and ``V^{\sigma}`` is the transport at the diagonally
paired velocity point. This 12-point stencil conserves both kinetic energy and potential
enstrophy in the limit of horizontally non-divergent flow.

### Active-weighted (wet-points-only) correction

Near immersed boundaries on a C-grid, the conventional averaging of velocities in the
Coriolis term includes masked (land) points where velocity is zero. As shown by
[JamartOzer1986](@citet), this underestimates the Coriolis force along solid boundaries:

> "The calculation of the ``fv`` term in the ``x`` momentum equation is usually performed by
> averaging the ``v`` values of the four closest neighbors of the ``u`` point under
> consideration. [...] In cases where the interior solution is uniform, the procedure amounts
> to reducing the Coriolis parameter ``f`` by a factor of 2 along such a wall."

This creates a **spurious numerical boundary layer** with artificial residual currents
that are entirely an artifact of the discretization.

The wet-points-only correction eliminates this artifact by dividing the interpolated
Coriolis term by the number of **active** (non-masked) nodes in the stencil, rather
than the full stencil size:
```math
    \left( \boldsymbol{f} \times \boldsymbol{v} \right)_x^{\text{corrected}}
    = \frac{\left( \boldsymbol{f} \times \boldsymbol{v} \right)_x}
           {N_{\text{active}}} \, ,
```
where ``N_{\text{active}}`` is the count of non-peripheral (wet) velocity nodes in the
4-point interpolation stencil. When all nodes are active, this reduces to the standard scheme.

!!! info "When to use the wet-points-only correction"
    [JamartOzer1986](@citet) found that the wet-points-only method is essential for 3D models
    with immersed boundaries, where it eliminates spurious boundary layers. However, the
    correction is generally not necessary for vertically integrated (2D) models, where the
    boundary condition of zero normal transport is sufficient.
