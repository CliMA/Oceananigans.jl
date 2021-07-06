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

### The non-traditional ``f``-plane approximation

In the *non-traditional* ``f``-plane approximation, the coordinate system rotates around
an axis in the ``y,z``-plane, such that
```math
    \boldsymbol{f} = f_y \boldsymbol{\hat y} + f_z \boldsymbol{\hat z} \, ,
```
where ``f_y`` and ``f_z`` are constants determined by the user.


[^3]: The ``f``-plane approximation is used to model the effects of Earth's rotation on anisotropic 
      fluid motion in a plane tangent to the Earth's surface. In this case, the projection of 
      the Earth's rotation vector at latitude ``\varphi`` and onto a coordinate system in which 
      ``x, y, z`` correspond to the directions east, north, and up is
      ```math
          \boldsymbol{f} \approx \frac{4 \pi}{\text{day}} \left ( \cos \varphi \boldsymbol{\hat y} + \sin \varphi \boldsymbol{\hat z} \right ) \, ,
      ```
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
as can be found in the paper by [DellarJFM2011](@cite) where 
``\beta = 2 \Omega \cos \varphi_0 / R`` and ``\gamma = -4 \Omega \sin \varphi_0 / R``.
