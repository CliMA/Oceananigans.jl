# Coriolis forces

The Coriolis model controls the manifestation of the term $\bm{f} \times \bm{u}$ in the momentum equation.

## The traditional and non-traditional $f$-plane approximation

Under an $f$-plane approximation[^3] the reference frame in which
the momentum and tracer equations are solved rotates at a constant rate. 

### The traditional $f$-plane approximation 

In the *traditional* $f$-plane approximation, the coordinate system rotates around
a vertical axis such that
```math
    \bm{f} = f \bm{\hat z}
```
where $f$ is constant and determined by the user.

### The non-traditional $f$-plane approximation 

In the *non-traditional* $f$-plane approximation, the coordinate system rotates around
an axis in the $y,z$-plane, such that
```math
    \bm{f} = f_y \bm{\hat y} + f_z \bm{\hat z}
```
where $f_y$ and $f_z$ are constant and determined by the user.


[^3]: The $f$-plane approximation is used to model the effects of Earth's rotation on
      anisotropic fluid motion in a plane tangent to the Earth's surface. In this case, the projection of Earth's
      rotation vector at latitude $\varphi$ and onto a coordinate system in which $x, y, z$ correspond to the 
      directions east, north, and up is
      ```math
          \bm{f} \approx \frac{4 \pi}{\text{day}} \left ( \cos \varphi \bm{\hat y} + \sin \varphi \bm{\hat z} \right ) \, , $
      ```
      where the Earth's rotation rate is approximately $2 \pi / \text{day}$.
      The *traditional* $f$-plane approximation neglects the $y$-component of this projection, which is appropriate for 
      fluid motions with large horizontal-to-vertical aspect ratios.

## The $\beta$-plane approximation

Under the $\beta$-plane approximation, the rotation axis is vertical as for the
$f$-plane approximation, but $f$ is expanded in a Taylor series around a central latitude such that
```math
    \bm{f} = \left ( f_0 + \beta y \right ) \bm{\hat z} \, ,
```
where $f_0$ is the planetary vorticity at some central latitude, and $\beta$ is the
planetary vorticity gradient.
The $\beta$-plane model is not periodic in $y$ and thus can be used only in domains that
are bounded in the $y$-direction.
