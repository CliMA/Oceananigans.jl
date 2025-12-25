# Coordinate systems

Every grid in Oceananigans is associated with an _extrinsic_ coordinate system.
The extrinsic coordinate system is either _Cartesian_ or _geographic_
(spherical / latitude-longitude).

## Cartesian coordinates

Cartesian coordinates are used by [`RectilinearGrid`](@ref) and provide a local, flat representation
of space with coordinates ``\boldsymbol{x} = (x, y, z)`` and unit vectors
``\boldsymbol{\hat x}``, ``\boldsymbol{\hat y}``, and ``\boldsymbol{\hat z}``.
By convention, ``\boldsymbol{\hat x}`` points east, ``\boldsymbol{\hat y}`` points north,
and ``\boldsymbol{\hat z}`` points "upward", opposite to the direction of gravitational acceleration.

We denote time with ``t``, partial derivatives with respect to time ``t`` or a coordinate ``x``
with ``\partial_t`` or ``\partial_x``, and the gradient operator
```math
\boldsymbol{\nabla} \equiv \partial_x \boldsymbol{\hat x} + \partial_y \boldsymbol{\hat y} + \partial_z \boldsymbol{\hat z} \, .
```
Horizontal gradients are
```math
\boldsymbol{\nabla}_h \equiv \partial_x \boldsymbol{\hat x} + \partial_y \boldsymbol{\hat y} \, .
```

We use ``u``, ``v``, and ``w`` to denote the velocity components in the ``x``, ``y``, and ``z`` directions,
such that the three-dimensional velocity is
``\boldsymbol{v} = u \boldsymbol{\hat x} + v \boldsymbol{\hat y} + w \boldsymbol{\hat z}``.
We use ``\boldsymbol{u} = u \boldsymbol{\hat x} + v \boldsymbol{\hat y}`` to denote the horizontal velocity.

## Geographic coordinates

Geographic (or spherical) coordinates ``(\lambda, \phi, z)`` represent longitude, latitude, and
the vertical coordinate and are used by [`LatitudeLongitudeGrid`](@ref) and [`OrthogonalSphericalShellGrid`](@ref).
The corresponding unit vectors are ``\boldsymbol{\hat \lambda}`` (eastward),
``\boldsymbol{\hat \phi}`` (northward), and ``\boldsymbol{\hat z}`` (upward, radially outward
from the center of the sphere).

The velocity components in geographic coordinates are ``(u, v, w)``
where ``u`` is the eastward velocity, ``v`` is the northward velocity, and ``w`` is the vertical velocity:
```math
\boldsymbol{v} = u \boldsymbol{\hat \lambda} + v \boldsymbol{\hat \phi} + w \boldsymbol{\hat z} \, .
```

## Intrinsic coordinates on `OrthogonalSphericalShellGrid`

While [`RectilinearGrid`](@ref) and [`LatitudeLongitudeGrid`](@ref) have extrinsic and intrinsic coordinate systems
that coincide, the [`OrthogonalSphericalShellGrid`](@ref) also possesses an _intrinsic_ coordinate system
that is associated with the local grid directions.

The intrinsic coordinate system on `OrthogonalSphericalShellGrid` is defined by the orientation of the grid
lines at each point. This intrinsic system may be rotated relative to the extrinsic geographic
coordinates (latitude and longitude). Vectors such as C-grid velocity components ``(u, v)``,
momentum fluxes, and other vector quantities are represented in the intrinsic coordinate system,
aligned with the local grid directions.

The operators [`intrinsic_vector`](@ref) and [`extrinsic_vector`](@ref)
convert vectors between these two coordinate systems:

- [`intrinsic_vector`](@ref): Converts from extrinsic (geographic) to intrinsic (grid-aligned) coordinates.
- [`extrinsic_vector`](@ref): Converts from intrinsic (grid-aligned) to extrinsic (geographic) coordinates.

For example, to set velocities on an `OrthogonalSphericalShellGrid` from geographic velocity data,
the velocities must first be rotated from geographic to intrinsic coordinates.

