# EquatorialLatitudeLongitudeGrid (Work in Progress)

## Motivation

The standard `LatitudeLongitudeGrid` contains coordinate singularities at the
geographic North and South poles.

While these singularities are acceptable for many applications, they complicate
regional simulations in the Arctic Ocean because grid metrics become highly
anisotropic near the pole.

The goal of `EquatorialLatitudeLongitudeGrid` is to relocate the coordinate
singularity away from the polar regions while preserving an orthogonal spherical
coordinate system and maintaining compatibility with Oceananigans'
finite-volume operators.

This work is currently focused on geometry construction and tracer-advection
validation.

## Current Status

### Completed

- [x] Construct `EquatorialLatitudeLongitudeGrid`
- [x] Verify grid metrics and geometry
- [x] Validate passive tracer initialization
- [x] Validate uniform x-advection
- [x] Validate uniform y-advection
- [x] Validate combined x-y advection
- [x] Perform resolution studies (64², 128², 256²)
- [x] Compare tracer centroid trajectories
- [x] Compare normalized displacements

## Coordinate Transformations

### LatitudeLongitudeGrid

Let

- $\lambda$ denote longitude,
- $\phi$ denote latitude,
- $R$ denote the planetary radius.

The mapping from spherical coordinates to Cartesian coordinates is

$$
x = R \cos\phi \cos\lambda,
$$

$$
y = R \cos\phi \sin\lambda,
$$

$$
z = R \sin\phi.
$$

The resulting metric is

$$
ds^2
=
R^2 \cos^2 \phi \, d\lambda^2
+
R^2 d\phi^2.
$$

The corresponding physical velocity components are

$$
u = R \cos\phi \frac{d\lambda}{dt},
$$

$$
v = R \frac{d\phi}{dt}.
$$

For constant angular motion,

$$
\frac{d\lambda}{dt}
=
\Omega,
$$

the zonal velocity must therefore satisfy

$$
u = R \cos\phi \, \Omega.
$$

This metric correction was used in the validation experiments described below.

### EquatorialLatitudeLongitudeGrid

The EquatorialLatitudeLongitudeGrid is obtained by rotating the pole of the
latitude-longitude coordinate system onto the geographic equator.

The coordinate transformation is

$$
x = R \sin\phi_e,
$$

$$
y = R \cos\phi_e \cos\lambda_e,
$$

$$
z = R \cos\phi_e \sin\lambda_e.
$$

The resulting metric is

$$
ds^2
=
R^2 d\phi_e^2
+
R^2 \cos^2\phi_e\, d\lambda_e^2.
$$

The corresponding physical velocity components satisfy

$$
u_e = R \frac{d\phi_e}{dt},
$$

$$
v_e = R \cos\phi_e \frac{d\lambda_e}{dt}.
$$

### Key Findings

- ELLC reproduces tracer transport in both coordinate directions.
- Normalized tracer displacement is independent of grid resolution.
- ELLC exhibits minimal tracer deformation in the current test suite.
- ELLC permits nearly uniform transport using spatially constant velocities.
- The current LLC comparison requires a latitude-dependent zonal velocity correction,
  u = U₀ cos(φ), to achieve uniform angular transport.

## Validation

Validation currently focuses on passive tracer transport.

A Gaussian tracer was initialized at the center of the computational domain and
advected using prescribed background velocities.

Three configurations were examined:

### Uniform x-advection

A velocity field corresponding to uniform angular transport in the x-direction.

Result:
- matching centroid trajectories;
- matching normalized displacement;
- comparable tracer evolution.

### Uniform y-advection

A velocity field corresponding to transport in the y-direction.

Result:
- comparable centroid trajectories;
- empirical ELLC velocity scaling currently required.

### Combined x-y advection

Simultaneous transport in both coordinate directions.

Result:
- matching x-displacement;
- similar y-displacement;
- successful proof-of-concept demonstration of transport on ELLC.

### Key Findings

- ELLC reproduces tracer transport in both coordinate directions.
- Normalized tracer displacement is independent of grid resolution.
- ELLC exhibits minimal tracer deformation in the current test suite.
- ELLC permits nearly uniform transport using spatially constant velocities.
- The current LLC comparison requires a latitude-dependent zonal velocity correction,
  u = U₀ cos(φ), to achieve uniform angular transport.

## Resolution Study

The following resolution study demonstrates that the tracer trajectories
and normalized displacement are essentially independent of grid resolution.

### Peak tracer magnitude

| Grid | N | Peak |
|------|---:|------:|
| LLC  | 64  | 1.164 |
| LLC  | 128 | 1.168 |
| LLC  | 256 | 1.169 |
| ELLC | 64  | 0.997 |
| ELLC | 128 | 1.000 |
| ELLC | 256 | 1.000 |

### Normalized displacement

### Normalized displacement

| Grid | N | Δi/N | Δj/N |
|------|---:|------:|------:|
| LLC  | 64  | 0.1295 | 0.1307 |
| LLC  | 128 | 0.1295 | 0.1307 |
| LLC  | 256 | 0.1295 | 0.1307 |
| ELLC | 64  | 0.1295 | 0.1228 |
| ELLC | 128 | 0.1295 | 0.1228 |
| ELLC | 256 | 0.1295 | 0.1228 |

## Future Work

### Geometry

- [ ] Verify ELLC coordinate transformation against implementation
- [ ] Derive inverse transformation

### Physics

- [ ] Coriolis operator
- [ ] Momentum equations
- [ ] Pressure-gradient terms
- [ ] Geostrophic balance tests
- [ ] Barotropic test cases

### Validation

- [ ] GPU validation
- [ ] Solid-body rotation
- [ ] Passive tracer vortex tests
- [ ] Long-time integrations
- [ ] Convergence studies

### Applications

- [ ] Arctic regional domains
- [ ] Polar cap simulations
- [ ] Comparison with LatitudeLongitudeGrid
