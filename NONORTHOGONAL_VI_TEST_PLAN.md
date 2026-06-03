# Must-Pass Test Plan for Non-Orthogonal C-grid Vector-Invariant Momentum Advection

**Project:** `SphericalShellGrid` + non-orthogonal C-grid `VectorInvariant` / `WENOVectorInvariant`  
**Primary current grid target:** `SphericalShellGrid(; mapping = OctaHEALPixMapping(N))`  
**Primary current model target:** `HydrostaticFreeSurfaceModel` with `free_surface = nothing`, followed by Breeze compressible and then HFSM free-surface configurations  
**Status baseline:** current evidence shows centered `VectorInvariant()` passes the OHPSG random-vortex gate; full true WENO VI does **not** yet pass. The current passing OHPSG `WENOVectorInvariant` is a fallback hybrid: centered rotational advection + centered KE/Bernoulli head + WENO divergence/self-upwind.

This document lists the tests that **must pass** before we can say the non-orthogonal vector-invariant implementation is correct. It is deliberately strict. A successful short turbulence run is not enough; correctness requires metric compatibility, topology consistency, stable seams, operator adjointness/energy structure, and isolated WENO component stability.

---

## 1. Definitions and non-negotiable conventions

### 1.1 Velocity and transport objects

The implementation must maintain a clear distinction between model velocities and transport velocities.

- `model.velocities`: velocity fields used by momentum tendencies and diagnostics.
- `model.transport_velocities` or `transport_velocities(model)`: velocity fields used for tracer/scalar advection.
- `volume_transport`: integrated volume transport, mathematically
  \[
  \mathcal V^i = J u^i .
  \]
- `mass_transport`: integrated mass transport, mathematically
  \[
  \mathcal U^i = J \rho u^i .
  \]

For simple orthogonal or fixed-grid cases, `transport_velocities === velocities` may be valid. For non-orthogonal grids it must be an explicit, tested diagnostic, not an accidental alias.

### 1.2 Oceananigans staggered locations

Use Oceananigans-style C-grid notation in tests and documentation:

| Location | Meaning | Typical object |
|---|---|---|
| `ccc` | cell center | scalars, density, kinetic energy |
| `fcc` | first-face / `u` location | `u`, `volume_transport.u`, `mass_transport.u` |
| `cfc` | second-face / `v` location | `v`, `volume_transport.v`, `mass_transport.v` |
| `ccf` | third-face / `w` location | `w`, vertical transport |
| `ffc` | horizontal edge | vertical relative vorticity / `ζ12` |
| `fcf` | first-vertical edge | `ζ13` |
| `cff` | second-vertical edge | `ζ23` |

### 1.3 Relative versus absolute vorticity

WENO must reconstruct **relative vorticity**, not total absolute vorticity:

\[
\zeta_{ij} = D_i u_j - D_j u_i,
\qquad
Z_{ij} = \zeta_{ij} + f_{ij}.
\]

The reconstructed absolute-vorticity factor should be assembled as

\[
\widehat Z_{ij}
=
\mathcal R_{\rm WENO}[\zeta_{ij}]
+
\mathcal R_{\rm smooth}[f_{ij}].
\]

A test must explicitly prevent silent WENO reconstruction of `ζ + f` when the intended behavior is WENO on `ζ` plus smooth reconstruction of `f`.

### 1.4 No hidden fallback in final true-WENO tests

It is acceptable to keep a temporary fallback hybrid for stability while development continues. However, a final true `WENOVectorInvariant` test **must fail** if the implementation silently falls back to centered rotational or centered KE-gradient terms. The fallback and true-WENO paths must have separate tests.

---

## 2. Test tiers

Use three tiers.

### Tier 0: unit and algebraic tests

Fast, deterministic, no full model time stepping. These should run in normal CI.

### Tier 1: short model gates

Short deterministic simulations that run fast enough for CI or a required nightly job.

### Tier 2: extended confidence tests

Longer or more expensive tests: multiple seeds, resolutions, hardware backends, turbulence spectra, convergence, and free-surface configurations. These should run nightly or before major merges.

---

## 3. Grid and metric tests

These tests must pass before trusting any advection result.

### 3.1 `SphericalShellGrid` construction smoke test

**Purpose:** Ensure the grid materializes without invalid metric values.

**Configuration:**

```julia
SphericalShellGrid(
    CPU(), Float64;
    mapping = OctaHEALPixMapping(32),
    z = (0, 1),
    radius = 1,
    halo = (5, 5, 3),
)
```

**Must check:**

- all metric fields finite in interior and halo regions,
- no `NaN` or `Inf` in node coordinates,
- no negative cell areas or volumes,
- expected topology/halo sizes,
- all required staggered locations exist: `ccc`, `fcc`, `cfc`, `ccf`, `ffc`, `fcf`, `cff`.

**Acceptance:** zero invalid values.

---

### 3.2 Cross-metric averaging / duality test

**Purpose:** Verify the current cross-metric averaging patch is enforced as a formal test, not just a one-off probe.

**Current known probe result:**

```text
metric_cross_probe max_x=0.0 max_y=0.0
```

**Must check:**

```text
G12_fca == avg_i(G12_cca)
G21_cfa == avg_j(G12_cca)
```

where notation should map to code fields with clear names such as:

```julia
contravariant_metric_cross_12_fca
contravariant_metric_cross_21_cfa
contravariant_metric_cross_12_cca
```

**Acceptance:** exact equality if construction is arithmetic by assignment, otherwise roundoff-level error.

**Failure implication:** cross-metric placement is not metric-dual; centered VI may be unstable or noisy.

---

### 3.3 Metric inverse consistency

**Purpose:** Verify covariant and contravariant metric tensors are internally consistent at every stored location.

For each relevant staggered location, check

\[
g_{ik} g^{kj} \approx \delta_i^{\ j}.
\]

**Required locations:** at least `ccc`, `fcc`, `cfc`; later `ccf`, `ffc`, `fcf`, `cff` when full 3D VI is enabled.

**Acceptance:** relative error near roundoff for analytic/generated paired metrics; convergent with resolution for metrics derived independently.

---

### 3.4 Positive metric determinant

**Purpose:** Prevent degenerate or inverted metric cells.

For each horizontal location:

\[
g_{11} g_{22} - g_{12}^2 > 0.
\]

For full 3D/deep grids, also check the full determinant is positive.

**Acceptance:** strictly positive with a safety margin; no near-zero determinant except masked/invalid boundary cells explicitly excluded.

---

### 3.5 Area and volume closure

**Purpose:** Ensure integrated geometry is correct.

For a one-panel grid on a unit sphere:

\[
\sum_{\text{panel cells}} A \approx \frac{4\pi}{6}.
\]

For a six-panel grid:

\[
\sum_{\text{all cells}} A \approx 4\pi.
\]

For a thin shell with radial thickness `Δr`, volume should match area times thickness to expected order.

**Acceptance:** convergence with resolution, and roundoff where exact formulas/consistent quadrature are used.

---

## 4. Hodge-map and staggered metric tests

These tests decide whether the chosen stencil for converting covariant velocities to transports is viable.

The implementation should support multiple candidate Hodge maps during development:

```julia
TargetMetricHodge()
ProductInterpolatedHodge()
EnergySymmetricHodge()
```

The final code may keep only the accepted default plus debugging variants, but the test campaign should compare them.

---

### 4.1 Hodge consistency for known vector fields

**Purpose:** Verify

\[
\mathcal V^i = \mathsf H^i{}_j u_j
\]

recovers the analytic volume transport.

**Test fields:**

1. constant Cartesian velocity,
2. solid-body rotation,
3. smooth manufactured tangent vector field,
4. optional field with vertical component for 3D tests.

**Procedure:**

- compute analytic covariant velocities `u_i` at staggered locations,
- apply candidate Hodge map,
- compare with analytic `volume_transport` at `fcc/cfc/ccf`.

**Acceptance:** expected convergence order under grid refinement.

---

### 4.2 Weighted adjointness of off-diagonal Hodge blocks

**Purpose:** Prevent asymmetric off-diagonal metric coupling from injecting energy or producing computational modes.

For example, the cross blocks should satisfy

\[
\langle u_1, \mathsf H^{12} u_2 \rangle_{fcc}
=
\langle \mathsf H^{21} u_1, u_2 \rangle_{cfc}.
\]

Compute defect:

\[
\epsilon_{\rm adj}
=
\frac{\| W_1 \mathsf H^{12} - (W_2 \mathsf H^{21})^T \|}
{\| W_1 \mathsf H^{12} \| + \| (W_2 \mathsf H^{21})^T \|}.
\]

**Acceptance:**

- roundoff for `EnergySymmetricHodge`,
- demonstrably small and convergent for a mass-lumped approximation,
- failure if defect is resolution-independent and large.

---

### 4.3 Positive kinetic-energy quadratic form

**Purpose:** Ensure the Hodge map defines positive kinetic energy.

Build the symmetric kinetic-energy operator

\[
\frac12(W\mathsf H + \mathsf H^T W)
\]

and verify the minimum eigenvalue is positive on representative small grids.

**Acceptance:** positive minimum eigenvalue for all unmasked degrees of freedom.

**Failure implication:** reject Hodge candidate; it can support negative-energy modes.

---

### 4.4 Free-stream preservation

**Purpose:** Check that a constant physical velocity does not generate fake divergence or tracer tendencies.

Given constant Cartesian velocity `u0`, compute covariant components analytically and Hodge-map them to volume transport:

\[
\mathcal V^i = \mathsf H^i{}_j u_j.
\]

Then check:

\[
D_i \mathcal V^i \approx 0.
\]

Also test constant scalar transport:

\[
D_i(\mathcal V^i c_0) \approx 0.
\]

**Acceptance:** roundoff for affine grids; convergent to zero for curved grids. This is a red-line test.

---

### 4.5 Orthogonal-limit consistency

**Purpose:** Ensure the non-orthogonal algorithm reduces to the existing orthogonal algorithm.

Construct a family of grids with skewness parameter `ε` such that

\[
g^{12}=O(\epsilon).
\]

Check that, as `ε → 0`, the non-orthogonal Hodge map reduces to the orthogonal metric formula and the VI tendencies agree with existing Oceananigans `VectorInvariant`.

**Acceptance:** difference is `O(ε)` and exactly/near-roundoff at `ε = 0`.

---

### 4.6 Hodge candidate scorecard

For every candidate Hodge map, record:

| Test | TargetMetricHodge | ProductInterpolatedHodge | EnergySymmetricHodge |
|---|---:|---:|---:|
| known-vector convergence |  |  |  |
| weighted adjointness |  |  |  |
| positive kinetic energy |  |  |  |
| free-stream preservation |  |  |  |
| orthogonal limit |  |  |  |
| centered VI stability |  |  |  |
| cost / memory |  |  |  |

**Decision rule:** use the simplest Hodge map that passes positivity, adjointness, free-stream preservation, and orthogonal-limit tests. If simple maps fail, use or mass-lump `EnergySymmetricHodge`.

---

## 5. Topology, seams, and halo tests

The current centered VI stability depends on fold-strip/tendency-mask regularization. That is acceptable for development but cannot be the final topological solution.

---

### 5.1 Paired vector seam-fill consistency

**Purpose:** Verify vector fields crossing OHPSG seams are filled with correct index rotation and component rotation.

**Must test fields:**

- covariant velocity pair `(u, v)`,
- contravariant velocity pair,
- transport velocities,
- volume transport,
- mass transport,
- relative vorticity where applicable.

**Procedure:**

- initialize a globally smooth analytic vector field,
- fill halos across all seams,
- compare halo values to analytic values,
- test both directions of each seam.

**Acceptance:** roundoff or expected interpolation error; no sign flips or index permutations wrong.

---

### 5.2 Seam continuity of reconstructed quantities

**Purpose:** WENO stencils must not see coordinate artifacts as physical discontinuities.

For smooth analytic flow, compute:

- relative vorticity `ζ12`,
- kinetic energy,
- covariant velocity components,
- transport divergence,
- WENO smoothness indicators.

Check continuity across seams/folds after halo fill.

**Acceptance:** seam jumps converge away or are roundoff for values that should be exactly continuous.

---

### 5.3 Fold-strip mask dependency test

**Purpose:** Identify whether stability is due to correct topology or due to ad hoc masking.

Run centered VI gates with:

1. current fold-strip/tendency mask enabled,
2. mask disabled after paired seam fills are implemented,
3. mask partially disabled one strip at a time.

**Acceptance for final implementation:** no fold-strip zeroing required except for genuinely constrained or singular topology cells with documented physical/numerical meaning.

---

### 5.4 Seam-local failure locator

**Purpose:** Preserve the diagnostic that found the west/east fold instability.

Track:

- max `|u|`, `|v|`, locations,
- alternating row/column projections,
- seam-local norms,
- kinetic energy contribution by seam bands.

**Acceptance:** no seam-local runaway; seam-band diagnostics comparable to interior-band diagnostics for smooth problems.

---

## 6. Centered VectorInvariant tests

The centered scheme is the foundation. WENO cannot be trusted until centered VI is correct.

---

### 6.1 Centered VI random-vortex OHPSG gate

**Current known passing setup:**

```julia
SphericalShellGrid(CPU(), Float64;
    mapping = OctaHEALPixMapping(32),
    z = (0, 1),
    radius = 1,
    halo = (5, 5, 3),
)

HydrostaticFreeSurfaceModel(
    grid;
    tracers = (),
    buoyancy = nothing,
    coriolis = nothing,
    free_surface = nothing,
    closure = nothing,
    momentum_advection = VectorInvariant(),
)
```

Initial condition: deterministic random-vortex streamfunction with `Random.seed!(42)`.  
Time step: `dt = 9.39248163e-03`.  
Gate: `533` steps to `t = 5.006193`.

**Current passing result:**

```text
step=533 t=5.006193 max|u|=1.145911e-01 max|v|=9.237383e-02
PASS
```

**Must keep passing.**

---

### 6.2 Centered VI energy budget

**Purpose:** Verify no hidden energy source from metric/Hodge/seam errors.

With no forcing, no closure, and centered nondissipative operators, track:

\[
E_h = \frac12 \langle u, \mathsf H u \rangle.
\]

**Acceptance:** energy drift consistent with time-integration error and expected boundary/mask effects; no monotone growth from the spatial operator.

---

### 6.3 Centered VI relative-vorticity diagnostics

Track:

- maximum relative vorticity,
- relative-vorticity variance/enstrophy proxy,
- seam-band vorticity diagnostics,
- grid-scale spectral tail.

**Acceptance:** no unexplained grid-scale growth, no seam-local vorticity explosion, no checkerboard branch.

---

### 6.4 Solid-body rotation / smooth exact-flow test

**Purpose:** Ensure the vector-invariant operator respects basic spherical geometry.

Initialize solid-body rotation on the sphere. Test:

- transport divergence is zero,
- relative vorticity and absolute vorticity are smooth and correct,
- tendency is zero or matches known forcing/pressure balance depending on setup.

**Acceptance:** convergence with resolution; no seam artifacts.

---

### 6.5 Orthogonal-grid regression

Run centered `VectorInvariant()` on an orthogonal grid and verify:

- existing Oceananigans results unchanged,
- non-orthogonal pathway in orthogonal limit agrees with old pathway,
- no performance or accuracy regression.

---

## 7. WENO VectorInvariant tests

The WENO path must be tested by decomposing it into sub-operators. A full simulation pass is not enough.

---

### 7.1 WENO divergence/self-upwind-only gate

**Current known result:** passes the OHPSG `t≈5` gate.

```text
weno_divergence_only:
  PASS t=5.006193
  step=533 max|u|=4.749623e-02 max|v|=4.016104e-02
```

**Must keep passing.**

**Purpose:** Establish that the WENO divergence/self-upwinding component is not the current blocker.

---

### 7.2 WENO vorticity-only gate

**Current known result:** fails.

```text
weno_vorticity_only:
  FAIL step=80 t=0.751399
  max|u|≈1.10e6 max|v|≈5.58e5
```

**Final requirement:** must pass without fallback.

**Additional required diagnostics:**

- confirm WENO reconstructs `relative_vorticity`, not `absolute_vorticity`,
- inspect seam-crossing WENO stencils,
- test WENO smoothness indicators across seams,
- compare with centered rotational advection on same fields,
- test on an affine skew grid where no seam topology exists.

---

### 7.3 WENO kinetic-energy-gradient-only gate

**Current known result:** fails.

```text
weno_ke_only:
  FAIL step=55 t=0.516586
  max|u|≈2.19e3 max|v|≈1.17e3
```

**Final requirement:** must pass without fallback.

**Additional required diagnostics:**

- compare centered and WENO covariant KE-gradient tendencies pointwise,
- test whether biased interpolation breaks Hodge adjointness,
- test whether KE-gradient WENO crosses invalid seams,
- test product-weighted versus target-metric KE-gradient reconstructions,
- verify no kinetic-energy source in centered limit.

---

### 7.4 Full true WENO VI gate

**Current known result before fallback:** failed at step 204.

```text
step=204 t=1.916066 max|u|≈4e5 max|v|≈4e5
FAIL
```

**Current fallback result:** passes, but this does not count as true WENO VI correctness.

```text
step=533 t=5.006193 max|u|=1.859528e-02 max|v|=1.489311e-02
PASS with centered rotational/KE fallback
```

**Final requirement:** full WENO vorticity + WENO KE-gradient + WENO divergence must pass without fallback.

**Test must assert:**

```julia
uses_centered_rotational_fallback(advection) == false
uses_centered_ke_gradient_fallback(advection) == false
```

or equivalent configuration introspection.

---

### 7.5 Fallback WENO VI gate

**Purpose:** Preserve development path while true WENO is under construction.

The fallback hybrid should have its own explicit test:

- centered covariant rotational advection,
- centered covariant Bernoulli/KE gradient,
- WENO divergence/self-upwind.

**Acceptance:** pass the OHPSG random-vortex gate. The test name and output must clearly say this is a fallback hybrid, not true WENO.

---

### 7.6 Small-Rossby WENO relative-vorticity test

**Purpose:** Ensure WENO dissipation acts on dynamically relevant relative vorticity, not smooth planetary vorticity.

Construct a small-Rossby flow with

\[
|f| \gg |\zeta|.
\]

Compare:

1. WENO reconstructing `absolute_vorticity = ζ + f`,
2. WENO reconstructing `relative_vorticity = ζ`, then adding smooth `f`.

**Acceptance:** the implemented scheme uses option 2; smoothness indicators and dissipation respond to `ζ`.

---

## 8. Hydrostatic split and divergence tests

These tests prevent double-counting divergence terms in hydrostatic versus compressible forms.

---

### 8.1 Hydrostatic vertical split identity

Verify the discrete identity for horizontal momentum components:

\[
D_3(\mathcal U^3 u_i) + u_i \mathcal D_{h,\mathcal U}
=
\mathcal U^3 D_3 u_i + u_i \mathcal D_{\mathcal U}.
\]

For hydrostatic-incompressible volume transport, use `volume_transport` instead of `mass_transport`.

**Acceptance:** identity holds to truncation error using the exact same discrete operators used by the tendency.

---

### 8.2 No double-counting divergence term

**Purpose:** Ensure implementation chooses one algebraic form, not both.

The tendency must use either:

1. full 3D VI form with `-u_i * mass_transport_divergence`, or
2. hydrostatic/WENO split with vertical flux divergence plus horizontal divergence flux,

but not both as independent corrections.

**Acceptance:** code path inspection plus manufactured discrete identity test.

---

### 8.3 Incompressible fixed-grid divergence

For fixed-grid hydrostatic-incompressible flow:

\[
D_i \mathcal V^i = 0.
\]

**Acceptance:** volume-transport divergence is roundoff/convergent zero for divergence-free manufactured fields and remains controlled in centered VI simulations.

---

### 8.4 Compressible mass-divergence consistency

For fully compressible Breeze later:

\[
\partial_t(J\rho) + D_i \mathcal U^i = 0.
\]

**Acceptance:** density tendency uses `mass_transport`, not covariant momentum. Scalar transport and density transport must share the same mass-transport diagnostics.

---

## 9. Scalar and tracer transport tests

VI correctness requires the transport velocities/fluxes that couple to tracers to be correct.

---

### 9.1 Constant tracer preservation

With divergence-free volume transport and constant tracer `c0`, verify:

\[
D_i(\mathcal V^i c_0) = 0.
\]

**Acceptance:** roundoff or expected convergence. Failure indicates transport velocity/metric inconsistency.

---

### 9.2 Tracer mass conservation

For closed or periodic domains, passive tracer integral must be conserved.

**Acceptance:** conservation to time-integration and solver tolerance.

---

### 9.3 Tracer variance behavior

For centered scalar advection with divergence-free transport, tracer variance should be conserved or have only known numerical changes. For WENO scalar advection, variance should dissipate monotonically or at least not grow spuriously.

---

### 9.4 `model.velocities` versus `model.transport_velocities` routing

For HFSM, verify tracer tendencies use `model.transport_velocities`, while momentum tendencies use `model.velocities`.

For Breeze, add equivalent tests for:

```julia
transport_velocities(model)
transport_fluxes(model)
```

**Acceptance:** scalar advection cannot accidentally consume covariant/model velocities on non-orthogonal grids.

---

## 10. Linear stability and computational-mode tests

These tests are needed because nonlinear turbulence can hide or delay computational modes.

---

### 10.1 Periodic affine skew-grid eigenanalysis

Use a periodic planar skew grid with constant metric. Linearize a simple shallow-water or acoustic system.

**Must check:**

- no positive-real-part eigenvalues for centered nondissipative operator,
- no checkerboard pressure/velocity branch,
- wave speeds match analytic skew-grid theory.

---

### 10.2 Variable-metric skew-grid eigenanalysis

Repeat on a smoothly varying metric grid.

**Purpose:** expose metric-placement instabilities that constant-metric tests cannot see.

**Acceptance:** no growing grid-scale modes; eigenvalue outliers understood and controlled.

---

### 10.3 Seam-local mode analysis

Construct perturbations localized near OHPSG folds/seams and measure growth under linearized centered VI.

**Acceptance:** no seam-local unstable modes after vector seam fill is implemented.

---

## 11. Multi-resolution, backend, and robustness tests

These are required before production use.

---

### 11.1 Resolution sweep

Run core gates at:

```text
N = 16, 32, 64
```

where feasible.

**Acceptance:** stability improves or remains controlled with resolution; convergence observed on manufactured tests.

---

### 11.2 Multiple random seeds

Run OHPSG random-vortex gates with at least five deterministic seeds.

**Acceptance:** no seed-specific blow-ups; statistics remain comparable.

---

### 11.3 Longer integration

Extend the random-vortex gate beyond `t≈5`:

```text
t ≈ 10, 25, 50
```

at least for nightly tests.

**Acceptance:** no delayed seam instability or energy/enstrophy pathology.

---

### 11.4 GPU and Float32

Run core gates on:

- CPU Float64,
- CPU Float32,
- GPU Float64 where supported,
- GPU Float32.

**Acceptance:** same qualitative stability; tolerances adapted for precision.

---

### 11.5 Closure and viscosity coupling

Test explicit viscosity and relevant turbulence closures. Include suspected polar-CFL-sensitive closures such as horizontal scalar diffusivity.

**Acceptance:** no new seam/fold or pole instability; closure stability limits understood.

---

## 12. Model integration tests

### 12.1 HFSM rigid-lid / no-free-surface gate

The current validated setup is in this category. It must remain the first HFSM gate.

**Acceptance:** centered VI and fallback WENO VI pass; true WENO VI must pass before final WENO claim.

---

### 12.2 HFSM tracer-coupled gate

Add passive tracers to the current no-free-surface setup.

**Must check:**

- tracer conservation,
- no transport noise near seams,
- scalar advection uses `transport_velocities`.

---

### 12.3 HFSM Coriolis gate

Add planetary vorticity / Coriolis.

**Must check:**

- relative vorticity is reconstructed with WENO,
- smooth planetary vorticity is added separately,
- no small-Rossby WENO blindness.

---

### 12.4 HFSM free-surface gates

These are later-stage, but must pass before claiming HFSM free-surface support.

Staged order:

1. static vertical coordinate with explicit free surface,
2. static vertical coordinate with implicit free surface,
3. split-explicit free surface,
4. z-star / moving vertical coordinate.

Each must check barotropic transport, vertical velocity from continuity, pressure correction, and tracer transport consistency.

---

### 12.5 Breeze compressible gate

For Breeze, verify:

\[
\partial_t(J\rho) = -D_i\mathcal U^i
\]

uses `mass_transport`, and that momentum uses the full compressible VI identity without the hydrostatic split unless deliberately selected.

Required tests:

- uniform flow,
- acoustic pulse or smooth compressible manufactured solution,
- passive tracer advection,
- forced 3D turbulence on skew grid,
- one-panel spherical-shell case.

---

## 13. Quantitative diagnostics that every model gate should log

Every VI model gate should log at least:

```text
step, time
max|u|, max|v|, max|w| if present
locations of max velocity components
max|relative_vorticity|
max|transport_divergence|
total kinetic energy
relative change in kinetic energy
relative-vorticity variance / enstrophy proxy
seam-band kinetic energy
seam-band relative vorticity
minimum/maximum metric determinant
NaN/Inf count
```

For WENO gates, also log:

```text
which sub-operators are WENO vs centered fallback
maximum WENO smoothness indicator near seams
maximum WENO smoothness indicator away from seams
vorticity-only tendency norm
KE-gradient-only tendency norm
divergence-only tendency norm
```

---

## 14. Required broken tests during development

Some tests should exist before they pass. Mark them `@test_broken` or equivalent until the implementation is ready.

Required broken tests now:

1. true WENO vorticity-only OHPSG gate,
2. true WENO KE-gradient-only OHPSG gate,
3. true full WENO VI OHPSG gate without fallback,
4. paired vector seam-fill replacement of fold-strip masks,
5. metric-duality residual beyond arithmetic cross-metric probe,
6. energy-symmetric Hodge candidate comparison,
7. small-Rossby relative-vorticity WENO test.

A final release claim requires these to be passing or explicitly scoped out.

---

## 15. Pass/fail definition for implementation claims

### Claim A: centered non-orthogonal `VectorInvariant` is implemented correctly

Required passing tests:

- grid/metric tests,
- Hodge consistency, positivity, adjointness, and free-stream preservation,
- paired seam-fill or documented stable topology treatment,
- centered random-vortex gate,
- centered energy and vorticity diagnostics,
- orthogonal-limit regression,
- at least one resolution sweep.

### Claim B: fallback OHPSG `WENOVectorInvariant` is stable

Required passing tests:

- all Claim A tests,
- WENO divergence-only gate,
- fallback full WENO gate,
- explicit introspection that rotational and KE-gradient terms are centered fallback.

### Claim C: true non-orthogonal `WENOVectorInvariant` is implemented correctly

Required passing tests:

- all Claim A tests,
- WENO divergence-only gate,
- WENO vorticity-only gate,
- WENO KE-gradient-only gate,
- true full WENO gate without fallback,
- small-Rossby relative-vorticity WENO test,
- seam-crossing WENO smoothness test,
- WENO convergence and nonlinear turbulence spectrum tests,
- at least one GPU/Float32 robustness check before production use.

### Claim D: HFSM non-orthogonal VI support is complete

Required passing tests:

- Claim A or C, depending on centered or WENO support,
- tracer-coupled HFSM tests,
- Coriolis tests,
- relevant free-surface stage tests if free surface is in scope.

### Claim E: Breeze compressible non-orthogonal VI support is complete

Required passing tests:

- Claim A or C,
- compressible mass-transport continuity tests,
- scalar transport uses same mass transport,
- full 3D compressible VI identity tests,
- acoustic/compressible manufactured solution or equivalent benchmark.

---

## 16. Recommended test file layout

```text
test/Grids/test_spherical_shell_grid_metrics.jl
test/Grids/test_octahealpix_cross_metrics.jl
test/Grids/test_octahealpix_vector_halo_fill.jl

test/Operators/test_hodge_consistency.jl
test/Operators/test_hodge_adjointness.jl
test/Operators/test_hodge_positivity.jl
test/Operators/test_free_stream_preservation.jl
test/Operators/test_orthogonal_limit.jl

test/Advection/test_vector_invariant_centered_ohpsg.jl
test/Advection/test_weno_vi_decomposition_ohpsg.jl
test/Advection/test_weno_relative_vorticity_reconstruction.jl
test/Advection/test_hydrostatic_split_identity.jl
test/Advection/test_nonorthogonal_scalar_transport.jl

test/Models/test_hfsm_ohpsg_rigid_lid_vi.jl
test/Models/test_hfsm_ohpsg_tracer_transport.jl
test/Models/test_hfsm_ohpsg_coriolis.jl
test/Models/test_breeze_nonorthogonal_compressible.jl
```

---

## 17. Immediate next tests to add

The next test-writing sprint should add, in order:

1. `test_octahealpix_cross_metrics.jl`: formalize the cross-metric probe.
2. `test_vector_invariant_centered_ohpsg.jl`: formalize the centered random-vortex gate.
3. `test_weno_vi_decomposition_ohpsg.jl`: preserve vorticity-only, KE-only, divergence-only diagnostics.
4. `test_weno_vi_fallback_ohpsg.jl`: explicitly test the fallback hybrid.
5. `test_free_stream_preservation.jl`: analytic constant-vector Hodge + divergence test.
6. `test_hodge_adjointness.jl` and `test_hodge_positivity.jl`: begin the Hodge candidate campaign.
7. `test_octahealpix_vector_halo_fill.jl`: prepare to replace fold-strip masks.

These tests provide a path from the current stable centered/fallback state to a defensible true non-orthogonal WENO VectorInvariant implementation.
