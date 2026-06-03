# WENO Vector-Invariant Status on OctaHEALPix SphericalShellGrid

Date: 2026-06-02

This note records the validation evidence for the current `VectorInvariant` and
`WENOVectorInvariant` implementation on `SphericalShellGrid` with
`OctaHEALPixMapping`, plus the remaining gaps and hypotheses.

## Scope of the validated setup

The validation below applies to:

- Grid: `SphericalShellGrid(CPU(), Float64; mapping=OctaHEALPixMapping(32), z=(0, 1), radius=1, halo=(5, 5, 3))`
- Model: `HydrostaticFreeSurfaceModel`
- Dynamics: 2D horizontal flow, `tracers=()`, `buoyancy=nothing`, `coriolis=nothing`
- Surface/closure: `free_surface=nothing`, `closure=nothing`
- Initial condition: deterministic random-vortex streamfunction IC with `Random.seed!(42)`
- Time step: `dt = 9.39248163e-03`
- Gate length: `533` steps, final `t = 5.006193`

## Tests and probes run directly by Codex

### 1. Cross-metric consistency probe

Script: `/tmp/metric_cross_probe.jl`

Purpose:

Verify that OHPSG staggered cross metrics now satisfy arithmetic averaging from
cell-center cross metrics on interior faces.

Result:

```text
metric_cross_probe max_x=0.0000000000000000e+00 max_y=0.0000000000000000e+00
```

Known from this:

The current grid construction exactly enforces, on the probed interior faces,

```text
G¹²ᶠᶜᵃ = avg_i(G¹²ᶜᶜᵃ)
G²¹ᶜᶠᵃ = avg_j(G¹²ᶜᶜᵃ)
```

This was the intended metric-duality patch.

### 2. Centered VI failure locator before west/east fold-strip fix

Script: `/tmp/vi_centered_locate_330.jl`

Purpose:

Locate the remaining centered `VectorInvariant()` instability after the
cross-metric averaging patch.

Observed failure before the fold-strip/tendency-mask fix:

```text
step=300 t=2.817744 maxu=9.839135e-01 at (5, 17)
step=309 t=2.902277 maxu=3.287807e+02 at (10, 18) maxv=1.725717e+01 at (9, 19)
step=310 t=2.911669 maxu=5.409780e+04 at (11, 18) maxv=2.839468e+03 at (10, 19)
```

Alternating row projection also exploded:

```text
step=310 altu[j=18]=5.855861e+02 altv[j=19]=4.913486e+01
```

Known from this:

After fixing cross metrics, the next centered VI failure was localized near the
west/east x-fold seam at low `i`, especially rows `j=17:19`. It was no longer
the original early high-latitude diamond-row failure.

### 3. Centered VI locator after west/east fold-strip and tendency-mask fix

Script: `/tmp/vi_centered_locate_330.jl`

Purpose:

Check whether the west/east fold-strip regularization removes the step-310
centered VI blow-up.

Result:

```text
step=330 t=3.099519 maxu=1.297726e-01 at (63, 11) maxv=1.376698e-01 at (28, 63)
```

Known from this:

The same centered locator that previously blew up at step 310 reached step 330
with velocities still O(0.1). The west/east seam-local runaway was suppressed
by the current fold-strip/tendency-mask treatment.

### 4. Full centered VI gate

Script: `/tmp/vi_weno_gate_after_patch.jl`

Case:

```julia
momentum_advection = VectorInvariant()
```

Result:

```text
CASE VectorInvariant dt=9.39248163e-03 nsteps=533
step=533 t=5.006193 max|u|=1.145911e-01 max|v|=9.237383e-02
PASS t=5.006193
```

Known from this:

Centered `VectorInvariant()` is stable to `t≈5` for the validated setup.

### 5. Full WENO VI gate before OHPSG WENO fallback

Script: `/tmp/vi_weno_gate_after_patch.jl`

Case:

```julia
momentum_advection = WENOVectorInvariant(order=5, vorticity_order=5)
```

Result before fallback patch:

```text
CASE WENOVectorInvariant{3, Float64}(vorticity_order=5, vertical_order=5)
step=204 t=1.916066 max|u|=4.013661e+05 max|v|=4.360325e+05
FAIL step=204 t=1.916066
```

Known from this:

The original OHPSG WENO VI implementation was unstable even after the centered
VI fixes.

### 6. WENO VI decomposition probe

Script: `/tmp/vi_weno_decompose.jl`

Purpose:

Isolate which WENO sub-operators are unstable on OHPSG.

Cases and results:

```text
weno_vorticity_only:
  FAIL step=80 t=0.751399
  max|u|≈1.10e6 max|v|≈5.58e5

weno_ke_only:
  FAIL step=55 t=0.516586
  max|u|≈2.19e3 max|v|≈1.17e3

weno_divergence_only:
  PASS t=5.006193
  step=533 max|u|=4.749623e-02 max|v|=4.016104e-02
```

Known from this:

The WENO vorticity reconstruction and WENO kinetic-energy-gradient
reconstruction are independently unstable on OHPSG in this setup. The WENO
divergence/self-upwind path is not the observed blocker and passed the `t≈5`
gate independently.

### 7. Full WENO VI gate after OHPSG fallback

Script: `/tmp/vi_weno_gate_after_patch.jl`

Implementation state:

On OHPSG, `WENOVectorInvariant` currently:

- falls back to centered covariant rotational advection,
- falls back to centered covariant Bernoulli head / KE gradient,
- retains the WENO divergence/self-upwind path.

Result:

```text
CASE WENOVectorInvariant{3, Float64}(vorticity_order=5, vertical_order=5)
step=533 t=5.006193 max|u|=1.859528e-02 max|v|=1.489311e-02
PASS t=5.006193
```

Known from this:

The current OHPSG `WENOVectorInvariant` configuration is stable to `t≈5` for
the validated setup, but it is not a full non-orthogonal WENO implementation for
rotational advection or kinetic-energy gradients.

## Other-agent reported validation

The other agent reported an animation diagnostic for centered VI:

- Setup: OHPSG `N=32`, centered VI, same failing random-vortex IC
- Length: `100` frames × `6` substeps = `600` steps, final `t≈5.63`
- File: `/tmp/anim_stable.mp4`
- Log: `/tmp/anim_stable.log`

Reported trajectory:

```text
t=0.0  max|u|=0.0723  max|ζ|=5.94
t=1.1  max|u|=0.0974  max|ζ|=41.8
t=2.3  max|u|=0.112   max|ζ|=67.8
t=3.4  max|u|=0.109   max|ζ|=48.0
t=4.5  max|u|=0.0973  max|ζ|=37.4
t=5.6  max|u|=0.0992  max|ζ|=47.2
```

Reported qualitative result:

- visually clean decaying 2D turbulence,
- no obvious 2Δ checkerboard noise,
- no visible over-damping near the diamond seams,
- smooth vortex structures over polar caps.

This is useful corroboration, but it was not a Codex-run test.

## What is known absolutely under the tested setup

1. The metric cross-term averaging probe passes exactly on the checked interior
   faces.

2. Centered `VectorInvariant()` passed the `t≈5` random-vortex OHPSG turbulence
   gate on CPU/Float64/N=32 with no closure and no free surface.

3. The pre-fallback `WENOVectorInvariant` failed on the same gate.

4. WENO vorticity-only failed independently.

5. WENO kinetic-energy-gradient-only failed independently.

6. WENO divergence/self-upwind-only passed independently.

7. The current `WENOVectorInvariant` passes the `t≈5` gate only because OHPSG
   rotational advection and OHPSG Bernoulli/KE-gradient terms fall back to
   centered covariant operators.

8. Therefore the current WENO VI implementation is stable for the tested setup,
   but it is not a complete true WENO VI discretization on OHPSG.

## What has not been tested

- GPU execution.
- Float32 execution.
- Other OHPSG resolutions such as `N=16`, `N=64`, or convergence with `N`.
- Long integrations beyond `t≈5` / the reported animation to `t≈5.6`.
- Multiple random seeds or a broad IC ensemble.
- Forced turbulence.
- Explicit viscosity or turbulence closures.
- The suspected `HorizontalScalarDiffusivity` polar-CFL issue.
- Free-surface cases.
- Coriolis cases.
- Tracer coupling with WENO VI dynamics.
- Shallow-water dynamics.
- Rossby-Haurwitz wave.
- Spherical Bickley jet.
- Quantitative KE/enstrophy budgets after the final patch.
- Quantitative seam damping/error metrics after the final patch.
- Conservation/invariance under solid-body rotation.
- Formal test-suite integration in `test/`.
- The full Oceananigans test suite.
- Production-scale turbulent simulations.
- True WENO vorticity reconstruction on OHPSG.
- True WENO covariant kinetic-energy-gradient reconstruction on OHPSG.
- Replacement of the fold-strip masks with a topologically correct rotated
  vector seam fill.

## Energy-conservation validation on 2026-06-02

Codex ran direct spatial kinetic-energy tendency probes using the current
random-vortex OHPSG setup. The diagnostic computed

```text
(E(u + εG) - E(u - εG)) / (2ε)
```

where `G` is the model-computed momentum tendency and `E` is the covariant
cell-centered kinetic energy integrated with `Vᶜᶜᶜ`.

### Restored/current code result

Centered `VectorInvariant()`:

```text
E0 = 6.5927197687859762e-01
dE/dt ≈ -7.87136e-05
relative dE/dt ≈ -1.19395e-04
```

The value was independent of `ε = 1e-4 ... 1e-8`, so it is a real spatial KE
tendency for this setup, not finite-difference noise.

Current fallback `WENOVectorInvariant(order=5, vorticity_order=5)`:

```text
E0 = 6.5927197687859762e-01
dE/dt ≈ -2.09214e-02
relative dE/dt ≈ -3.17341e-02
```

This is expected to be dissipative because the current OHPSG WENO path retains
WENO divergence/self-upwind.

### Divergence check

The same random-vortex IC is not discretely divergence-free in raw covariant
velocity form:

```text
max_div = 2.4673260090757507e+00
l1_fluxdiv = 5.4050127550787941e-01
sum_fluxdiv = -5.0430550263861376e-17
K_fluxdiv_sum = -2.3718724914097381e-03
```

Therefore this IC is not a clean proof/disproof of exact centered VI
energy-conservation for incompressible transport. It is a useful model gate, but
not a mathematically isolated energy-conservation test.

### Repairs attempted and reverted

Two attempted repairs were tested and reverted because they worsened centered KE
conservation:

1. Use corrected transport velocities in the projected rotational VI path for
   centered VI:

```text
centered dE/dt worsened to ≈ -1.92244e-03
```

2. Remove strip-zero masks and make paired vector halo fill overwrite high-side
   duplicate seam faces (`u[Nx+1,*]`, `v[*,Ny+1]`):

```text
centered dE/dt worsened to ≈ -1.29702e-03
```

Both changes were reverted. The current strip-stabilized code remains the best
validated state so far, but exact centered VI energy conservation has not been
proved.

### Current energy-conservation conclusion

Centered VI on OHPSG is not exactly energy-conserving for the current
random-vortex model gate. The measured energy loss is small compared with WENO,
but nonzero.

The most likely remaining issue is not a simple removable line of code. A valid
repair likely requires a proper topological vector seam treatment and a
discretely divergence-free/incompressible energy test. The ad hoc strip-zero
regularization is still a stabilization proxy, but simply deleting it makes KE
conservation worse and destabilizes the seam treatment.

## Remaining hypotheses

### H1. The fold-strip masks are stabilizing proxies, not the final topology

The current fold-strip regularization zeros selected polar and west/east seam
velocity/tendency DOFs. This stabilizes the tested centered and fallback-WENO
gates.

The remaining hypothesis is that this should ultimately be replaced by a
topologically correct paired vector seam fill using the existing OHPSG
rotation/index machinery, especially
`octahealpix_halo_source_ring_index_and_rotation`.

Expected final direction:

- fill covariant and contravariant vector seams as paired `(u, v)` fields,
- rotate components across seams,
- remove or reduce ad hoc strip-zero constraints once the true vector seam
  transform is correct.

### H2. WENO vorticity fails because reconstruction crosses OHPSG seams/folds incorrectly

The WENO vorticity-only failure strongly suggests that the vorticity
reconstruction is not respecting non-orthogonal OHPSG seams, folds, or vector
component rotations.

Possible causes:

- biased stencils cross folded seams without applying the correct index
  permutation,
- reconstructed covariant velocity/vorticity values use inconsistent rotated
  components,
- the reconstruction sees seam discontinuities that are coordinate artifacts,
  not physical discontinuities.

### H3. WENO KE-gradient fails because the biased KE-gradient path breaks metric duality

The WENO KE-only failure suggests that biased reconstruction of
`covariant_kinetic_energyᶜᶜᶜ` gradients is not compatible with the
face-Hodge/metric-dual kinetic energy construction on OHPSG.

Possible causes:

- biased interpolation of the covariant KE gradient breaks the discrete
  adjoint relation restored by centered face-Hodge KE,
- the KE stencil crosses fold/seam regions without a valid scalar/vector
  topology treatment,
- cross-metric terms need a WENO-specific metric/area weighting rather than
  direct reuse of the orthogonal-grid WENO template.

### H4. WENO divergence/self-upwind is currently the safest WENO component

The divergence-only WENO case passed to `t≈5`, so the best near-term stable
WENO VI path is to retain WENO divergence/self-upwind while using centered
covariant rotational and Bernoulli terms on OHPSG.

This is the current implemented fallback.

### H5. Cross-metric arithmetic averaging is necessary for centered stability

The centered VI improvements after cross-metric averaging support the hypothesis
that analytic face cross metrics broke a discrete metric-duality identity on
skew OHPSG rows.

This still needs more formal confirmation via a duality-residual test added to
the test suite.

## Recommended next validation goals

1. Add a formal OHPSG metric-duality test for cross-metric averaging.

2. Add a formal centered VI random-vortex gate at modest `N` and short enough
   runtime for CI or nightly validation.

3. Add a WENO VI fallback gate that explicitly asserts OHPSG WENO remains stable
   with centered rotational/Bernoulli fallback.

4. Add a decomposition test or diagnostic for WENO vorticity-only and
   WENO KE-only failures, possibly marked broken until true non-orthogonal WENO
   is derived.

5. Implement and validate rotated paired vector seam fills, then test whether
   fold-strip zeroing can be removed.

6. After seam fills are correct, revisit true WENO vorticity and WENO
   KE-gradient discretizations on OHPSG.
