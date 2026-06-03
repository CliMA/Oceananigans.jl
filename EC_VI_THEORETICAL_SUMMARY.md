# Energy-Conserving VI on OctaHEALPix: Theoretical Summary

A summary of the geometric and numerical-analysis context for the
ongoing work on an energy-conserving vector-invariant momentum
advection on the OctaHEALPix `SphericalShellGrid` (OHPSG).

---

## 1. The non-orthogonal C-grid setting

### 1.1 The OctaHEALPix mapping

The OctaHEALPix sphere covering tiles `S²` into `4N²` equal-area
cells arranged as a single `2N × 2N` matrix in `(i, j)` index
space. Latitudes are stratified into `2N - 1` rings; the cells per
ring vary as `min(4j, 8N - 4j)`. The mapping is conformal on each
panel but introduces a **fold**: the `(i, j)` matrix wraps onto the
sphere via the `QuadFolded` topology, with a polar fold at `j=1`,
`j=Ny+1` and an east-west seam at `i=1`, `i=Nx`.

Each cell carries a `2×2` covariant metric `gₐᵦ` and a Jacobian
`J = √det g`, both spatially varying. The grid is **non-orthogonal**:
`g₁₂ ≠ 0` away from the equator, with `g₁₂` reaching `O(1)` at the
polar tip and the seam. The conjugate (contravariant) metric is
`Gᵃᵇ = (g⁻¹)ᵃᵇ`, satisfying `g·G = J·I` (an invariant verified by
`test_octahealpix_metric_invariants.jl`).

### 1.2 Covariant and contravariant velocity

On a non-orthogonal grid the prognostic velocity `u` admits two
distinct natural representations:

- **Covariant components** `uₐ`: project onto the local coordinate
  basis. Naturally circulation-like; line integrals of `uₐ dxᵃ` give
  the discrete vorticity.
- **Contravariant components** `uᵃ`: project onto the *dual* basis.
  Naturally flux-like; `uᵃ Δxᵇ Δz` gives the volume flux through the
  face with normal in direction `c`.

The transformation is via the metric: `uᵃ = Gᵃᵇ uᵦ`. On orthogonal
grids the two coincide (up to scaling) and the distinction collapses;
on OHPSG, the off-diagonal `g₁₂` mixes them.

We store **covariant velocities** as prognostic. Fluxes, vorticities,
and the kinetic-energy form all require the metric transformation,
which happens at every face.

### 1.3 The Hodge map

The discrete *Hodge* map `H` sends covariant velocities to volume
fluxes:

```
F = H u   ⟺   F_face = (Δx_face Δz) · G^{ab}_face · u_b_face
```

Energy is the natural pairing under `H`:

```
KE = ½ ⟨u, H u⟩  =  ½ ∫ G^{ab} u_a u_b dV
```

`H` is **block-diagonal in faces** (each face has its own 2×2
metric tile), and positive-definite. The standard L²-style inner
product for momentum is then `⟨u, v⟩_W = ⟨u, H v⟩`.

### 1.4 Vector-invariant momentum advection

The vector-invariant form of the momentum equation is:

```
∂_t u_a + ζ × u^a + ∇_a (½ u·u) + advection_z + Coriolis + pressure = 0
```

where:
- `ζ × u^a` is the **rotational** flux (curl of velocity crossed
  with contravariant velocity, in 2D this reduces to `-ζ U^y, +ζ U^x`).
- `∇_a(½ u·u)` is the **Bernoulli** head (gradient of kinetic energy
  density).
- The two together equal `(u·∇)u_a` in the continuum.

This decomposition has nice properties on a C-grid: `ζ` lives
naturally at `F-F` corners, and the gradient of a scalar `½u·u` is
the natural staggering for the Bernoulli term. On orthogonal grids
each piece is **individually** energy-conserving in the appropriate
discrete sense (Sadourny 1975, Arakawa-Lamb 1981).

### 1.5 Rigid-lid projection

For incompressible flow with `free_surface = nothing`, the pressure
is determined by a Poisson constraint:

```
div(K⁻¹ ∇p) = div(u*)
```

where `K = H/2` is the per-face mass-matrix scaling. The discrete
operator factorizes elegantly: `D · K⁻¹ · Dᵀ = (D · H⁻¹) · (D · H⁻¹)ᵀ`
when `K = H/2`, where `D · H⁻¹ ≡ B̃` is a sparse boundary-respecting
divergence. This factorization gives a positive-semidefinite Poisson
operator with a one-dimensional null space (constant pressure),
solved by preconditioned CG.

Numerical positivity of `H` on the independent-DOF subspace is
verified by `test_hodge_positivity.jl` (min eigenvalue 4.82e-3).

---

## 2. Energy conservation as a discrete identity

In the continuum, both rotational and Bernoulli pieces are
**individually** energy-conserving against any divergence-free
velocity:

```
∫ u · (ζ × U) dV   = 0    (ζ × U is solenoidal)
∫ u · ∇(½|U|²) dV  = 0    (gradient of scalar, on div-free u)
```

These follow from vector calculus identities and integration by parts.

In the **discrete** setting, the two cancellations are **not
automatic**: each is a separate stencil property that depends on:

1. How `ζ` is computed at the F-F corner (line integral around the
   cell, with halo or topology mapping at the seam).
2. How `U_contra` is averaged from F-C and C-F faces to the F-F
   corner.
3. How `½ u·u` is averaged from F-C and C-F faces to the C-C cell
   center.
4. How the gradient `∇_a` is taken back from C-C to F-C and C-F.

The Sadourny (1975) and Arakawa-Lamb (1981) constructions identify
**specific corner-weighting choices** for the orthogonal C-grid such
that the discrete energy identity holds **at every cell**, including
boundaries with appropriate ghost-cell treatment. These constructions
have an additional 8-12-factor freedom that can be tuned for
**enstrophy** conservation in addition to energy.

The construction does NOT trivially extend to non-orthogonal grids,
and AL-1981 does not address topology-folded seams.

---

## 3. What we built and tested

### 3.1 Validated infrastructure

- **Cross-metric averaging**: F-C and C-F face metrics are arithmetic
  averages of the four neighboring C-C values. Locked in by
  `test_octahealpix_cross_metrics.jl`.
- **Halo fill on the seam**: `QuadFoldedCovariantZipperBoundaryCondition`
  applies a rotation-with-sign-flip across the seam. Validated by
  `test_octahealpix_vector_halo_fill.jl` (2380 PASS) — confirms that
  the production halo values match the topology-aware source-face
  lookup to roundoff.
- **Metric invariants**: `g · G = J · I` to roundoff at all cells in
  Float32 and Float64.
- **Area closure**: `Σ Az = 4π · R²` to roundoff at `N ∈ {4, 8, 16}`.
- **Hodge positivity**: `H` is SPD on the seam-aware independent-DOF
  subspace with min eigenvalue ≈ 4.82e-3 at `N = 4`.
- **Rigid-lid projection**: preserves `max|div u| ≤ 1e-13` throughout
  N=32 integration. Initial-state projection hook in `update_state!`
  catches admissibility for arbitrary IC.

### 3.2 Failed patches (informative)

Four distinct attempts to fix the seam blow-up by **re-routing data**
through the topology maps all failed:

| Patch                                | N=32 failure step | vs. baseline 375 |
|--------------------------------------|-------------------|------------------|
| Combined ζ + U_contra topology remap | 159               | 2.4× *worse*     |
| Covariant ζ remap only               | (no change)       | no-op            |
| Scalar KE remap before Bernoulli     | 252               | 1.5× *worse*     |
| Metric-aware contravariant flux remap| 159               | 2.4× *worse*     |

Three of four made things worse. The fourth was a no-op. Conclusion:
the halo cells contain the correct topology-aware data; the **stencil**
that consumes them is what's wrong.

### 3.3 Quantitative measurement of the defect

Two diagnostics arrived at the same conclusion:

**(a)** Per-column energy production probe on div-free random IC
(amplitude 0.01, multiple seeds, `N ∈ {4, 8, 16}`):

```
Global Σᵢ P_col     max|max P_col|
N=4    O(10⁻⁷)          7.6 × 10⁻⁷
N=8    O(10⁻⁷)          1.5 × 10⁻⁷
N=16   O(10⁻⁸)          2.5 × 10⁻⁸
```

Total non-skew defect ∝ `1/N²` — consistent with a discretization
that's *consistent* (limits to the exact identity) but not exact
at any finite resolution.

At low `N` the seam columns (`i=1`, `i=Nx`) carry slightly elevated
defect (3-4× bulk). At `N=16` the seam dominance fades.

**(b)** Term-split decomposition of the same probe (Codex):

```
                  Rotational    Bernoulli       Sum (full VI)
N=4   seed 42    -8.376 × 10⁻⁷  +1.159 × 10⁻⁶   +3.216 × 10⁻⁷
N=8   seed 42    -1.596 × 10⁻⁷  +1.848 × 10⁻⁶   +1.688 × 10⁻⁶
N=16  seed 42    -1.213 × 10⁻⁶  +2.120 × 10⁻⁶   +9.063 × 10⁻⁷
```

The residual `(full - rotational - Bernoulli)` is `O(10⁻¹⁷)` — **the
defect lives entirely in the horizontal rotational + Bernoulli
coupling**. Not the vertical advection, not the Hodge projection,
not the Coriolis (which is absent in these tests), not the halo fill.

**At N=16 seam column i=1**:
- Bernoulli production: `+2.866 × 10⁻⁶`
- Rotational production: `−1.508 × 10⁻⁶`
- Net spurious injection: `+1.358 × 10⁻⁶` (positive growth)

These must cancel in the continuum; they don't in the discretization.

### 3.4 Why N=16 passes and N=32 fails

The full VI production is **cubic in velocity amplitude** (rotational
flux is bilinear in `u`, then inner-producted with `u` to get energy
production). At gate amplitude `a = 0.3`, the per-cell defect is
`(0.3 / 0.01)³ = 2.7 × 10⁴` times the probe value, scaled by `N²` cells.

For `N=32`, estimated e-folding time:

```
KE ≈ ½ a² A_sphere ≈ 0.56
|dKE/dt|_seam ≈ 1.4
τ_e ≈ KE / |dKE/dt| ≈ 0.4 sec
```

Observed N=32 blow-up: 3.5 sec, growth 0.3 → 10 = 33× = e^3.5.
Implied rate ≈ 1/sec. Order-of-magnitude consistent with the probe.

At `N=16` the integration runs for only 1.7 sec and uses lower
amplitudes; the same defect remains below the visible threshold.

---

## 4. Theoretical interpretation

The discrete identity that needs to hold is the **Sadourny / SBP
energy-conservation condition**:

```
For all div-free covariant u:
  ⟨u, G_rot[u]⟩_W + ⟨u, G_bern[u]⟩_W  =  0
```

where the inner product is the natural metric-weighted KE form.

This is *not* a constraint we can read off from the continuous PDE
— it must be **constructed** at the discrete level. The Sadourny
(1975) construction for orthogonal C-grids is:

```
G_rot[u]_F = -ℑ_F^FF [ζ_FF · U^y_FF]   (rotational)
G_bern[u]_F = ∇_F [½ ⟨u, u⟩_C]         (Bernoulli)
```

with **specific** corner-PV weighting (Q-form weights) such that
the contribution at each F-face from rotational exactly cancels
the contribution from Bernoulli on every div-free `u`.

On non-orthogonal grids, the corner-PV weighting must be **metric-
aware** (the corner ζ involves `g_{12}` cross-terms, and the
contravariant `U^y_FF` involves `G^{12}`). On a topology-folded
geometry like OHPSG, the corner ζ at i=1 and i=Nx is computed from
**rotated source faces** — the seam introduces an additional sign
structure that the Sadourny weights must absorb.

The current stencil uses the *standard* Sadourny form
(`covariant_vertical_circulationᶠᶠᶜ` × `contravariant_velocityᶠᶠᶜ`),
which satisfies the energy identity in the orthogonal limit. On
OHPSG the cross-metric terms break this, and the seam fold
contributes an additional uncancelled term.

### Open theoretical question

**Does an Arakawa-Lamb-type Q-form extension exist for non-orthogonal
quad-folded grids?**

The answer is almost certainly yes — the construction is
algebraic, and the seam map is a discrete isometry of the metric
(it's a rotation), so the SBP identity can be enforced by a finite
linear system in the corner weights. The construction is just not in
the published literature, and we'd have to derive it ourselves.

A constructive procedure:
1. Write the local rotational stencil at a generic seam corner in
   symbolic form, with unknown weights `w₁, ..., w_k`.
2. Write the local Bernoulli stencil with corresponding weights.
3. Sum and impose `⟨u, G_rot + G_bern⟩_W = 0` for arbitrary div-free
   `u`. This is a finite linear system in the weights.
4. Verify the same weights vanish the identity at all corners
   (interior, polar fold, west seam, east seam, four corners of
   the OHP fundamental polygon).

The numerical cost of the resulting stencil is comparable to the
current implementation — what changes is the choice of which
neighboring values are averaged, with what weights.

---

## 5. Status and forward paths

### What works at baseline

- N=16 short gate (180 steps to t=1.69) passes with bounded `|u|, |v| < 0.2`.
- Incompressibility regression: 5/6 pass + 1 `@test_broken`
  (the broken assertion is the raw-tendency divergence, which is
  not expected to be zero — it's the *projected* update that
  matters).
- All halo-fill, metric, area, and Hodge regressions pass.

### What remains broken

- N=32 extended gate (533 steps to t=5) fails at step ~375 with a
  seam-localized blow-up. Marked `@test_broken` while the EC
  derivation proceeds.

### Three forward paths

1. **Rigorous**: Derive the OHPSG Q-form weights. Few days to a
   week of derivation + implementation + test.
2. **Pragmatic**: Modify the Bernoulli term alone to exactly
   close against the existing rotational stencil. Less principled
   but faster.
3. **Stopgap**: Apply targeted dissipation at the seam columns.
   Unblocks downstream validation while EC work continues.

The rotational + Bernoulli term-split (Codex) and the per-column
scaling (reviewer) together give a clear quantitative target: any
new stencil must drive the per-column defect to roundoff
(`max|P_col| / KE < 10⁻¹²`) at every column, including seam.

This is a well-defined regression test that locks in whatever
stencil emerges from the derivation.
