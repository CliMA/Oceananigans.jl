# Goal: No-mask energy-conserving non-orthogonal vector-invariant dynamics

Date: 2026-06-02

## Objective

Implement and validate a correct staggered vector-invariant dynamics pathway for
non-orthogonal `SphericalShellGrid` / `OctaHEALPixMapping` grids, with no strip
masks, no artificial damping, no seam suppression, and no stability hacks.

The target is not merely a stable demo. The target is a derived mimetic C-grid
calculus in which covariant velocity one-forms, contravariant transports,
metric/Hodge maps, vorticity, rotational advection, and Bernoulli gradients are
compatible on the OHPSG seams, folds, and polar rows.

## Non-negotiable constraints

- No strip-zero velocity masks.
- No strip-zero tendency masks.
- No seam buffer damping, Laplacian smoothing, checkerboard filters, or other
  empirical stabilizers.
- No special cases that suppress a problematic degree of freedom without a
  topological or mimetic derivation.
- Paired vector halo/seam handling must be an index permutation plus vector
  rotation, not scalar copying or zeroing.
- Tests must include quantitative conservation/stability diagnostics, not only
  visual checks.

## Mathematical target

Use the non-orthogonal C-grid vector-invariant form as a mimetic finite-volume /
discrete exterior calculus operator:

- Covariant velocity components live on C-grid faces.
- Contravariant volume fluxes are obtained through a metric-compatible Hodge
  map.
- Vorticity is the exterior derivative of the covariant velocity one-form.
- Rotational advection and Bernoulli gradients are adjoint-compatible in the KE
  inner product.
- For admissible divergence-free transport, the centered spatial nonlinear
  operator satisfies

```text
<u, N(u)>_H = 0
```

to roundoff, including OHPSG seams, folds, duplicate high-side faces, and polar
rows.

## Current known state

The source currently has the old OHPSG strip-zero velocity regularizers and
strip-zero tendency masks removed. `rg "mask_octahealpix|regularize_octahealpix"
src test` should remain empty.

The centered spatial VI operator now passes direct KE identity tests on
admissible divergence-free OHPSG states at `N=4` and `N=8`, including nonzero
polar-fold transport. Rossby-Haurwitz vector-invariant accuracy tests also pass
for centered VI and the current WENO variants.

The direct spatial KE result is necessary but not sufficient. No-mask centered
dynamics still fails in time integration:

- Random-vortex OHPSG gate fails near the north polar row around `t≈1.44` with
  fixed transport CFL `0.3`.
- The same gate survives longer with smaller CFL but still fails later.
- An admissible Hodge-inverted divergence-free initial condition also fails in
  time stepping.
- Energy-over-time diagnostics show bounded KE error followed by rapid explicit
  blow-up, consistent with either polar metric stiffness or a remaining unstable
  polar/topological operator mode.

The polar-stiffness hypothesis has been tested on the documented centered
random-vortex gate and rejected for that gate. Halving the fixed time step down
to `dt/8` did not move the failure time; all runs failed near `t≈1.56-1.58`,
with the growing mode localized to the south polar interior band (`j≈3-4`).

This makes the current blocker a spatial/topological positive-growth mode, not
just a missing OHPSG timestep safety factor. The Hodge map is weighted-adjoint
and positive on the constrained independent face-DOF space, so the next target
is the assembled centered VI tendency: polar-row vorticity/rotational advection,
Bernoulli/KE-gradient, vertical/hydrostatic split, and their discrete energy
coupling.

The latest admissible-state diagnostics refine this further: a projected
discrete-divergence-free state still fails if it is evolved without pressure
projection, because the centered VI tendency does not preserve the discrete
rigid-lid constraint. A Hodge-weighted projection

```text
x <- x - K^{-1} D^T (D K^{-1} D^T)^{-1} D x
```

where `K` is the weighted covariant-to-volume-flux Hodge operator and `D` is the
OHPSG horizontal volume-flux divergence, annihilates the divergence to roundoff
and is energy non-increasing. A dense `N=8` prototype that applies this
projection after every step reaches the `t≈5` milestone. A Euclidean projection
fails immediately by exciting folded duplicate modes, and the existing
large-`g` `ImplicitFreeSurface` Helmholtz path fails at folded/polar faces.

Two tempting projection shortcuts have also been ruled out by
`test/Operators/test_hodge_projection.jl`: the dense Hodge correction
`K^{-1}D^T p` is not equivalent to the existing local covariant-gradient
free-surface correction, and it is not equivalent to the raw weighted adjoint of
a separately materialized volume-flux divergence operator. The production
projection must therefore reproduce the actual dense independent-face
`K^{-1}D^T` correction map, including OHPSG fold/duplicate-face topology, rather
than only matching a scalar Laplacian away from seams.

The constructive factorization is now identified. If `D` is the exact
independent-face covariant-velocity divergence matrix, `H` is the independent
covariant-to-volume-flux Hodge matrix, and `K = W H` is the weighted Hodge
energy matrix, then

```text
B_tilde = D H^{-1}
K^{-1} D^T p = W^{-1} B_tilde^T p
```

The recovered `B_tilde` has exactly four nonzero face entries per cell at
`OctaHEALPixMapping(4)`. Interior rows match the raw face-flux divergence
stencil, while folded/duplicate OHPSG seam rows have different coefficients.
Thus the production path is to implement the exact `B_tilde` topology and solve
the scalar projection system `B_tilde W^{-1} B_tilde^T p = D u`, rather than
solving a coupled face-space system.

The first source-level `B_tilde` building block is now implemented:
`hodge_compatible_volume_flux_div_xyᶜᶜᶜ` applies the exact Hodge-compatible
volume-flux divergence using ordinary interior flux differences plus
Hodge-ratio boundary flux maps derived from the covariant OHPSG seam transform.
It matches the dense `D H^{-1}` reference to roundoff at `N=4` in
`test/Operators/test_hodge_projection.jl`; an `N=8` diagnostic also matches to
roundoff. The source-level adjoint pressure-correction apply
`W^{-1}B_tilde^T p` and scalar Schur-complement action `D K^{-1}D^T p` are also
implemented and match dense references to roundoff in the same Tier-0 test. The
field-level rigid-lid projection scaffold now computes projection RHS,
pressure-correction fields, Schur-complement action, and velocity correction
application; the field-level Schur action matches dense reference in a focused
diagnostic. A standalone `RigidLidProjectionSolver` now solves the scalar
Schur-complement system with `ConjugateGradientSolver` and projects an OHPSG
random face state to the dense Hodge projection reference to roundoff. The
projection is now wired into `HydrostaticFreeSurfaceModel` construction for
`SphericalShellGrid + free_surface = nothing` as a first-class
`model.rigid_lid_projection` field and into the AB2/RK rigid-lid velocity update
path. The current validation target is the centered OHPSG random-vortex dynamics
gate, including shorter instrumented `N=8` and `N=16` runs before relying on the
expensive formal `N=32` gate.

The first dynamics evidence is positive: an instrumented `N=8` run passed 120
steps with post-projection divergence at `O(10^-13)`, and an instrumented `N=16`
run passed 220 steps through `t=2.0663459586`, crossing the old `t≈1.56` blow-up
window with `max|u|=0.1165218618490599`, `max|v|=0.10898999631347438`, and
maximum horizontal divergence `2.3050138187041824e-13`. The repository test gate
has been split into an always-on shorter `N=16`, 180-step gate with progress
logging and an opt-in `EXTENDED_OHPSG_VI_TESTS=true` `N=32`, 533-step gate.
Post-refactor `test/Advection/test_vector_invariant_centered_ohpsg_incompressibility.jl`
passes its projected-update assertion with maximum projected divergence
`4.659267763273967e-13`; the raw tendency divergence assertion remains
deliberately broken because the projection, not the raw explicit VI tendency, is
the constraint-preserving update.

However, the full formal `N=32`, 533-step centered random-vortex gate is not yet
passed. The original uninstrumented run failed at step 380,
`t=3.569143019399984`, with `max|u|=111709.44094213727` and
`max|v|=10546.190587007375`. Therefore the current milestone state is:
projection fixes the earlier `t≈1.56` failure mode and keeps short projected
dynamics divergence-controlled, but Claim A is still open until the `N=32`,
`t≈5` gate passes with divergence, energy, and seam diagnostics.

An instrumented `N=32` replay confirms that the projection holds
`max|D u|≈10^-13` through at least step 150 (`t=1.4088722445`) with CG iteration
counts around `770`, before that diagnostic process segfaulted inside Julia
kernel dispatch / tendency caching. This means the next correctness target is
not the early rigid-lid projection algebra, but the late-time projected
seam/fold energy mode seen near west/east faces in the failed full run.

A first-step admissibility gap has now been fixed: `update_state!` projects
rigid-lid SSG velocities at `clock.iteration == 0` before vertical velocity and
momentum tendency diagnostics. This prevents user `set!` initial conditions from
producing the first tendency from a non-Hodge-divergence-free state. A random
N=4 smoke test reduced initial divergence from `0.06854134202412751` to
`5.165453568350964e-13` during `update_state!`. For the N=32 random-vortex
initial condition, `set!` now leaves `max|D u|=1.1073000155681356e-13`, so the
formal gate should be rerun from an admissible initial state.

A permanent Hodge-skew diagnostic has now been added in
`test/Advection/test_vector_invariant_centered_ohpsg_energy.jl`. It constructs
an `N=4` projected random OHPSG state and computes the Hodge-weighted work of
the full centered VI tendency. The divergence guard passes with
`max|D u|=2.0577983761427276e-13`, while the energy-skew assertion is
intentionally `@test_broken`: the current total Hodge work is
`3.2157493806338586e-7`, and the largest column defect is at the west seam
column `i=1`. A broader `/tmp` probe over `N ∈ (4, 8, 16)` also found `i=1` as
the dominant column RMS at all three resolutions. This quantifies the remaining
spatial/topological stencil defect independently of long time integration.

A term-split Hodge-work probe further localizes the defect. On projected random
states, the residual between full VI work and rotational-plus-Bernoulli work is
roundoff (`O(10^-17)`), so vertical coupling is not the source. The defect is a
failed cancellation between `covariant_rotational_advection_*` and
`covariant_bernoulli_head_*`; at `N=16`, seed 42, the full work is
`9.063212e-7`, rotational work is `-1.213183e-6`, Bernoulli work is
`2.119504e-6`, and the largest Bernoulli column defect is at the west seam
`i=1`.

Therefore the next production implementation target is the centered VI seam
stencil itself: the rotational-advection / Bernoulli-gradient coupling must be
re-derived so the Hodge-weighted work of the full projected centered VI update is
skew-symmetric to roundoff on admissible OHPSG states.

## Immediate acceptance fork

### A. Polar-stiffness hypothesis confirmed

If a CFL scan shows that sufficiently small metric-aware time steps stabilize
the admissible divergence-free centered VI dynamics, then:

- Audit `cell_advection_timescale` / simulation timestep selection for OHPSG.
- Ensure the stable timestep is based on the actual contravariant polar-row
  rates and metric/Hodge factors, not covariant velocity amplitudes or bulk
  cell widths.
- Add a targeted test showing the recommended OHPSG timestep is dominated by
  the polar row when appropriate.
- Re-run centered no-mask dynamics to the milestone target with the corrected
  timestep criterion.

### B. Polar-stiffness hypothesis rejected

This is the active branch for the current random-vortex gate.

If arbitrarily small practical CFL still fails, or if failure rate does not
scale with `Δt`, then:

- Treat the remaining problem as a spatial/topological operator bug.
- Localize the positive-growth mode by row, face type, term, and seam/fold
  category.
- Re-derive the polar-row `u` and `v` updates, including duplicate/folded face
  topology and Bernoulli/vorticity adjointness.
- Fix the operator without suppressing prognostic degrees of freedom.

## Work plan

1. Add/keep the independent-face Hodge positivity and weighted-adjointness test
   as a Tier-0 guard. This now passes at `N=4` for `Float32` and `Float64`.
2. Add/keep the Hodge-weighted divergence-free projection algebraic test. This
   now passes at `N=4` and is the reference formula for the rigid-lid pressure
   projection.
3. Implement a production sparse/PCG projection equivalent to
   `K^{-1} D^T (D K^{-1} D^T)^{-1} D`, avoiding dense matrices and preserving
   GPU-compatible kernels. The first green production-kernel test must compare
   both the exact `B_tilde` stencil and the pressure-correction vector
   `W^{-1}B_tilde^T p` against dense references; matching only the scalar Schur
   complement is not sufficient.
4. Insert the projection after horizontal velocity prediction in AB2 and RK
   timesteppers for `SphericalShellGrid + free_surface=nothing`, then refill
   paired vector halos and recompute vertical velocity/transport consistently.
5. Replace the broken incompressibility-preservation test with a passing test
   that the projected centered VI tendency/update preserves `D u = 0` to solver
   tolerance.
6. Re-run the centered random-vortex gate and a resolution sweep only after the
   algebraic projection test and update-level incompressibility tests are green.
7. Keep centered VI acceptance separate from WENO VI. WENO VI must not be
   claimed complete until the WENO vorticity and Bernoulli/KE-gradient pieces
   are independently non-orthogonal and seam-correct.
8. Update `HANDOFF.md` after each diagnostic block with results and concrete
   tasks for the other agent.

## Milestone acceptance criteria

- Centered VI spatial KE identity passes on admissible OHPSG states without
  masks.
- Centered VI no-mask dynamics has a quantitatively justified stable timestep
  criterion and reaches the agreed milestone without polar/seam blow-up.
- The implementation contains no strip-zero masks or empirical damping.
- The documentation/tests state exactly what is validated for centered VI and
  what remains unresolved for WENO VI.

## Acceptance test plan

The authoritative must-pass acceptance plan is in `NONORTHOGONAL_VI_TEST_PLAN.md`.
Implementation claims must use the claim definitions in that file:

- Claim A: centered non-orthogonal `VectorInvariant` correctness.
- Claim B: fallback OHPSG `WENOVectorInvariant` stability.
- Claim C: true non-orthogonal `WENOVectorInvariant` correctness.
- Claim D: complete HFSM non-orthogonal VI support.
- Claim E: complete Breeze compressible non-orthogonal VI support.

No implementation should be called complete based only on a short turbulence run.

## 2026-06-03 update: first Q-form/Hodge-Bernoulli production attempt

The first production attempt at the active branch has been implemented for
`OHPSG` in `src/Operators/nonorthogonal_metric_operators.jl`:

- rotational advection uses a centered Q-form built from Hodge momenta
  `K u = hodge_weight * covariant_to_contravariant_flux` at FF corners;
- the Q-form rotational tendency is scaled back by the local face Hodge weight;
- Bernoulli head uses a Hodge-compatible kinetic-energy gradient with the same
  independent-face boundary adjoint structure as the rigid-lid pressure
  correction.

This is not a complete Claim A fix yet, but it is the first source change that
suppresses the known N32 west/east seam mode through the old failure window
without masks or damping.

Current validation evidence:

- `test/Advection/test_vector_invariant_centered_ohpsg_energy.jl` remains
  intentionally broken on the global Hodge-skew assertion, but improves the
  total Hodge work defect from `+3.2157493806338586e-7` to
  `-1.6043728259597716e-7` for the `N=4` projected random state.
- `test/Advection/test_vector_invariant_centered_ohpsg_incompressibility.jl`
  still passes projected-update divergence with
  `maximum_projected_update_divergence = 5.301314942585122e-13`; the raw
  tendency divergence is larger (`0.1827465130773747`) and remains
  deliberately `@test_broken`.
- The `N=16`, 180-step short random-vortex gate passes with max divergence
  `3.7051427464324105e-12`.
- A fresh `N=32`, 390-step low-allocation replay under the scaled source passed
  through the old failure window. At step 375, `t = 3.522180611249984`,
  `maxu = 0.0724542365024563` at `(32, 50)`, `maxv = 0.057814258433474985` at
  `(42, 64)`, seam max `u = 0.022727965104191644`, seam max
  `v = 0.057814258433474985`, and max divergence
  `1.001292798674669e-12`.

Remaining caveats:

- The Q-form/Hodge-Bernoulli dynamics are much flatter than the prior centered
  VI baseline, so the physical scaling/sign is not yet settled.
- The formal `N=32`, 533-step gate to `t≈5` has not been rerun under this path.
- The Hodge-skew diagnostic is improved but not fixed to roundoff; local column
  work transfers are still large.

Next work should focus on deriving the correctly scaled Hodge-compatible
Q-form/Bernoulli pair, then rerunning the formal `N=32` gate only after the
Hodge-skew diagnostic and short dynamics gate improve together.

## 2026-06-03 update: Q-form/Hodge-Bernoulli attempt reverted

The first production Q-form/Hodge-Bernoulli attempt described above was tested
further and reverted from source. It did suppress the N32 seam mode through the
old failure window, but it did so by introducing a severe scaling mismatch:

- the rotational tendency became under-scaled relative to the old centered VI
  path (`N=16` random-vortex max `5.799350e-03` versus old centered rotational
  max `4.754795e-01`);
- the Hodge-compatible Bernoulli tendency became over-scaled (`N=16`
  random-vortex max `2.972261e+02` versus old centered Bernoulli max
  `2.881482e-01`);
- the resulting dynamics were artificially flat because the large gradient-like
  Bernoulli contribution was mostly removed by the rigid-lid projection while
  the remaining rotational update was too weak.

Two prototype rescalings were also rejected before further source edits:

- a shared corner `hodge_weight` denominator was equivalent to the already
  rejected scaled source path on this grid;
- a shared corner `hodge_weight * J` denominator restored old rotational
  magnitude but also restored a large Hodge-work defect (`N=16` projected
  random-state rotational work `-1.510e-6`), so it is not energy-compatible.

The current source is therefore back to the smaller-defect centered VI baseline
plus the durable rigid-lid projection work and the permanent Hodge-skew
regression diagnostic. The next source edit should be preceded by a bilinear
matrix/symmetric-part diagnostic at `N=4` that identifies the exact stencil
entries responsible for the remaining small symmetric form.

## 2026-06-03 update: bilinear symmetric-part localization

A dense `N=4` bilinear symmetric-part diagnostic was built in
`/tmp/ohpsg_vi_bilinear_symmetric_probe.jl` and split by term in
`/tmp/ohpsg_vi_bilinear_term_split_probe.jl`. The diagnostic finite-differences
the centered VI tendency around a projected random state, restricts perturbation
directions to the dense nullspace of the horizontal divergence matrix, and forms
the symmetric part of the constrained Hodge Jacobian.

The dominant symmetric entries are seam/fold couplings. On the current baseline
source, the largest full entry is the south-fold/west-seam pair
`v(8,1) ↔ u(1,1)` at `N=4`, with normalized symmetric value
`9.284039e-02` and raw value `1.281918e-03`. Term splitting shows this pattern
is dominated by the Bernoulli/KE-gradient Jacobian:

- full max normalized symmetric entry: `9.284039e-02`;
- rotational max normalized symmetric entry: `3.164128e-02`, with a different
  top pattern (`u(8,5) ↔ u(1,4)`);
- Bernoulli max normalized symmetric entry: `9.256639e-02`, with the same
  `v(8,1) ↔ u(1,1)` / south-row pattern as the full operator.

Therefore the next production stencil attempt should target KE/Bernoulli
placement across folded scalar/face topology, especially component-swapping
south-fold/west-seam entries, rather than broad rotational Q-form replacement.

### Rejected Bernoulli boundary-adjoint/width experiment

A bounded source experiment modified only the OHPSG Bernoulli head to reuse the rigid-lid
pressure-correction boundary topology for the kinetic-energy difference, while still dividing
by the usual computational width rather than by the Hodge weight. This was intended to avoid
the previously rejected oversized Hodge-weight-scaled Bernoulli gradients.

The permanent Hodge-work diagnostic worsened:

- baseline before the experiment: total Hodge work `+3.2157493806338586e-7`, worst column `i=1`, maximum column work `7.706298540891233e-7`
- experiment: total Hodge work `-1.7061524302295935e-6`, worst column `i=1`, maximum column work `1.249746629540033e-6`, maximum divergence `2.0577983761427276e-13`

The source experiment was reverted. This rules out a simple reuse of the projection boundary
pressure correction as a width-scaled Bernoulli-gradient topology fix.

### Boundary-correction split probe

A temporary `/tmp` probe preserved the existing Bernoulli scalar halo gradient and added a tunable
boundary kinetic-energy correction instead of replacing the raw gradient. This separated the effect
of the source experiment's raw-gradient replacement from the boundary correction itself.

Symmetric all-component coefficient scan:

- N=4 baseline total `+3.2157493807517376e-7`; `α=1` reduces total to `+3.902207887425268e-8`; `α=2` gives `-2.4353078032666824e-7`
- N=8 baseline total `+1.6884873066849135e-6`; symmetric `α=2` gives `+1.104861265262827e-6`; best small grid result was `αu=-2`, `αv=2`, total `+4.2654951963984855e-7`
- N=16 baseline total `+9.063212207903408e-7`; symmetric `α=2` gives `-1.0128563770069836e-7`, but the maximum-column defect shifts to `i=31`

A second split probe decomposed the boundary correction by source boundary and target component.
The north-boundary channels are negligible; the active correction channels are east x-face boundary
component-swapping terms:

- N=4: baseline `+3.2157493807517397e-7`; `u_x=+2.3513976180962697e-7`, `v_x=-5.227565203832186e-7`, `u_y=+6.902207537112877e-9`, `v_y=-1.8383081644423082e-9`
- N=8: baseline `+1.688487306684913e-6`; `u_x=+1.6952514916274493e-7`, `v_x=-4.5400308539157945e-7`, `u_y=+5.278724299970915e-11`, `v_y=-7.3878717252085436e-9`
- N=16: baseline `+9.063212207903414e-7`; `u_x=+9.01500792245866e-8`, `v_x=-5.949174664202025e-7`, `u_y=-9.888162628528254e-12`, `v_y=+9.738461127244605e-10`

This supports the bilinear symmetric-part result: the remaining defect is an east-seam component
swap in the Bernoulli/kinetic-energy topology, not a north-boundary scalar-gradient problem. A
coefficient-tuned correction is not justified because the best coefficient changes with resolution
and component.

### Polar cross-metric and scalar-KE ghost diagnostics

The reviewer hypothesis that the dominant `v(Nx,1) ↔ u(1,1)` Bernoulli symmetric entry comes from a
nonzero polar-row `G¹²` cross-metric term was tested in `/tmp/ke_polar_cross_metric_probe.jl` and
`/tmp/metric_cross_row_probe.jl`.

Results:

- Replacing the Bernoulli KE with a candidate that zeros `G¹²ᶠᶜᶜ` at the south row, both polar rows,
  or a broader south cross-term variant produced no measurable change in the N=4 projected-state
  Hodge work or the projected `u(1,1)` / `v(Nx,1)` pair diagnostic.
- The actual metric values explain why: `G¹²ᶠᶜᶜ` is already exactly zero at rows `j=0,1,2,7,8,9` for
  `OctaHEALPixMapping(4)`, and `G²¹ᶜᶠᶜ` is zero or roundoff (`O(10^-16)`) there.

The alternative unilateral-truncation probe `/tmp/u_bernoulli_zero_probe.jl` zeroed `u` Bernoulli on
polar rows without changing source. This is not a valid production fix because it suppresses a
prognostic tendency row, and it was not robust anyway:

- N=4: source total `+3.2157493807517376e-7`; zero south-row `u` Bernoulli total
  `+9.632099860284121e-8`; zero both polar rows total `+2.7155948845588184e-7`.
- N=8: source total `+1.6884873066849135e-6`; zero south-row `u` Bernoulli total
  `+1.746039072736927e-6`; zero both polar rows total `+1.2672100851353975e-6`.
- N=16: source total `+9.063212207903408e-7`; zero south-row `u` Bernoulli total
  `+1.177319660472888e-6`; zero both polar rows total `+1.1159674624355023e-6` with the max-column
  defect shifted to `i=31`.

The stronger finding is that function-valued KE is not scalar-continuous in OHPSG ghost cells when it
is recomputed from rotated vector halos. For the N=4 projected random state:

- `KE(0,1)` recomputed from vector halos is `0.0001663317748607441`, but the scalar source
  `KE(8,1)` is `0.0004421271635209623`.
- `KE(9,1)` recomputed from vector halos is `0.0002868594348596006`, but the scalar source
  `KE(1,1)` is `3.604295775144059e-6`.
- Similar mismatches appear on other west/east ghost rows.

A broad scalar-continuous KE ghost replacement in `/tmp/scalar_continuous_ke_probe.jl` is also not a
complete fix:

- N=4: source total `+3.2157493807517376e-7`; scalar-continuous KE total
  `-9.094600291380093e-7`, max column `4.604051131241139e-7` at `i=1`.
- N=8: source total `+1.6884873066849135e-6`; scalar-continuous KE total
  `+5.0542619340037445e-8`, max column `9.314830683735087e-7` at `i=4`.
- N=16: source total `+9.063212207903408e-7`; scalar-continuous KE total
  `-1.1921968155167485e-6`, max column `1.3309928437051154e-6` at `i=31`.

Interpretation: the scalar-vs-vector KE ghost mismatch is real and aligns with the Bernoulli seam
problem, but replacing all ghost KE evaluations by scalar-source KE overcorrects and shifts the
energy defect. The next diagnostic should split scalar-continuous KE by west/east seam row or by the
specific x-face component-swap map, but the first row-split script was too slow and was abandoned.

### Abandoned x-ghost split runtime probe

A narrower `/tmp/scalar_ke_xghost_split_probe.jl` was prepared to test only west/east x-face scalar
KE ghost replacements in `u` Bernoulli, split by row bands. It did not produce a baseline line after
several minutes, and sandbox process-control restrictions prevented interrupting or killing the
non-tty Julia session. No source files were modified and no mathematical conclusion should be drawn
from this aborted run.

The next retry should avoid `update_state!`/CG setup by constructing the N=4 projected state with a
dense local projection, or should run under an explicit timeout/TTY. The intended first split remains
`i=0, j=1` scalar KE feeding the `u(1,1)` Bernoulli gradient.

### Dense-projection x-ghost split retry did not produce output

A retry script `/tmp/scalar_ke_xghost_dense_projected_probe.jl` was prepared to avoid
`update_state!`/CG by constructing an N=4 dense projection from the local `D` and `K` matrices, then
splitting only west x-ghost scalar KE replacements in `u` Bernoulli. It was launched single-threaded
under a Perl alarm but exited after the alarm with no output. This should be treated as a runtime
resource/process-control failure, not mathematical evidence. No source files were modified.

After stale Julia diagnostics are cleared, rerun that script before attempting a production stencil
edit. The target remains the west x-ghost scalar KE mismatch, especially `KE(0,1)` feeding
`u(1,1)` Bernoulli and coupling to `v(Nx,1)`.

### Scalar KE ghost split results

The earlier row-split and x-ghost split probes eventually produced useful N=4/N=8 data.

`/tmp/scalar_ke_row_split_probe.jl` applies scalar-continuous KE to all ghost cells on one selected
`j` row. N=4 baseline total was `+3.2157493807517397e-7`, with max column
`7.706298540890692e-7` at `i=1`. The largest N=4 row effect was `row=6`, which changed the total to
`-6.089041565842846e-7` and reduced max column to `1.7914794128188289e-7` at `i=7`. South rows were
small: `row=1` delta `-1.908526907296658e-8`, `row=2` delta `+1.6967225453879234e-8`.

At N=8, baseline total was `+1.688487306684913e-6`, max column `1.7027672557026597e-6` at `i=1`.
The largest row effects were high rows: `row=13` delta `-6.399756897551271e-7`, `row=16` delta
`-4.764168815260129e-7`, `row=9` delta `-2.3236937103030986e-7`, and `row=14` delta
`-1.8970174712005818e-7`. Again, south rows were small (`row=1` delta `+8.596187822010796e-9`,
`row=2` delta `+6.087027074238506e-9`).

`/tmp/scalar_ke_xghost_split_probe.jl` changed only west/east x-face scalar KE ghost values used by
`u` Bernoulli at N=4. East x-ghost variants were exact no-ops, as expected because interior
`u(i,j)` gradients use `KE(i,j)-KE(i-1,j)` and only `KE(0,j)` feeds west column `i=1`.

N=4 x-ghost results:

- baseline total `+3.215749380751741e-7`, Bernoulli work `+1.1591915698302406e-6`, max column
  `7.706298540890692e-7` at `i=1`.
- `west_row1` total `+3.024896690022075e-7`, delta `-1.908526907296658e-8`.
- `west_row2` total `+3.385421635290533e-7`, delta `+1.6967225453879234e-8`.
- `west_south_band` total `+3.1945689445608715e-7`, delta `-2.1180436190869215e-9`.
- `west_north_band` total `+1.5051254371330752e-7`, delta `-1.7106239436186656e-7`, max column
  `5.995674597272025e-7` at `i=1`.
- `west_all` total `-9.094600291380095e-7`, delta `-1.2310349672131836e-6`, Bernoulli work
  `-7.184339738294325e-8`, max column `4.604051131241139e-7` at `i=1`.

Conclusion: the actionable scalar KE ghost defect for `u` Bernoulli is west x-ghost topology, but it
is not a south-row-only effect. A blanket west scalar remap overcorrects, while north/high-row bands
account for most of the correction in the current random projected states. The next step is to map
these rows to OHPSG ring/quadrant/seam categories and derive a topology-specific scalar KE
continuation rather than tuning row coefficients.

### Additional row/topology split evidence

The row-split run completed N=16. Baseline was `+9.063212207903414e-7`, max column
`1.357967056045507e-6` at `i=1`. The strongest row was `row=7`, which changed total work to
`-1.089307111461367e-6` (`delta=-1.9956283322517084e-6`) and shifted the max row to `j=7`.
Other notable N=16 row deltas were `row=6` `+2.318853378720522e-7`, `row=25`
`+2.245014368855818e-7`, `row=32` `-1.9819728329283703e-7`, `row=8`
`-1.5418068301656972e-7`, and `row=27` `-1.1745795930461105e-7`.

An N=4 west x-ghost per-row split confirmed the row effects are directly from the x-ghost KE feeding
`u` Bernoulli. The dominant N=4 row was `row=6`, with delta `-9.304790946594586e-7`, rotation 2,
source quadrant 3/current quadrant 1, component kind 1, sign -1. Secondary N=4 rows were `row=4`
(delta `-1.6702992941610016e-7`, rotation 3, kind 2, sign -1) and `row=7` (delta
`-1.4680740205887427e-7`, rotation 1, kind 2, sign +1).

Key N=16 topology rows show mixed categories: row 7 is rotation 1/component-swap/sign +1, row 6 is
rotation 0/kind 1/sign +1, row 25 is rotation 1/component-swap/sign +1, and row 32 is rotation
3/component-swap/sign -1. Thus the scalar KE ghost problem is not a single polar row, rotation class,
or component-kind class.

The next derivation should treat KE as a scalar under the full local metric/vector transform at ghost
centers. The missing piece is likely a metric-transformed center-ghost KE evaluation, not a manual row
selection or tuned scalar continuation.

### Center-metric ghost KE and Hodge-adjoint Bernoulli probes

Two more derived candidates were tested and rejected before source edits.

`/tmp/center_metric_west_ghost_ke_probe.jl` computed west ghost KE from center-interpolated covariant
components and the center inverse metric `Gᶜᶜ`, using it only in west x-ghost KE for `u` Bernoulli.
It did not recover scalar continuity. At N=4, row 6 current ghost KE was
`1.7906851667461618e-3`, scalar-source KE was `4.360363626530197e-4`, and center-metric KE was
`1.6135887458952562e-4`. The Hodge-work total worsened/sign-flipped from the source
`+3.215749380751741e-7` to `-9.940929515664377e-7`.

`/tmp/hodge_adjoint_bernoulli_probe.jl` replaced the Bernoulli term with a direct
Hodge-adjoint-style `B_tildeᵀ KE / hodge_weight` candidate. Both signs were rejected:

- source total `+3.215749380751741e-7`, Bernoulli `+1.1591915698302406e-6`, max column
  `7.706298540890692e-7` at `i=1`;
- hodge-adjoint plus total `+8.009216365212047e-6`, Bernoulli `+8.846832996967119e-6`, max column
  `1.6070253940054308e-5` at `i=8`;
- hodge-adjoint minus total `-9.68444962872218e-6`, Bernoulli `-8.846832996967119e-6`, max column
  `1.338441926434455e-5` at `i=8`.

Thus neither local center-metric ghost KE nor wholesale projection-adjoint Bernoulli topology is the
missing fix. The remaining lead is more specific: identify why the existing face-averaged KE at west
x-ghost centers is not scalar-continuous, but repair only the inconsistent face/metric contribution
rather than replacing the full Bernoulli stencil.

### West x-ghost KE contribution decomposition

A N=4 contribution probe decomposed west ghost KE into its face contributions. The scalar-continuity
mismatch is dominated by the `v`-face contribution in
`ℑyᵃᶜᵃ(covariant_kinetic_energy_vᶜᶠᶜ)`, not by the `u`-face contribution.

Key rows:

- row 4: total ghost/source KE diff `+4.579104652003722e-3`, of which the v contribution is
  `+4.587822346200861e-3` and the u contribution is only `-8.717694197138092e-6`.
- row 6: total diff `+1.354648804093142e-3`, all from the v contribution; u diff is exactly `0.0`.
- row 7: total diff `+1.0002525036754753e-3`, with v diff `+1.3833752779937342e-3` partly offset by
  u diff `-3.831227743182589e-4`.

Substituting only the scalar-source v contribution in all west ghost rows changed the N=4 Hodge work
from `+3.215749380751741e-7` to `-8.678219813774817e-7`, while substituting only the u contribution
changed it only to `+2.799368903146466e-7`. Row-limited v-contribution substitution showed row 6 is
the dominant correction (`delta=-9.304790946594586e-7`, total `-6.089041565842844e-7`), with rows 4
and 7 secondary useful corrections.

This is a sharper target: repair the scalar continuity of the `v`-face contribution to KE at west
ghost centers. The next derivation should inspect the two `v` faces sampled by `ℑy` at those west
ghost centers and how `covariant_to_contravariant_velocity_vᶜᶠᶜ` handles polar folds and
component-swapped halos there. Row-specific patching remains rejected.

### V-face sample split and component-aware source-face candidate

The dominant `v` contribution in west x-ghost KE was split into the two `v` faces sampled by
`ℑyᵃᶜᵃ(0,j)`. Upper samples dominate the useful change: replacing all upper samples changed N=4 total
work from `+3.215749380751741e-7` to `-7.78141303007323e-7` and reduced max-column work from
`7.706298540890692e-7` to `3.290863869934279e-7`. Lower samples only changed total to
`+2.318942597050153e-7` and max-column work to `6.809491757189107e-7`.

The natural topology unit is a shared west ghost `v` face. Replacing `face_j=7` alone changed the
N=4 total by `-1.1293013629370274e-6`, giving total `-8.077264248618533e-7` and max-column work
`3.5867150884795795e-7`. The factor probe showed this face is component-swapped: `v(0,7)` maps from
source `u(8,7)` with sign `-1`; the inflated KE is due to the component-swapped covariant value, not
metric magnitude.

A component-aware source-face KE candidate was tested: for west ghost `v` faces, use the covariant
vector halo source kind to choose source `u`-face KE or source `v`-face KE. This is more derived than
plain scalar-source `v`-face KE.

Resolution sweep:

- N=4: source total `+3.215749380751741e-7`, candidate total `-6.457971343868861e-7`; max-column work
  improved from `7.706298540890692e-7` to `1.967422183729908e-7`.
- N=8: source total `+1.688487306684912e-6`, candidate total `+6.241839585232821e-7`; max-column work
  improved from `1.7027672557026595e-6` to `9.314830683735086e-7`.
- N=16: source total `+9.063212207903414e-7`, candidate total `-5.03589322113899e-7`; max-column work
  barely changed, from `1.3579670560455076e-6` to `1.330992843705115e-6`, and moved to `i=31`.

Conclusion: component-aware source-face KE is a strong local diagnostic candidate but not a full
production fix. It likely fixes one side of the duplicated seam while exposing the opposite side.
Next work should either apply a symmetric duplicated-seam treatment in a diagnostic, or rerun the N=4
bilinear symmetric-part probe with this candidate to see which top entries remain.

### Symmetric boundary-center component-aware KE candidate

A symmetric boundary-center diagnostic was tested in
`/tmp/component_aware_boundary_center_ke_probe.jl`. It applies component-aware source-face KE at both
low-side and high-side out-of-domain face samples used by `u` Bernoulli center KE: center `i=0` and
center `i=Nx`. Interior center KE, rotational advection, and `v` Bernoulli are unchanged.

Results:

- N=4: total improved from `+3.215749380751741e-7` to `-1.91028517338446e-7`; max-column work
  improved from `7.706298540890692e-7` at `i=1` to `3.4027716752699316e-7` at `i=8`.
- N=8: total improved from `+1.688487306684912e-6` to `+1.8989438582621636e-7`; max-column work
  improved from `1.7027672557026595e-6` to `9.314830683735086e-7`.
- N=16: total changed from `+9.063212207903414e-7` to `-4.4894999884931067e-7`, but max-column work
  barely improved, from `1.3579670560455076e-6` to `1.330992843705115e-6`, and moved to `i=31`.

Conclusion: this is an improved but insufficient diagnostic candidate. It is not source-ready. The
remaining defect likely involves either `v` Bernoulli/y-boundary KE samples or a different symmetric
entry exposed after the `u`-Bernoulli x-boundary repair. Next step: bilinear symmetric-part
localization with this candidate, or a similarly derived component-aware treatment for `v` Bernoulli.

### Combined component-aware x/y-boundary KE candidate and pair check

A combined component-aware candidate was tested in `/tmp/component_aware_xy_boundary_ke_probe.jl`.
It uses component-aware source-face KE for out-of-domain x faces and polar y faces in center KE, then
uses that KE in both `u` and `v` Bernoulli. The candidate improves N4/N8 but remains insufficient:

- N=4: total `+3.215749380751741e-7` -> `-1.9400984527372568e-7`, max-column work
  `7.706298540890692e-7` -> `3.254863553664863e-7`.
- N=8: total `+1.688487306684912e-6` -> `+1.571978898804624e-7`, max-column work
  `1.7027672557026595e-6` -> `8.127374074764226e-7`.
- N=16: total `+9.063212207903414e-7` -> `-6.710826834934751e-7`, max-column work only improves
  `1.3579670560455076e-6` -> `1.2062864473802514e-6`, and shifts to `i=27`.

A selective N=4 bilinear check of the original top channel `u(1,1)` / `v(Nx,1)` shows the candidate
strongly suppresses that pair: normalized value drops from `+4.471697117326832e-3` to
`+1.690363241472014e-4`. Therefore the candidate repairs the originally localized channel but leaves
other symmetric entries. The next diagnostic should be a full N=4 bilinear symmetric-part
localization with this component-aware candidate to identify the remaining top entries.

### Full bilinear localization with combined component-aware candidate

A full N=4 bilinear projected-label localization was run with the combined component-aware candidate.
It confirmed why the candidate is not source-ready.

Under this projected-label ranking, the combined candidate suppresses the original south/west pair but
leaves or creates dominant west/east seam `u` entries. The most important entries are:

- candidate `u(1,5) x u(1,5)`: normalized `+1.119246e-01`;
- candidate `u(1,4) x u(1,4)`: normalized `-1.080923e-01`;
- candidate `u(1,5) x u(8,5)`: normalized `+9.046453e-02`;
- candidate `u(1,4) x u(8,5)`: normalized `+8.872365e-02`.

A focused term split shows the issue: the candidate destroys an existing Bernoulli cancellation of a
large rotational cross-seam symmetric part.

- Source `u(1,5) x u(8,5)`: full `+3.317862e-03`, rotational `+1.045856e-01`, Bernoulli
  `-1.012678e-01`.
- Candidate `u(1,5) x u(8,5)`: full `+9.046453e-02`, rotational `+1.045856e-01`, Bernoulli
  `-1.412112e-02`.
- Source `u(1,4) x u(8,5)`: full `+1.783694e-03`, rotational `+1.006942e-01`, Bernoulli
  `-9.891047e-02`.
- Candidate `u(1,4) x u(8,5)`: full `+8.872365e-02`, rotational `+1.006942e-01`, Bernoulli
  `-1.197052e-02`.

Thus the component-aware candidate is too broad. It repairs one localized component-swapped KE
channel but removes necessary Bernoulli cancellation for mid-row cross-seam `u` modes. Future
candidates must be checked against these bilinear entries, not only against global Hodge work.

### Narrow v-face-only component-aware variants rejected

A sweep of narrower `v`-face-only component-aware KE variants was run in
`/tmp/narrow_vface_variant_sweep.jl`. These variants leave the `u`-face KE contribution untouched and
select only polar y faces or component-swapped out-of-domain x y-faces.

All variants are rejected as production fixes:

- `polar_y_only` is too weak: at N=16 it changes total work from `+9.063212207903414e-7` to
  `+9.125223263619937e-7` and max-column work from `1.3579670560455076e-6` to
  `1.296292008102179e-6`.
- `x_kind1_negative` improves N4/N8 max-column work but at N=16 gives total
  `-6.54251241199241e-7` and max-column `1.330992843705115e-6` at `i=31`.
- `x_kind1` similarly fails N16 with total `-5.069520208210091e-7` and max-column
  `1.330992843705115e-6` at `i=31`.
- `x_kind1_negative_plus_polar` gives the best N16 max-column among these (`1.2062864473802514e-6`)
  but still shifts the defect to `i=27` and flips total work negative.

The conclusion from the bilinear guard still stands: source-kind/sign selection fixes some local
component-swapped KE channels but does not preserve the broader Bernoulli/rotational cancellation.
The next useful step is not another row/sign selector; it is deriving the missing cancellation with
rotational advection or constructing a candidate that explicitly preserves the mid-row bilinear guard
pairs while fixing the original south/west pair.

### 2026-06-03: Bernoulli delta face/sample decomposition

- Added `/tmp/bernoulli_delta_face_decomp.jl` to decompose the N=4 broad component-aware candidate-minus-source Bernoulli bilinear delta by tendency face using a linear Hodge matrix.
- Result: original seam pair `u11-vN1` improves from source full `+4.471697e-03` to candidate full `+1.690363e-04`, dominated by `u(1,1)=-4.570112e-03`.
- Guard pairs are destroyed by the same broad replacement: `u15-uN5` source full `+3.317862e-03` becomes `+9.046453e-02`, dominated by `u(1,5)=+9.177938e-02`; `u14-uN5` source full `+1.783694e-03` becomes `+8.872365e-02`, dominated by `u(1,4)=+9.167939e-02`.
- Added `/tmp/west_center_sample_decomp.jl` to split those west-center deltas into individual KE samples. The useful original-pair correction is west center row 1 `u_left=-4.239975e-03` plus `v_lower=-3.314206e-04`; the damaging guard-pair corrections are the shared west ghost `v`-face `(0,5)`: row 5 `v_lower=+9.230471e-02` for `u15-uN5` and row 4 `v_upper=+9.235001e-02` for `u14-uN5`.
- Added `/tmp/xface_u_only_ke_probe.jl` to test the narrower hypothesis of replacing only out-of-domain `x`-ghost `u`-face samples. It is mixed and not source-ready: N=4 total worsens from `3.215749380703277e-7` to `7.654975397261744e-7`; N=8 improves to `1.2697060601852193e-6`; N=16 improves to `7.565989544726359e-7` but shifts max column to `i=31` with maxcol `1.3309928436987071e-6`.
- Current conclusion: blanket component-aware source-face KE replacement is rejected because ghost `v`-face substitutions break existing Bernoulli/rotational cancellation. A viable candidate must correct the polar/seam `u`-face defect while preserving guard-pair cancellation for west ghost `v`-faces such as `(0,5)`.

### 2026-06-03: x-ghost source-kind/sign sweep

- Added `/tmp/ghost_sample_mapping_probe.jl`: the useful west `xface(0,1)` sample is `kind=2`, `source=(8,1)`, `sign=-1`; the damaging guard-pair `yface(0,5)` sample is `kind=1`, `source=(8,5)`, `sign=1`.
- Added `/tmp/xface_u_kind_sweep.jl`: restricting out-of-domain `x`-ghost `u`-face substitutions to `kind=2, sign=-1` is better than replacing all `x` ghosts.
- `x_u_kind2_negative` results: N=4 total `3.2177033491644443e-7` (source `3.215749380703277e-7`), N=8 total `1.1233865099261691e-6` (source `1.6884873066947022e-6`), N=16 total `3.2497759613394154e-7` (source `9.063212207888395e-7`) but N=16 max column remains `1.3309928436987071e-6` at `i=31`.
- `x_u_kind2_positive` is harmful: N=4 total `7.653021428800576e-7`, N=8 total `1.8348068569537537e-6`, N=16 total `1.3379425791275383e-6`.
- Current conclusion: a geometry-aware correction may involve only component-swapped negative-sign `x` ghost `u` faces. This is not sufficient for Claim A yet; the N=16 east/seam-side residual at `i=31` still has to be explained before any source edit.

### 2026-06-03: N16 side split for x_u_kind2_negative

- Added `/tmp/xface_kind2_negative_side_probe.jl`.
- Source N=16 top columns: `i=1` `+1.357967056046e-06`, `i=31` `-1.330992843699e-06`, `i=27` `-1.173397995353e-06`.
- West-only `kind=2, sign=-1` correction reduces `i=1` to `+8.445924968769e-07` and total to `+3.929466616202e-07`, but leaves `i=31` and `i=27` unchanged.
- East-only `kind=2, sign=-1` barely changes the top columns; both-side correction total is `+3.249775961339e-07` with max column still `i=31`.
- Positive-sign west component-swapped `x` ghosts are harmful: `west_kind2_positive` total `+1.229543309443e-06`, maxcol `+1.681189144700e-06` at `i=1`.
- Current conclusion: the west `x_u_kind2_negative` correction is a real partial mechanism, but the remaining N=16 defects at `i=31` and `i=27` need a separate localization.

### 2026-06-03 continuation: N16 residuals and adjoint structure

- Added `/tmp/n16_column_row_term_probe.jl`: the remaining N=16 residuals after the west `x_u_kind2_negative` mechanism are not out-of-domain halo cases. `i=31` total is `-1.330992843699e-06` with rotational `-1.569212322712e-06` and Bernoulli `+2.382194790136e-07`; `i=27` total is `-1.173397995353e-06` with rotational `-5.546301899146e-07` and Bernoulli `-6.187678054382e-07`.
- Added `/tmp/n16_residual_topology_probe.jl`: dominant `i=31`/`i=27` rows have zero immediate halo rotations and identity covariant maps. The exposed residual is an interior cancellation / folded-unique-face adjoint issue, not the same component-swapped ghost-KE pathology.
- Added `/tmp/rotational_corner_transport_probe.jl`: replacing corner contravariant velocities by corner-averaged fluxes makes rotational work much closer to Hodge-skew (`N=16` rot `-1.959621339838e-08`), and Hodge-flux corner transport nearly zeros rotational work (`N=16` rot `-1.202408594737e-10`), but total work worsens with current Bernoulli because Bernoulli remains `+2.119503941601e-06`.
- Added `/tmp/bernoulli_adjoint_identity_probe.jl`: direct Bernoulli work is not `K * div(volume flux)` or `K * div(Hodge covector)` over the unique faces; both are roundoff while direct Bernoulli work is `+1.159191569825e-06` at N=4, `+1.848069130523e-06` at N=8, and `+2.119503941601e-06` at N=16.
- Added `/tmp/bernoulli_work_face_region_probe.jl`: N=16 Bernoulli work splits into `u_left=+1.980617109261e-06`, `u_int=+4.862469645811e-06`, and `v_int=-4.723582813471e-06`; the dominant left row is `j=7`, `+1.476214259920e-06`.
- Added `/tmp/bernoulli_hodge_weight_scaling_probe.jl`: raw gradient divided by Hodge weight is rejected; it blows up total work to `+1.096981174125e-05` at N=4, `+7.513752852454e-05` at N=8, and `+3.442122551887e-04` at N=16.
- Added `/tmp/dense_exact_adjoint_bernoulli_probe.jl`: for N=4, exact global `B = H \\ (D' * K)` has Bernoulli work `-1.641211038600e-16`, but current rotational work remains `-8.376166317550e-07`. The exact adjoint vector is very different from current local Bernoulli (`norm_delta=0.17678181042905464`).
- Added `/tmp/dense_adjoint_plus_rot_variants_probe.jl`: exact adjoint Bernoulli plus flux-corner rotational gives total `-1.525638055830e-07`; exact adjoint plus Hodge-flux-corner rotational gives total `-1.497791673624e-08` at N=4. This is diagnostic only; Hodge-flux transport is likely dynamically under-scaled.
- Current conclusion: the baseline centered VI is partially cancelling two incompatible pieces: a non-adjoint local Bernoulli gradient and a non-skew centered rotational term. Ghost-KE tuning alone is not enough. The next principled direction is deriving a local, dimensionally correct approximation to `H^{-1}DᵀK` together with a Hodge-skew rotational corner transport.

### 2026-06-03 continuation 2: exact local Bernoulli adjoint identified

- Added `/tmp/adjoint_residual_covector_probe.jl`: current Bernoulli violates the adjoint identity in covector space. For projected fields, direct Bernoulli work equals `-xᵀ(HB_current - DᵀK)` to roundoff. N=16 has Bernoulli work `+2.119503941601e-06`, residual norm `2.686549092577e-01`, `DᵀK` norm `2.670442249386e-01`, and `HB_current` norm only `1.636180326448e-03`.
- Added `/tmp/diagonal_adjoint_bernoulli_probe.jl`: `B_diag = DᵀK / diag(H)` is exact for the tested projected states: residual through `H` is `4.644507460540e-19` at N=4, `3.417227767636e-18` at N=8, and `4.071173361963e-17` at N=16; Bernoulli work is roundoff.
- Added `/tmp/hodge_pressure_correction_equals_adjoint_probe.jl`: existing `hodge_compatible_pressure_correction_uᶠᶜᶜ/vᶜᶠᶜ(K)` equals `DᵀK / diag(H)` to roundoff. This is the exact local energy-adjoint Bernoulli target.
- Added `/tmp/hodge_compatible_rotational_flux_probe.jl`: hodge-compatible boundary wrapping does not change hodge-flux rotational work; N=4 `-1.497791657212e-08`, N=8 `-3.809987555704e-11`, N=16 `-1.202408594737e-10`.
- Added `/tmp/hodge_flux_rotational_polar_zero_probe.jl`: disabling polar-fold zeroing does not fix hodge-flux rotational residual.
- Added `/tmp/rotational_variant_norm_probe.jl`: flux/hodge-flux rotational forms are close to current after a scalar fit but differ by 5-7% in shape. Flux-fit scalars scale roughly like inverse cell area: N=4 `5.582749025787e+00`, N=8 `2.047595801451e+01`, N=16 `8.071851052196e+01`.
- Added `/tmp/common_corner_denominator_rotational_probe.jl`: common-corner denominator variants are scalar multiples of the same flux/hodge-flux shape and do not remove the 5-7% mismatch.
- Current conclusion: Bernoulli is solved diagnostically by using the existing hodge-compatible pressure correction on covariant kinetic energy. The remaining task is deriving a Hodge-skew rotational form with the correct metric-interpolation shape; simple boundary compatibility, polar-fold toggling, and denominator tuning are insufficient.

### 2026-06-03 continuation 3: metric corner denominators improve shape but not skew

- Added `/tmp/metric_corner_denominator_rotational_probe.jl`.
- Common corner denominators based on `hodge_weight * J` recover the current rotational operator shape much better than raw flux/hodge-flux variants: N=4 relative fit improves to `1.66e-02` to `1.91e-02`; N=8 to about `3.33e-02`; N=16 to about `5.08e-02`.
- However, scaling the hodge-flux form to physical magnitude amplifies the folded-boundary skew residual. Example: N=16 `sqrt_wJ` has work `-1.599902658320e-06`, comparable to current rotational work, despite a good shape fit.
- Current conclusion: the rotational fix likely requires transpose-consistent corner interpolation on the folded unique-face topology. Metric denominator tuning alone cannot prove Claim A.

### 2026-06-03 continuation 4: transpose replacement rejected; residual is small but distributed

- Added `/tmp/transpose_consistent_rotational_probe.jl`: exact rectangular incidence/transpose is Hodge-skew but dynamically poor. N=4 `mean_wJ` fit to current rotational is `8.184967674087e-01`; N=16 fit `3.401915534064e-01`.
- Added `/tmp/folded_vector_interpolation_transpose_rotational_probe.jl`: full two-component folded interpolation transpose using paired vector halo fills is also skew but still poor: N=4 `unit` fit `7.109884061904e-01`, N=16 `unit` fit `2.700957240674e-01`.
- Added `/tmp/functional_hodge_flux_transpose_rotational_probe.jl`: even a transpose map built through the actual hodge-flux functions is too disruptive: N=4 `unit` fit `5.615567150439e-01`, N=16 `unit` fit `3.481382421663e-01`.
- Added `/tmp/rotational_global_orthogonalization_probe.jl`: the good-shape metric-denominator rotational candidates need only a small global correction to zero Hodge work. For `sqrt_wJ`, relative correction is `1.58e-02` at N=4, `8.47e-04` at N=8, and `5.46e-03` at N=16; fit to current changes only slightly.
- Added `/tmp/metric_rotational_work_localization.jl`: the `sqrt_wJ` non-skew work residual is distributed across paired rows/columns rather than a single boundary defect. N=16 dominant columns are `i=13`, `i=31`, `i=7`, `i=1`; dominant rows are `j=19`, `j=7`, `j=18`, `j=20`.
- Current conclusion: wholesale exact-transpose replacement is rejected. The viable path is a local correction to the standard metric-denominator hodge-flux rotational form that removes its small non-skew component while preserving its good shape.

### 2026-06-03 continuation 5: exact-energy construction rejected dynamically

- Added `/tmp/boundary_corner_hybrid_rotational_probe.jl`: replacing x-boundary corner transport by interior-adjoint rectangular interpolation zeros rotational work but damages shape. N=16 `rect_x_boundary` has work `~0` but relative fit `3.500634224865e-01` versus standard metric-denominator fit `5.083761883024e-02`.
- Added `/tmp/exact_energy_candidate_sweep.jl`: combining `rect_x_boundary` rotational with `hodge_compatible_pressure_correction(K)` Bernoulli zeros total Hodge work to roundoff across N=4,8,16 and seeds 1,2,42, but the candidate tendency is dynamically unusable. For seed 42, relative fit to current VI is `9.999090339867e-01` at N=4, `9.959040909401e-01` at N=8, and `9.999407126741e-01` at N=16; candidate norms are orders of magnitude larger than current.
- Current conclusion: exact energy cancellation alone is the wrong acceptance criterion. The viable fix must preserve current centered VI dynamics while removing its small Hodge-work defect. Do not source-edit a direct `hodge_compatible_pressure_correction(K)` + rectangular boundary-corner rotational candidate.

### 2026-06-03 continuation 6: local support correction sweep

- Added `/tmp/local_state_aligned_correction_probe.jl`.
- A single all-face state-aligned correction zeros `sqrt_wJ` rotational work with small shape impact; seed 42 fits change N=4 `1.879041e-02 -> 2.469980e-02`, N=8 `3.393530e-02 -> 3.398081e-02`, N=16 `5.083762e-02 -> 5.084291e-02`.
- Restricting the same correction to x-boundary or top work columns/rows is moderately worse but still not catastrophic; however top supports vary with seed/resolution.
- Independent per-column/per-row cancellation is too destructive: seed 42 N=16 per-column fit `9.075681e-02`, per-row fit `1.718250e-01` versus base `5.083762e-02`.
- Current conclusion: do not pursue per-column/per-row local cancellation or fixed support selection. The non-skew component looks like a small global component of the good-shape rotational operator, not a robust localized defect.

### 2026-06-03 continuation 7: projected exact-energy candidate still rejected

- Added `/tmp/projected_tendency_comparison_probe.jl`.
- Dense Hodge-compatible projection reduces the exact-energy candidate's huge raw mismatch with current VI, but the projected mismatch remains too large: N=4 projected H-relative difference `3.416762737620e-01`; N=8 projected H-relative difference `4.173270199855e-01`.
- Projection itself is working: projected divergences are roundoff for both current and exact candidates.
- Current conclusion: the exact-energy candidate is not merely a pressure-gradient artifact. It remains dynamically too different after projection and is rejected.

### 2026-06-03 continuation 8: partial replacements identify Bernoulli as dynamic breaker

- Added `/tmp/projected_partial_variant_comparison.jl`.
- Replacing Bernoulli with exact `hodge_compatible_pressure_correction(K)` breaks projected dynamics: N=4 projected H-relative diff `1.517803e+00`, N=8 `3.909397e-01` when rotational is left current.
- Replacing only rotational with the `sqrt_wJ` metric-denominator form is dynamically close after projection: N=4 projected H-relative diff `3.626613e-02`, N=8 `3.834976e-02`.
- Added `/tmp/metric_rotational_blend_sweep.jl`: the scalar blend coefficient needed to zero work is not stable. It changes sign and magnitude across seeds/resolutions (for example N=8 seed 42 `-1.802989e+01`, N=16 seed 42 `+2.343611e+00`, N=16 seed 99 `-3.591466e+01`).
- Current conclusion: exact Bernoulli is dynamically unacceptable; metric rotational is dynamically plausible but not an energy fix. A fixed blend between current and metric rotational is rejected.

### 2026-06-03 continuation 9: partial exact-Bernoulli blending rejected

- Added `/tmp/partial_exact_bernoulli_tradeoff_probe.jl`.
- Blending current Bernoulli toward exact `hodge_compatible_pressure_correction(K)` cannot zero work while preserving projected dynamics. Zero-work fractions are unstable: N=4 seed 1 `θ=-248.46`, N=4 seed 42 `θ=0.277`, N=8 seed 2 `θ=-6.08`, N=8 seed 42 `θ=0.914`.
- Zero-work projected H-relative drift is too large: N=4 seed 42 `4.210585e-01`, N=8 seed 42 `3.571819e-01`, N=8 seed 2 `3.360017e+00`.
- Current conclusion: exact-Bernoulli replacement, full or partial, is rejected as a production direction.

### 2026-06-03 continuation 10: corner Hodge-flux reconstruction rejected

Tested `/tmp/corner_matrix_rotational_probe.jl`, which keeps current Bernoulli fixed and changes only the corner velocity reconstruction in the rotational flux.

- `sqrt_wJ` remains dynamically plausible but not energy-correct: seed42 projected H-relative drift is `3.626613e-02` at N=4 and `3.834976e-02` at N=8, with work `+2.829787886874e-07` and `+1.782136628057e-06`.
- Component-wise `J` denominators are similarly dynamic (`proj_Hrel 3.542494e-02` at N=4, `3.622863e-02` at N=8) but do not improve the work consistently.
- Component-wise or 2x2 `G`-metric inversions are rejected: seed42 projected H-relative drift is `1.547169e+00` at N=4 and `1.006852e+00` at N=8.
- Seed sweeps across N=4, 8, 16 show no stable work reduction for `component_J` or `G`-based inversions.

Status: no production source change. This rules out the direct local Hodge-matrix inversion hypothesis. Continue searching for a small local correction around current/sqrt_wJ rotational transport with current Bernoulli fixed.

### 2026-06-03 continuation 11: local corner work-balance correction rejected

Tested `/tmp/local_corner_balance_probe.jl`, a local correction that keeps current Bernoulli fixed and minimally adjusts `sqrt_wJ` corner velocities to cancel assigned corner work.

- N=4 seed42 projected H-relative drift is `6.043327e-01` for nonpolar correction and `6.178662e-01` when correcting polar-fold corners too.
- N=4 seed42 work changes from current `+3.215749380752e-07` to `-5.722406427033e-07` / `-6.612239025112e-07`; it does not solve the measured work defect.
- N=8 raw tendency drift is `5.684676e-01` to `7.415645e-01` across seeds 1, 2, 42, 99.
- N=16 raw tendency drift is `6.468463e-01` to `7.960135e-01`; N=16 seed99 has maximum local correction factor `2.204220e+02`.

Status: rejected. Pointwise corner work cancellation is too disruptive and the simple corner distribution of Bernoulli work is not the right OHPSG adjoint. Continue focusing on small local corrections that preserve the current/sqrt_wJ projected dynamics rather than enforcing local corner cancellation.

### 2026-06-03 continuation 12: polar corner zeroing rejected

Tested `/tmp/polar_corner_zero_toggle_probe.jl`, toggling the OHPSG polar-fold zeroing of contravariant corner transport velocities in the rotational flux while keeping current Bernoulli fixed.

- `zero_u` matches current and `zero_v` matches `no_zero`; only unzeroing the polar `u` corner velocity has an effect.
- N=4 unzeroing raw drift is `3.033398e-01` to `1.642183e+00`; work changes are inconsistent.
- N=8 unzeroing raw drift is `3.603073e-01` to `1.515903e+00`.
- N=16 unzeroing raw drift is `1.034607e+00` to `2.571184e+00`; seed42 work worsens from `+9.063212207903e-07` to `+9.642604138707e-07`.

Status: rejected. The polar-fold zeroing of contravariant corner velocity should not be removed for this fix.

### 2026-06-03 continuation 13: kinetic-energy gradient consistency check

Tested `/tmp/kinetic_energy_gradient_consistency_probe.jl` to compare the Hodge covector in the broken energy diagnostic with finite-difference gradients of the cell-centered kinetic energy.

- N=4 seed42: weighted cell-energy gradient differs from the Hodge covector by relative `9.890758e+00` but has correlation `9.912877e-01`. Work is `+3.212810552591e-06` against the weighted gradient and `+3.215749380752e-07` against the Hodge covector.
- N=8 seed42: weighted gradient relative difference is `3.949118e+01` with correlation `9.948807e-01`. Work is `+6.210809269135e-05` against the weighted gradient and `+1.688487306685e-06` against the Hodge covector.

Status: diagnostic only. The defect is not just a scalar normalization error in the Hodge-skew test, because the raw centered VI tendency has nonzero work against both the current kinetic-energy gradient and the independent-face Hodge covector. The current Bernoulli energy and Hodge energy are nevertheless distinct enough on OHPSG that future derivations must state which energy is being conserved.

### 2026-06-03 continuation 14: full-current mask localization

Tested `/tmp/full_current_mask_correction_probe.jl`, applying minimum state-aligned corrections on selected supports to cancel the full current centered VI Hodge work.

- N=4 seed42: all-state correction `rel_H=1.343768e-02`; west-only `5.019297e-02`; x-near columns `(1,2,Nx-1,Nx)` `1.785796e-02`; near-polar `1.912056e-02`.
- N=8 seed42: all-state `2.237716e-02`; west-only `9.300714e-02`; x-near `4.218020e-02`; top4 columns `4.354184e-02`.
- N=16 seed42: all-state `3.501444e-03`; west-only `2.195931e-02`; x-near `9.556370e-03`; top4 columns `9.857234e-03`.
- N=16 seed2 is harder: all-state `2.058265e-02`; west-only `1.197188e-01`; x-near `5.646572e-02`.

Status: diagnostic only. The defect is not cleanly isolated to one seam or polar support; it behaves like a distributed adjointness mismatch. State-aligned corrections remain nonphysical and are not source candidates.

### 2026-06-03 continuation 15: current Bernoulli scaling rejected

Tested `/tmp/current_bernoulli_scaling_probe.jl`, computing the scalar multiplier `α_zero` that would make current rotational plus `α * current Bernoulli` Hodge-skew for each projected random state.

- N=4 `α_zero`: `+2.494616716745e+02`, `-7.986906533363e-01`, `+7.225868903427e-01`, `+9.724912103472e-02`, `+4.206665436671e+00` for seeds 1, 2, 42, 99, 1234.
- N=8 `α_zero`: `-2.094926588073e+00`, `+7.079554367423e+00`, `+8.635057054464e-02`, `+1.584200165980e+00`, `-3.548059136550e+00`.
- N=16 `α_zero`: `+3.963938881832e-01`, `-3.316518268887e-01`, `+5.723899337959e-01`, `-1.372617422332e+00`, `+1.727734902049e+00`.
- N=32 `α_zero`: `-7.592558215532e-01`, `+1.816141425333e-01`, `+1.706796553408e+00`, `-1.131528357526e+00`, `+1.307514135777e+01`.

Status: rejected. The Hodge work defect is not a constant-factor error in the current Bernoulli gradient.

### 2026-06-03 continuation 16: independent-face adjoint rotational identity

Tested `/tmp/independent_adjoint_rotational_probe.jl`, which constructs the rotational corner transport from the exact independent-face adjoint Hodge covectors rather than topology-blind corner averages.

- The independent-adjoint rotational term has Hodge work at roundoff for all tested states.
- N=4 seed42: `rot_work=-2.117582368136e-22`, but total with current Bernoulli has `raw_total_rel=1.841773e+00` versus current.
- N=8: exact rotational work persists, but `raw_total_rel` is `5.844673e-01` to `8.949622e-01`.
- N=16: `raw_total_rel` is `1.712806e-01` to `4.619056e-01`.
- N=32: `raw_total_rel` is `2.648896e-01` to `5.156613e-01`.
- The older `op_sqrt` corner form remains much closer dynamically (`~0.03` to `0.08` raw total drift at N>=8) but is not exactly Hodge-skew.

Status: positive derivation identity, rejected as a direct replacement. This identifies the correct independent-face adjoint corner grouping and should guide a matched rotational/Bernoulli derivation or a smaller topology correction.

### 2026-06-03 continuation 17: exact rotational correction is x-edge-only, with west-side cost

Tested `/tmp/adjoint_rotational_boundary_flux_subset_probe.jl`, a corrected corner-flux subset diagnostic. A prior velocity-mixing attempt produced NaNs/garbage and should be ignored.

The exact independent-adjoint rotational correction from continuation 16 localizes entirely to x-edge corners for seed42 at N=4, 8, 16, and 32. Extra y-edge/boundary replacement has no additional effect. Replacing both x edges gives exact rotational Hodge-skew but reproduces the large direct independent-adjoint drift.

Key seed42 results:

- N=4: `op_sqrt raw_total_rel=4.246872e-02`; west-only `1.836891e+00`; east-only `1.405749e-01`; x-edges exact skew with `1.841773e+00`.
- N=8: `op_sqrt 4.302650e-02`; west-only `7.048223e-01`; east-only `1.326572e-01`; x-edges exact skew with `7.159058e-01`.
- N=16: `op_sqrt 6.581647e-02`; west-only `4.588690e-01`; east-only `8.442679e-02`; x-edges exact skew with `4.619056e-01`.
- N=32: `op_sqrt 3.847309e-02`; west-only `2.545692e-01`; east-only `8.271160e-02`; x-edges exact skew with `2.648896e-01`.

Status: structural diagnostic. Exact rotational Hodge-skew requires x-edge topology treatment; the west edge is the dynamically costly side. Next useful target is the west x-edge relationship between independent-adjoint corner coefficients, OHPSG covariant vector halo maps, and Hodge-ratio duplicate-face handling.

### 2026-06-03 continuation 18: x-edge skew is numerator/topology driven, not denominator driven

Tested `/tmp/xedge_corner_coefficients_probe.jl` and `/tmp/xedge_numerator_denominator_decomposition.jl`.

Coefficient diagnostics show the largest west-edge differences are numerator/sign/topology effects. Examples:

- N=16 seed42 west `j=7`: `opV=+8.805603e-01`, `indV=-8.429860e-01`, `op_work=-2.626701e-06`, `vmap=(1, 32, 7, -1)`.
- N=32 seed42 west `j=49`: `opV=-3.031774e+00`, `indV=+9.305486e-01`, `op_work=+8.021551e-07`, `vmap=(1, 64, 49, -1)`.

Numerator/denominator split:

- Using independent x-edge numerators with the old `op_sqrt` denominator makes x-edge rotational work vanish to roundoff: N=8, N=16, and N=32 seed42 all have `x_edges ind_num_op_den rot_work≈0`.
- This exact-skew numerator replacement is still dynamically too disruptive: `raw_total_rel=6.665887e-01` at N=8, `3.917525e-01` at N=16, and `2.079082e-01` at N=32.
- Changing only the denominator is smaller but does not enforce skew: west `op_num_ind_den raw_total_rel=2.524970e-01` at N=8, `1.252070e-01` at N=16, `8.815930e-02` at N=32.

Status: diagnostic. Exact rotational Hodge-skew at x edges is controlled by the independent-adjoint numerator/covector grouping, not by the denominator. The next source-worthy derivation should target the OHPSG covariant vector halo/duplicate-face grouping that produces the west x-edge numerator, while preserving dynamic fidelity.

### 2026-06-03 continuation 19: source-mapped x-edge Hodge numerator rejected

Tested `/tmp/xedge_source_mapped_hodge_numerator_probe.jl`, which replaces x-edge ghost Hodge covector contributions by signed covariant halo-source Hodge covectors. This is more structured than ghost-point Hodge evaluation and less extreme than exact independent ghost dropping.

Results for seed42:

- N=8: x-edge source-mapped numerator has `rot_work=+1.155917407364e-06`, `raw_total_rel=4.358272e-01`; exact independent numerator has `rot_work≈0`, `raw_total_rel=6.665887e-01`; current/op has `raw_total_rel=4.302650e-02`.
- N=16: x-edge source-mapped numerator has `rot_work=-2.847165577579e-07`, `raw_total_rel=2.837538e-01`; exact independent numerator has `rot_work≈0`, `raw_total_rel=3.917525e-01`; current/op has `raw_total_rel=6.581647e-02`.
- N=32: x-edge source-mapped numerator has `rot_work=-1.309994567181e-06`, `raw_total_rel=1.371390e-01`; exact independent numerator has `rot_work≈0`, `raw_total_rel=2.079082e-01`; current/op has `raw_total_rel=3.847309e-02`.

Status: rejected as direct source candidate. Source-mapped ghost Hodge numerators are still too disruptive and do not enforce skew. Denominator variations remain secondary.

### 2026-06-03 continuation 20: x-edge exact-Bernoulli pairing rejected

Tested `/tmp/xedge_matched_bernoulli_probe.jl`, pairing the x-edge independent-adjoint rotational numerator with localized exact Hodge-compatible Bernoulli corrections on the affected x-edge face DOFs.

- N=4 seed42: local exactB pairing has `raw_rel=5.246438e+01`; work-canceling scaled local exactB still has `raw_rel=1.384222e+01`.
- N=8 seed42: local exactB `raw_rel=9.513731e+01`; scaled local exactB `1.977586e+01`.
- N=16 seed42: local exactB `raw_rel=7.259226e+02`; scaled local exactB `1.726595e+01`.
- N=32 seed42: local exactB `raw_rel=1.856553e+02`; scaled local exactB `3.425463e-01`.

Status: rejected. Localizing exact Hodge-compatible Bernoulli to x-edge-affected faces does not fix the dynamic scale problem. The matched construction must be derived from a compatible local KE/rotational calculus, not by grafting the exact pressure-correction adjoint onto the current Bernoulli term.

### 2026-06-03 continuation 21: local x-edge cell-KE gradient fit rejected

Tested `/tmp/xedge_cell_ke_gradient_fit_probe.jl` and reduced continuation `/tmp/xedge_cell_ke_gradient_fit_probe_reduced.jl`. The diagnostic fits local cell-centered `K` perturbations in x-edge columns and applies the existing scalar Bernoulli-gradient operator to cancel the Hodge work of `xedge_rot + currentB`.

- N=4 seed42: base `raw_rel=1.85462631960297`; fitted `x_edges` total_rel `1.848712e+00`; fitted `x_near` total_rel `1.847432e+00`.
- N=8 seed42: base `raw_rel=6.665887038481539e-01`; fitted `x_edges` total_rel `7.078464e-01`; fitted `x_near` total_rel `7.012801e-01`.
- N=16 seed42: base `raw_rel=3.917525449047831e-01`; fitted `x_edges` total_rel `3.992035e-01`; fitted `x_near` total_rel `3.968775e-01`.
- N=32 seed42: base `raw_rel=2.0790818162021246e-01`; fitted `x_edges` total_rel `2.091238e-01`; fitted `x_near` total_rel `2.087199e-01`.

Status: rejected. Local scalar KE-gradient corrections can zero the scalar work, but they do not repair the dynamic drift introduced by the exact x-edge rotational numerator. This means a matched local KE-gradient patch is not sufficient unless the rotational x-edge numerator itself is made dynamically compatible.

### 2026-06-03 continuation 22: x-edge numerator blend/class parameterization rejected

Tested `/tmp/xedge_numerator_blend_class_probe.jl`, which blends the dynamically close `op_sqrt` x-edge numerator toward the exact independent-adjoint numerator by all x edges, west/east separately, and halo-source classes `k{1,2}_s{±1}`.

Representative zero-work blend coefficients:

- N=8 `all θ`: `+1.552839e+00`, `+8.476457e-01`, `-2.702971e+01`, `+3.484451e-01` for seeds 1, 2, 42, 99.
- N=16 `all θ`: `-2.205134e+00`, `+3.652117e+00`, `-3.247706e-01`, `+1.695082e+00`.
- N=32 `all θ`: `+2.200098e+00`, `-3.969562e+00`, `+3.776991e-01`, `+1.806441e+00`.
- N=32 `east θ`: `-1.149085e+02`, `+1.614100e+02`, `+9.285340e-01`, `-2.648211e+01`.

Status: rejected. No fixed edge-level or halo-class blend is stable across states/resolutions. The x-edge numerator must be re-derived as a local nonlinear mimetic expression rather than a parameterized blend toward the independent-adjoint numerator.

### 2026-06-03 continuation 23: closest local Hodge-skew x-edge projection is partial positive

Tested `/tmp/xedge_closest_skew_projection_probe.jl`. At x-edge corners, this projects the current `op_sqrt` transport `(U_op, V_op)` onto the locally skew line `(U, V) = α (H_u, H_v)`, where `(H_u, H_v)` are independent-face corner Hodge covectors.

Caveat: script labels `op` and `ind` both represent the direct independent x-edge numerator due to a naming bug. Use projected variant comparisons against `ind`; true `op_sqrt` baselines are from continuations 17/18.

Results:

- Rotational work is roundoff for all projected variants.
- N=8 seed42: direct independent raw_rel `6.665887e-01`; Euclidean projection `3.546929e-01`; V-weighted projection `3.503183e-01`.
- N=16 seed42: direct independent `3.917525e-01`; Euclidean `2.043353e-01`; V-weighted `2.092565e-01`.
- N=32 seed42: direct independent `2.079082e-01`; Euclidean `1.448957e-01`; V-weighted `1.398600e-01`.
- N=32 seed99: direct independent `5.042291e-01`; Euclidean `1.658119e-01`; V-weighted `1.631275e-01`.

Status: partial positive, not complete. The closest local skew projection is a better rotational building block than the direct independent numerator, but full VI work remains unbalanced because current Bernoulli is unchanged. It is not source-ready by itself.

### 2026-06-03 continuation 24: projected x-edge rotational plus local KE fit rejected

Tested `/tmp/projected_xedge_rot_ke_fit_probe.jl`, which fixes the prior projection-script labeling issue and pairs true `op_sqrt`, Euclidean projected x-edge rotational, V-weighted projected x-edge rotational, and direct independent x-edge rotational with local `x_near` cell-centered KE-gradient fits.

Representative corrected results:

- N=8 seed42: `op` raw_rel `4.302650e-02`, fitted `1.587948e-01`; Euclidean projected raw_rel `3.546929e-01`, fitted `3.961503e-01`; V-weighted `3.503183e-01`, fitted `3.900961e-01`.
- N=16 seed42: `op` raw_rel `6.581647e-02`, fitted `6.843196e-02`; Euclidean `2.043353e-01`, fitted `2.187320e-01`; V-weighted `2.092565e-01`, fitted `2.245778e-01`.
- N=32 seed42: `op` raw_rel `3.847309e-02`, fitted `4.073670e-02`; Euclidean `1.448957e-01`, fitted `1.462966e-01`; V-weighted `1.398600e-01`, fitted `1.413618e-01`.
- N=32 seed99: `op` raw_rel `3.519712e-02`, fitted `5.866758e-02`; Euclidean `1.658119e-01`, fitted `1.663450e-01`; V-weighted `1.631275e-01`, fitted `1.635086e-01`.

Status: rejected. Projected x-edge rotational variants are cleaner rotationally but remain too dynamically far from current, and local KE-gradient fitting does not recover dynamic fidelity. True `op_sqrt` remains the best rotational anchor despite not being exactly skew.

### 2026-06-03 update: x-edge least-norm work-correction lower bound

Added `/tmp/xedge_minimal_work_correction_probe.jl` to quantify the smallest x-edge corner-flux correction that cancels the scalar Hodge work residual around `op_sqrt + current Bernoulli` using `δW = H_u δF_u - H_v δF_v`.

Finding: the correction cancels work to roundoff, but it depends on the global work residual and has seed-dependent size. Both x-edges require `corr_rel` ranges of about `0.019-0.067` at N=8, `0.0087-0.105` at N=16, and `0.0054-0.0277` at N=32 for seeds 1, 2, 42, 99. This is useful as a lower-bound diagnostic and suggests perturbative x-edge fixes around `op_sqrt` may be viable, but the exact construction is nonlocal and not source-ready.

### 2026-06-03 update: multiplicative x-edge flux rescaling rejected

Added `/tmp/xedge_parallel_flux_correction_probe.jl` to constrain work cancellation to corrections parallel to existing `op_sqrt` x-edge corner fluxes, plus a uniform x-edge scaling check. Per-corner parallel corrections can cancel work but require unstable multipliers, including `max_gamma≈256` for N=32 seed99. Uniform scaling is worse, with corrected drift near `0.91` for N=32 seed99 and `corr_rel≈16.6` for N=8 seed42. This rejects simple multiplicative x-edge rescaling as a viable source direction.

### 2026-06-03 update: naive local residual x-edge correction rejected

Added `/tmp/xedge_local_residual_correction_probe.jl`. The rotational plus Bernoulli corner-density decomposition closes to roundoff, but the x-edge density is not a stable local representation of the global work residual: `xsum_t/W` ranged from `-0.1267` to `+1.6532` in sampled N=8, 16, and 32 cases. Local Hodge-covector corrections based on these densities require huge local coefficients and increase drift to about `0.13-0.30` in representative N16/N32 cases. This rejects naive local corner-residual replacement for the global least-norm x-edge correction.

### 2026-06-03 update: closest-projection rotational plus exact Hodge-adjoint Bernoulli rejected

Added `/tmp/projected_rot_exact_bernoulli_probe.jl` to close out the earlier recommendation to pair closest-projection x-edge rotational variants with `hodge_compatible_pressure_correction(K)`. The pairing gives roundoff work for the exactly skew rotational variants, but the exact Bernoulli replacement is dynamically unusable: `exactB_rel≈1.03e2` at N=8 seed42, `7.39e2` at N=16 seed42, `1.38e3` at N=16 seed99, and about `6.0e2-6.25e2` at N=32 seeds 42/99. This rejects the matched exact-adjoint Bernoulli pair as a source strategy.

### 2026-06-03 update: vorticity-weighted x-edge Hodge-covector corrections

Added `/tmp/xedge_zeta_weighted_correction_probe.jl` and `/tmp/xedge_den_weighted_correction_probe.jl`. The free global Hodge-covector correction remains the smallest in tendency norm but has pathological implied transport increments at small-vorticity corners, for example N32 seed99 has `max_dU≈208`, `max_dV≈315`. Weighting by `ζ²` gives a clean rotational-source shape and cancels work, but increases drift: N32 seed42 corrected drift `0.03930`, N32 seed99 `0.04961`. Denominator normalization (`ζ²/den`) does not materially improve this. Weighting by `absζ` is dynamically closer and avoids the largest transport pathologies, but implies a nonsmooth `sign(ζ)` transport correction and is not source-ready. These probes support a source-shaped perturbation around `op_sqrt`, but not yet a local production candidate because exact cancellation still relies on a global scalar normalization.

### 2026-06-03 update: fixed local feature x-edge correction rejected

Added `/tmp/xedge_fixed_feature_fit_probe.jl` to test whether fixed local x-edge features can replace the global scalar in the successful Hodge-covector correction. Features used west/east splits and weights `1`, `absζ`, `ζ²`, `absζ/den`, and `ζ²/den`, with coefficients fitted on N16 seeds 1-8 and evaluated on held-out N16 seeds 9-16 plus N32 seeds 1, 2, 42, 99. The 10-feature model overfits training work to roundoff but has mean N16 held-out drift `2.255` and N32 mean drift `19.24`. Two-feature models preserve N32 drift moderately but leave most work residual and have N16 held-out mean drift around `0.34-0.38`. A single constant feature is dynamically harmless but leaves about all N32 residual work. This rejects fixed-coefficient local feature maps as a replacement for the global work normalization.

### 2026-06-03 update: projected-tendency work is unchanged

Added `/tmp/projected_tendency_work_probe.jl`. Dense small-N Hodge projection matrices were built from source operators (`D`, `G=K⁻¹Dᵀ`, `P=I-G(DG)^+D`). Random velocities were projected first, then current/source VI and `op_sqrt + currentB` tendencies were recomputed and projected. Projection removes tendency divergence and changes the tendency by `~0.66-0.80` relative norm, but energy work is unchanged to roundoff: for example N8 seed42 current `W=+1.688487306772e-06`, projected `+1.688487306753e-06`; N8 seed42 `op_sqrt` `W=+1.782136628145e-06`, projected `+1.782136628124e-06`. This shows the residual is not a pressure/gauge artifact; it lies in the divergence-free tangent work.

### 2026-06-03 update: fixed linear x-edge transport corrections rejected

Added `/tmp/xedge_linear_transport_fit_probe.jl`, testing source-shaped quadratic corrections `δF=ζ δtransport`, with `δtransport` linear in local `op_sqrt` transport and/or Hodge-adjoint transport components. Coefficients fitted on N16 seeds 1-8 do not generalize into a viable correction. `op_components`, `op_plus_hodge`, and `exact_minus_op` produce order-one drift at N16/N32. `hodge_cross` still has N32 mean drift `0.440`. `hodge_components` is dynamically modest but leaves almost all residual work (`N32 mean relative residual work 0.929`). This rejects fixed local linear x-edge transport corrections as a source strategy.

### 2026-06-03 update: alternate energy pairings rejected

Added `/tmp/energy_pairing_work_probe.jl` and `/tmp/energy_pairing_fit_probe.jl`. Direct pairings with raw covariant velocity, contravariant flux, target Hodge flux, `J`-weighted covariant velocity, Hodge-weighted covariant velocity, `J`-weighted contravariant flux, and `hodge*J` covariant velocity do not reveal a stable hidden conserved energy. The best cancellation changes by state. A fitted N16 linear combination can cancel work by subtracting about `0.006135923 * contra_flux`, but this nearly annihilates the N16 Hodge covector itself (`qrel=1`) and fails at N32 with relative residual work `≈3` and `qrel=4`. Covariant-style restricted fits have large covector changes and poor held-out/resolution generalization. This rejects hidden-energy-covector mismatch as an explanation; the defect remains in the nonlinear tangent VI operator under the target Hodge energy.

### 2026-06-03 update: x-edge work is distributed, not a small-corner defect

Added `/tmp/xedge_work_concentration_probe.jl` and `/tmp/xedge_topk_correction_probe.jl`. Ranking x-edge corner work shows strong signed cancellation and state-dependent top locations. West x-edge usually dominates absolute work, but east contributions can be top-ranked and are not negligible. Restricting the successful global least-norm correction to top-K corners confirms the defect is distributed: for N32 seed99, all-xedge correction has `corr_rel=0.02321`, drift `0.04215`; top-rot K=4 has `corr_rel=0.06933`, drift `0.07775`; K=32 still has `corr_rel=0.03360`, drift `0.04878`; K=64 approaches the all-xedge bound with `corr_rel=0.02558`, drift `0.04350`. This rejects a fixed small-corner patch. A source-ready fix would need a systematic full-x-edge topology operator.

### 2026-06-03 update: west/east edge-wise normalization mostly rejected

Added `/tmp/xedge_edgewise_normalization_probe.jl`. Separate west/east edge normalizations were tested using rotational and total edge densities. Local edge-density cancellation often leaves substantial global work. Globally scaled edge-density splits cancel work, but are not robustly better than the uniform all-xedge lower bound. Example N32 seed99: all-xedge `corr_rel=0.02321`, drift `0.04215`; rotational edge-scaled `corr_rel=0.03201`, drift `0.04747`; total edge-scaled `corr_rel=0.03091`, drift `0.04675`. N32 seed42 has only a tiny improvement from rotational edge-scaling. This rejects simple west/east residual normalization as a source strategy.

### 2026-06-03 update: source topology inspection and current-anchor lower bound

Inspected `src/Operators/nonorthogonal_metric_operators.jl`, `src/Grids/spherical_shell_grid.jl`, and `src/BoundaryConditions/fill_halo_regions_quadfoldedzipper.jl`. The production VI rotational operator uses ordinary interpolated contravariant corner velocities, while the validated Hodge-compatible topology machinery lives separately in `hodge_compatible_boundary_flux_*`, `hodge_compatible_volume_flux_div_xyᶜᶜᶜ`, and `hodge_compatible_pressure_correction_*`, using explicit `octahealpix_covariant_*face_halo_source` maps and diagonal ratios. This reinforces the current diagnosis: the remaining defect is an x-edge rotational/topology issue, not projection, pressure, or hidden-energy pairing.

Added `/tmp/current_anchor_minimal_correction_probe.jl` to measure the all-xedge least-norm Hodge-covector correction around the actual source `current_total` anchor. The required correction is the same order as around `op_sqrt`: N32 seeds 1,2,42,99 need `corr_rel=0.01093`, `0.02837`, `0.00631`, `0.02208`; N16 seeds 42,99 need `0.01519`, `0.02171`. West-only/east-only corrections are larger. This means the lower-bound correction scale is not an artifact of changing from current source VI to the `op_sqrt` anchor.

### 2026-06-03 update: anisotropic exact-skew x-edge projection rejected

Added `/tmp/xedge_projection_weight_scan_probe.jl`, scanning exact local Hodge-skew x-edge projections `(U,V)=α(H_u,H_v)` with anisotropic closeness weight `|δU|² + r|δV|²` for `r=1e-8...1e8`. The best ratios are around `1.78-3.16`, but best dynamic drift remains much larger than `op_sqrt`: N32 seed42 improves only to `0.13879` versus `op_raw_rel=0.03847`, and N32 seed99 to `0.16153` versus `0.03520`. This rejects exact local skew projection independently of the arbitrary weight choice used in earlier probes.

### 2026-06-03 update: Hodge-compatible halo-source mapped rotational velocity rejected

Added `/tmp/xedge_hodge_mapped_velocity_probe.jl` to test source-level x-edge rotational transport using `octahealpix_covariant_*face_halo_source` maps. The diagonal-ratio variant based on `hodge_compatible_boundary_flux_*` reproduces current source rotational advection to roundoff, so it is not the missing topology correction. A sign-only halo-source map changes work but has unacceptable drift: `total_rel=0.4357` at N8 seed42, `0.2759` at N16 seed42, `0.1316` at N32 seed42, and `0.4268` at N32 seed99. This rejects direct reuse of existing Hodge-compatible halo-source maps inside rotational corner velocity as a source fix.

### 2026-06-03 update: x-edge mapped-vorticity topology is already source-equivalent

Added `/tmp/xedge_mapped_vorticity_probe.jl`, recomputing x-edge `ζ` with covariant halo-source mapping for the line-integral values while leaving source rotational transport unchanged. The mapped-vorticity variant is identical to source rotational advection to roundoff across N8, N16, and N32 sampled seeds. This rejects vorticity halo mapping as the missing topology correction; the defect remains in corner transport / work-adjoint grouping.

### 2026-06-03 update: norm-preserving exact-skew transport rotation rejected

Added `/tmp/xedge_norm_preserving_skew_probe.jl`, which rotates each x-edge `op_sqrt` transport vector onto the independent-adjoint skew direction `(H_u,H_v)` while preserving its magnitude. This exact-skew variant tests whether prior projected-skew failures were due to lost amplitude. It is still too dynamically far: N32 seed42 `op_rel=0.03847`, norm-preserving skew `0.16008`; N32 seed99 `0.03520` versus `0.17079`; N32 seed2 `0.05152` versus `0.29988`. Alignment statistics show distributed bad corners despite good median alignment, e.g. N32 seed99 has `|cosθ|` min `0.00794`, q25 `0.569`, median `0.932`. This rejects norm-preserving exact-skew rotation and reinforces that the exact skew direction itself is dynamically too far.

### 2026-06-03 update: projected all-xedge correction is mostly tangent

Added `/tmp/projected_minimal_correction_probe.jl`. Dense N4/N8 projection matrices were used to project the all-xedge least-norm Hodge-covector correction around the actual `current_total` VI tendency. Projection removes correction divergence and leaves its energy work unchanged, but retains most of the correction norm: N8 seed1 raw `corr_rel=0.02092`, projected `0.01715`; N8 seed42 raw `0.06377`, projected `0.05623`; N8 seed99 raw `0.02082`, projected `0.01800`. The correction has a nontrivial pressure/gauge component but is mostly a real tangent-space correction. It cannot be ignored by relying on projection.

### 2026-06-03 update: global all-xedge correction is not a quadratic VI operator

Added `/tmp/global_correction_quadratic_probe.jl`. The all-xedge least-norm correction around current VI has correct sign symmetry and homogeneity (`C(-u)=C(u)`, `C(2u)=4C(u)`, `C(u/2)=C(u)/4`), but it fails the quadratic polarization identity `C(u+v)+C(u-v)=2C(u)+2C(v)`: relative errors are `0.247` at N8, `0.882` at N16, and `0.590` at N32 for seeds 42/99. The source VI tendency passes the same identity to roundoff. Therefore the global correction is a homogeneous rational energy fixer, not a bilinear/quadratic VI operator; it remains a lower bound diagnostic, not a source-ready advection term.

### 2026-06-03 update: class-split exact-minus-op x-edge correction rejected

Added `/tmp/xedge_class_split_exact_minus_op_fit_probe.jl`, fitting the difference between `op_sqrt` x-edge transport and exact independent-adjoint transport with coefficients split by side, component, covariant halo-source kind, and sign. The 24-feature model overfits N16 training work to roundoff but has mean training drift `19.57`, held-out N16 mean drift `19.39`, and N32 mean drift `18.40`. Nonzero coefficients are huge, up to `154`. This rejects richer topology-class blending toward the exact independent-adjoint transport as a local quadratic source strategy.

### 2026-06-03 update: linear beta predictor rejected

Added `/tmp/xedge_beta_linear_predictor_probe.jl` to approximate the global all-xedge scalar `β=-W/S` by a linear functional of x-edge state features, making the correction quadratic if successful. The model overfits N16 training seeds exactly, but fails held-out N16 with mean beta/relative-work error `3.66` and max `7.86`. N32 evaluation is not viable: seed2 leaves almost all work (`relwork=0.997`), seed42 overshoots (`relwork=3.54`), and seed99 leaves most work (`0.846`). This rejects the simplest quadratic/bilinearization of the global all-xedge correction.

### 2026-06-03: nonlocal post-advection energy-fixer branch quantified

`/tmp/nonlocal_energy_fixer_probe.jl` evaluated the global all-xedge minimal Hodge-covector correction applied around the current source tendency. It cancels Hodge work to roundoff for all sampled states. Relative correction sizes were modest, especially at N32: N16 seeds 1:8 had mean/max correction norm `5.13e-2 / 1.01e-1`; N32 seeds 1:8 had `1.84e-2 / 2.84e-2`. The N32 max component-relative correction was at most `7.37e-2` over seeds 1:8.

However this remains a fallback mechanism, not a VI operator candidate. The correction depends on a global scalar `β = -W / sum(Hu^2 + Hv^2)`, has substantial pre-projection divergence (`~0.65-0.90` normalized on N32), and prior polarization diagnostics showed it is homogeneous but not bilinear/quadratic. An amplitude scan confirmed the homogeneity (`beta/amp` constant and `corr_rel` constant for amplitudes `0.25, 0.5, 1, 2`). This path is viable only if the goal pivots from a local quadratic vector-invariant discretization to a global energy-fixer/projection mechanism.

### 2026-06-04: compact local x-edge source/raw/op stencil family rejected

`/tmp/xedge_compact_stencil_fit_probe.jl` tested a genuinely local quadratic additive correction to the existing source tendency, using x-edge corner flux basis terms built from west/east side, target corner flux (`Fu`, `Fv`), and local transported modes (`sourceU`, `sourceV`, `rawU`, `rawV`, `opU`, `opV`). The completed `source_raw_op` family used N16 seeds 1:16 for fitting, N16 seeds 17:24 for holdout, and N32 seeds 1:8 for evaluation.

The family is rejected. Unregularized fits cancel training work but are dynamically enormous and fail holdout/N32: at `lambda=0`, train relative work was `~1e-12`, but N16 holdout mean/max relative work was `1.01e2 / 3.17e2`, N32 was `8.55 / 16.7`, and correction norms reached order `10`. Regularized fits reduce correction size only by leaving the work defect: at `lambda=1e4`, N32 correction norms were only `9.11e-3 / 1.66e-2`, but N32 relative work remained `9.49e-1 / 1.04`.

This closes another local quadratic corner-stencil route. The second `hodge_num` family in the same script remained too slow because the script repeatedly builds projected models and basis tendencies; it should be replaced by a smaller/reused-model probe rather than treated as a source-ready result.

The delayed `hodge_num` branch of `/tmp/xedge_compact_stencil_fit_probe.jl` also completed and is rejected. It used x-edge local Hodge/numerator modes (`hU`, `hV`, `numU`, `numV`, `opU`, `opV`). Weakly regularized fits did not even cancel training work cleanly and failed badly across resolution: at `lambda=0`, train relative work was `2.40e-1 / 7.18e-1`, N16 holdout was `1.19e1 / 5.02e1`, and N32 was `2.14e2 / 4.85e2`, with N32 correction norms up to `6.97e1`. Strong regularization reduced correction size but left the defect (`lambda=1e4`: N32 relative work `1.31 / 1.78`, correction norms `1.28e-1 / 2.14e-1`).

Thus both compact 24-term local x-edge stencil families in this probe are rejected. The remaining local path likely requires a derived topological regrouping of the seam/fold degrees of freedom rather than a fixed-coefficient local basis built from the obvious corner transports, Hodge covectors, or metric numerators.

### 2026-06-04: simple west/east seam-pair skew regrouping rejected

`/tmp/xedge_pairwise_skew_probe.jl` tested a direct topology hypothesis: reconstruct the current rotational term from corner fluxes and project each same-`j` west/east x-edge corner pair so its local Hodge work is zero. The reconstruction was exact (`recon_rel = 0`) for all sampled N8/N16/N32 states, confirming that corner-flux diagnostics faithfully represent the current source rotational advection.

The pairwise skew constraint is rejected. It zeroes x-edge rotational work by construction, but total Hodge work is not corrected and often worsens. Examples: N16 seed 42 changes total work from `+9.063e-7` to `+2.505e-6` with correction norm `1.35e-1`; N32 seed 42 changes `+1.382e-6` to `-1.760e-6` with correction norm `9.32e-2`; N32 seed 99 changes `+4.263e-6` to `+1.781e-6` with correction norm `1.11e-1`.

This shows the target x-edge rotational contribution is not zero pair-by-pair. It must cancel a state-dependent Bernoulli/interior residual. The next viable local derivation needs to couple Bernoulli gradients and rotational corner fluxes in one seam/fold complex, rather than imposing a standalone rotational skew condition on paired x-edge corners.
