# Non-orthogonal Spherical C-grid Implementation Plan (Quantitative, 2026-05-23, revised)

## Contract and scope

Goal: production-quality non-orthogonal C-grid dynamics on `SphericalShellGrid` in Oceananigans, with a strict phase-gate rollout:

1. new grid + mapping stack (`SphericalShellGrid`, `OctaHEALPixMapping`, `OctaHEALPixConnectivity`)
2. non-orthogonal operator/Hodge stack
3. transport-state separation (`model.velocities` vs `model.transport_velocities`)
4. `QuadFolded` seam orientation for scalar, vector, and transport fields
5. tracer advection, linear momentum, centered vector-invariant, WENO vector-invariant on folded topologies

Hard constraints:
- keep `OrthogonalSphericalShellGrid` API and regression behavior unchanged
- no root dependency changes in `Project.toml` (SpeedyWeather/RingGrids are offline references only)
- obey AGENTS GPU rules (`@kernel`, type stability, no model logic in kernels, no `Float64` literals)

Current critical blocker status:
- `compute_transport_velocities!` on `SphericalShellGrid` has been migrated from copy-on-update to explicit non-orthogonal `covariant_to_volume_flux` conversion in this run.

Source evidence:
- `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl`
- `src/Advection/tracer_advection_operators.jl`
- `src/Grids/spherical_shell_grid.jl`
- `test/test_spherical_shell_grid.jl`

## Shared metrics

For scalar field \(q\), face/area weights \(A\), and exact reference \(q^\star\):

\[
\|q\|_{2,A} = \left(\frac{\sum A q^2}{\sum A}\right)^{1/2},\quad
\epsilon_2 = \frac{\|q-q^\star\|_{2,A}}{\|q^\star\|_{2,A}+\epsilon_A},\quad
\epsilon_M(t) = \frac{|M(t)-M_0|}{|M_0|+\epsilon_M}.
\]
\[
E_{\text{seam}} = \max_{(a,b)\in\mathcal S}|q_a-\mathcal T_{ab}q_b|,\quad
R_J=\frac{E_{\text{seam}}}{\max(\Delta q_{\text{interior}},\,100\epsilon_{\text{mach}}\|q\|_\infty)}.
\]

Convergence order from \(N_1=64\), \(N_2=128\):
\[
p=\frac{\log(E_{N_1}/E_{N_2})}{\log2}.
\]

Pass policy:
- release hard pass: explicit quantitative bound met
- warning band: metric-specific relaxed bounds (for local iteration)
- CI fail: any explicit hard bound miss (hard gate)

## Current source audit (2026-05-23)

Deep dive into current Oceananigans source shows the remaining implementation front is narrowly scoped:

1. `SphericalShellGrid`/`OctaHEALPixConnectivity` are functional and tested for topology and ring/matrix indexing.
2. Transport conversion in HFSM is explicit (`compute_transport_velocities!`) and no longer a pure copy for spherical shells.
3. The `QuadFolded` seam is currently handled for scalar halos only in
   `src/Fields/field.jl` via a special `fill_halo_regions!` path keyed on `OctaHEALPixConnectivity`.
4. There is no generic `QuadFold` vector/flux orientation map yet (no dedicated boundary transform type; no `QuadFoldZipperBoundaryCondition` tensor kernels).
5. Existing tripolar zipper machinery (`UZBC/FZBC`) is in-source but not wired to `QuadFolded` octahedral semantics.

Current hard blockers (must be closed before Phase 4 onward):

- `test_octahealpix_seam_halo_consistency` (in-tree) currently exercises `H=1` scalar continuity only.
- `src/BoundaryConditions/field_boundary_conditions.jl` and `src/BoundaryConditions/fill_halo_regions_*.jl` lack full edge/corner orientation transforms for:
  - vector components on `fcc/cfc`,
  - transport tensors on `fcc/cfc/ffc`,
  - folded paths at halo depths `H = 1,2,3,6`.

Phase 3 and later will therefore include one explicit “seam-contract” gate before any WENO unlock:

- seam identity mismatch: 0 (exact), all tested fold paths and halo depths.
- mapped vector/tensor component continuity: machine tolerance `1e-13`.
- no unmapped fold-edge/corner transport blocks.
- seam transport ratio `R_J ≤ 1` at release.

## Phase 0 — Algebraic/operator foundations (blocking all later phases)

## Phase 0 — Algebraic/operator foundations (blocking all later phases)

**Files:** `src/Operators/nonorthogonal_metric_operators.jl`, `src/Operators/divergence_operators.jl`, `src/Operators/Operators.jl`, `test/test_spherical_shell_grid.jl`

Targets:
- lock metric algebra on `SphericalShellGrid`
- choose Hodge candidate by test-driven criteria (not by convention

Gates (both CI + release):
- free-stream residual \( \|D\mathbf F\|_1 / \|\mathbf F\|_1 \le 1e{-14}\)
- metric duality \(g_{ij}g^{jk}\) max-normalized error \(\le 1e{-13}\)
- adjointness residual \( |\langle \eta, D V\rangle - \langle G\eta,V\rangle|/(|\langle V,G\eta\rangle|+100\varepsilon_{mach}) \le 1e{-13}\)
- symmetric positive-definite Hodge: \( \lambda_{min}(W H) / \lambda_{max}(W H) \ge 1e{-12}\)
- orthogonal-limit test at least \(1e{-13}\) for all shared kernels used by non-orthogonal grid

Exit rule: all release-gate checks pass.

## Phase 1 — OctaHEALPix geometry and mapping parity

**Files:** `src/Grids/spherical_shell_grid.jl`, `test/test_spherical_shell_grid.jl`

Targets:
- correct topology, bijections, and equal-area structure at \(N \ge 64\)
- parity checks against independent formulas/implementations (offline only)

Gates:
- topology is exactly `(QuadFolded, QuadFolded, Bounded)` for `OctaHEALPixMapping(N)`
- `topology` grid size is exactly `(2N, 2N, 1)` (release)
- exact formula checks for:
  - `octahealpix_number_of_cells = 4N^2` (CI + release)
  - `octahealpix_nlon_per_ring` (CI + release)
  - latitude/longitude formulas (`atol ≤ 1e-14`)
- connectivity round trips (`rcq2ring`, `ring2rcq`, `ring2matrix`, neighbors):
  - CI for \(N=8,16,32\)
  - release for \(N=64,128\)
- area closure and equal-area spread: relative error/ratio \(\le 5e{-14}\)

## Phase 2 — Transport-state lifecycle in HFSM (high priority)

**Files:** `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl`, `src/Models/HydrostaticFreeSurfaceModels/compute_w_from_continuity.jl`, `src/Advection/tracer_advection_operators.jl`, `test/test_spherical_shell_grid.jl`

Targets:
- replace blind transport copy with explicit non-orthogonal conversion from model `u, v` to transport-consistent fields
- make transport continuity/advection rely on `model.transport_velocities` and validated `div_Uc`

Status:
- ✅ Completed: transport conversion now writes transport-conservative face fluxes for `SphericalShellGrid`.
- ✅ Added regression tests that compare `transport_velocities` to `covariant_to_volume_flux` and require nontrivial separation from raw `velocities`.

Gates:
- non-orthogonal explicit separation:
  - `parent(model.transport_velocities.u) !== parent(model.velocities.u)` etc (CI exact)
  - `model.transport_velocities` numerically differs from velocities except on orthogonal mappings
- continuity-level check with `u,w` update:
  - one-step tracer mass drift \(\le 1e{-11}\)
  - 10-step tracer mass drift \(\le 1e{-11}\)
- transport diagnostics liveness:
  - continuity update residual \( \| \mathcal D_{\mathcal V} + \partial_t\eta\|_\infty \le 200\,\epsilon_{mach}\) in folded non-orthogonal runs
- seam transport consistency:
  - \(R_J \le 2\) (warning) must reach \(R_J\le1\) at release for transport diagnostics

Required code behavior:
- `compute_transport_velocities!` must not fallback to field copy for `SphericalShellGrid`.
- `update_vertical_velocities!` remains source of `w` transport consistency.

## Phase 3 — QuadFolded seam orientation (`u,v`, edge/corner transport tensors)

**Files:** `src/BoundaryConditions/field_boundary_conditions.jl`, `src/BoundaryConditions/fill_halo_regions_upivotzipper.jl`, `src/BoundaryConditions/fill_halo_regions_fpivotzipper.jl`, `src/BoundaryConditions/fill_halo_kernels.jl`, `test/test_spherical_shell_grid.jl`

Targets:
- generalize folded halo behavior from tripolar pivots to full `QuadFolded` tensor-aware transforms
- support all field locations (`ccc`, `fcc`, `cfc`, `ffc`, halo depths `H=1,2,3,6`)

Gates:
- scalar seam identity mismatches exactly 0 for all tested `H=1,2,3,6` and fields
- vector and tensor source mapping completeness for every edge/corner fold path = 0 unmapped
- orientation/sign matrix mismatch max abs \( \le 1e{-13}\) for all halo maps
- transport tensor mismatch on folded boundaries = 0 for analytic regression fields
- seam ratio for mapped `λ/φ/u/v/transport` fields at closure: \(R_J\le1\) (release)

## Phase 4 — Non-orthogonal tracer advection campaign

**Files:** `src/Advection/tracer_advection_operators.jl`, `src/Advection/Advection.jl`, `test/test_spherical_shell_grid.jl`

Targets:
- finalize `NonOrthogonalScalarAdvection` path if needed
- move from one-step smoke tests to sustained folded-domain transport validation

Gates:
- cosine-bell, 1 rotation @ `N=32`: \( \epsilon_M \le 1e{-12}\), \( \epsilon_{2} \le 5e{-2}\)
- cosine-bell, 10 rotations @ `N=128`: \( \epsilon_2 \le 1e{-2}\), centroid drift \(\le 0.5\) mean-cell spacing
- cosine-bell, 100 rotations @ `N=128`: \( \epsilon_M \le 1e{-10}\), \( \epsilon_2 \le 5e{-3}\), centroid drift \(\le 0.25\) mean-cell spacing
- seam transport ratio for tracer flux fields: \(R_J \le 2\) (CI), \(\le1\) (release)

## Phase 5 — Linear momentum path (`momentum_advection = nothing`)

**Files:** `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl`, `src/Models/HydrostaticFreeSurfaceModels/compute_hydrostatic_free_surface_tendencies.jl`

Targets:
- verify momentum-disabled dynamics for conservation and weak-mode balance

Gates:
- constant-field tendency residuals \( \le 1e{-13}\)
- mass conservation drift over 20 inertial periods \( \le 1e{-10}\)
- low-mode gravity-wave frequency/energy trend within reference band from existing orthogonal baseline (same geometry scaling)
- orthogonal-limit consistency for all shared non-orthogonal operator paths at \(N=128\): residual \(\le 1e{-12}\)
- seam ratio for `(η,u,v)` and continuity diagnostics at closure: \(R_J\le1\)

## Phase 6 — Centered non-orthogonal vector invariant

**Files:** `src/Advection/vector_invariant_advection.jl`, `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_tendency_kernel_functions.jl`, `test/test_spherical_shell_grid.jl`

Targets:
- complete nonlinear centered vector-invariant momentum with non-orthogonal vorticity/transport coupling

Gates:
- split identity (`U·∇u = skew/rotational + Bernoulli`) relative residual \(\le 1e{-12}\)
- orthogonal limit recovery: residual \(\le 1e{-12}\)
- solid-body rotation mass drift:
  - 10 rotations @ `N=64`: \(\le 1e{-11}\)
  - 100 rotations @ `N=128`: \(\le 1e{-10}\)
- energy trend bounded (no secular growth vs shared baseline over 100 steps)
- convergence of nonlinear centered residuals with smooth forcing \(p\ge1.8\) on non-orthogonal meshes

## Phase 7 — WENO vector invariant

**Files:** `src/Advection/vector_invariant_advection.jl`, `src/Advection/tracer_advection_operators.jl`, `test/test_spherical_shell_grid.jl`

Core rule: reconstruct relative vorticity first, then add planetary term.
\[
\widehat Z = \mathcal R_{\mathrm{WENO}}[\zeta] + \mathcal R_{\mathrm{smooth}}[f].
\]

Targets:
- pass contamination and seam gates before removing runtime block
- replace constructor-level rejection with working pathway

Gates:
- unit test confirms WENO input is \(\zeta\), not absolute vorticity (exact)
- small-Rossby contamination ratio \( \le 2\) (CI), \( \le 1.25\) release at \(|f|/|\zeta|=10^4\)
- smooth convergence: \(p\ge4.0\) (CI), \(p\ge4.5\) release away from folds
- seam anomaly:
  - \(R_J \le 1.25\) in reconstructed vorticity and transports
  - fold-crossing amplification ratio (across vs away) \(\le 1.25\)
- no fold-mode unstable branch: spectral growth under identical forcing bounded against non-orthogonal centered baseline
- all previous phase gates remain valid once WENO is enabled

## PR gate sequence

1. Phase 0
2. Phase 1
3. Metric/Hodge candidate choice + Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7

No phase may advance unless all release gates in the prior phase are satisfied.

## External parity workflow (offline only)

SpeedyWeather/RingGrids can be used for parity, but never as runtime dependencies:
1. local checkout only
2. export ring/matrix/indexing/latitude-longitude arrays
3. compare against:
   - `test_octahealpix_reference_formulas`
   - `test_octahealpix_indexing`
   - dedicated parity scripts (CI-light, one-way)

Release acceptance:
- all phase hard gates + full regression of `OrthogonalSphericalShellGrid` behavior
