# `SphericalShellGrid` design for non-orthogonal C-grid dynamics

## 1. Purpose

`SphericalShellGrid` is the proposed Oceananigans/Breeze grid type for structured finite-volume C-grid dynamics on a spherical shell. The name is intentionally neutral: we should not split the API into `OrthogonalSphericalShellGrid` and `NonOrthogonalSphericalShellGrid`. Orthogonality is a property of the mapping and metric/Hodge operators, not a different model concept.

The grid must support both:

1. **thin-atmosphere / shallow-shell dynamics**, where the radial coordinate is vertically orthogonal to the spherical surface and horizontal metric factors are evaluated at a representative radius; and
2. **deep-atmosphere / full spherical-shell dynamics**, where horizontal metrics vary with radius and the Coriolis, gravity, and geometric terms are allowed to use the full radius-dependent geometry.

The first implementation should target the thin-shell case because it isolates the hardest C-grid issue: non-orthogonal horizontal metrics and staggered Hodge maps.

## 1.0 Source-state snapshot (2026-05-23, in-tree)

This file is now treated as the execution design spec for current Oceananigans work, not a future rewrite target.

- `SphericalShellGrid`, `EquiangularGnomonicCubedSpherePanel`, `OctaHEALPixMapping`, and `OctaHEALPixConnectivity` are implemented in-tree and tested in `test/test_spherical_shell_grid.jl`.
- `nonorthogonal_metric_operators` and routing for `div_Uc` and vorticity diagnostics are present and tested.
- `HydrostaticFreeSurfaceModel` supports `momentum_advection = nothing` and centered `VectorInvariant` on `SphericalShellGrid`, while still rejecting `WENOVectorInvariant`.
- `model.transport_velocities` on `SphericalShellGrid` now uses explicit non-orthogonal conversion in `compute_transport_velocities!`; transport separation is no longer copy-only for this path.
- `QuadFolded` boundary defaults exist, but transport/flux orientation transforms are not yet complete for all edge/corner cases (`H = 1,2,3,6`) in folded halo kernels.

Hard implementation commitments from source inspection:

1. Do not touch `AGENTS.md` constraints: no root dependency changes, no hardcoded `Float64` in kernels/constructors, explicit imports, GPU-safe kernels.
2. Keep `OrthogonalSphericalShellGrid` behavior unchanged until all phase gates pass on non-orthogonal paths.
3. Only one grid type family (`SphericalShellGrid`) for shell dynamics; mapping determines orthogonality and metric behavior.
4. Keep WENO unblocked only after explicit contamination and seam-orientation gates pass.

If you are implementing phases from the quantitative plan, use it as the gatekeeper:
`oceananigans_nonorthogonal_spherical_cgrid_quantitative_plan.md`.

### Current execution posture

In-tree priority is:

1. `compute_transport_velocities!` on `SphericalShellGrid` now uses explicit conversion (completed).
2. `QuadFolded` halo transforms must cover scalar, vector, and flux orientations at all required depths (`H = 1,2,3,6`).
3. Centered nonlinear vector-invariant and linear-momentum phase gates close.
4. WENO remains blocked until the above gates pass.

## 1.1 Source-informed revision (2026-05-23)

This design has to fit the current Oceananigans implementation, not just the continuous geometry.

### Update to this plan in-progress

- Transport conversion is now explicit on `SphericalShellGrid`; the current priority is fold orientation completion and vector-invariant closure.
- `QuadFolded` BC dispatch is in place, but transport/vector orientation maps are still incomplete (vector and transport tensors across fold edges/corners are not yet implemented).
- The immediate architecture should prioritize:  
  1) `compute_transport_velocities!` non-orthogonal specialization,  
  2) seam orientation identity gates for scalar/vector/transport,  
  3) centered nonlinear vector-invariant acceptance,  
  4) WENO unlock.

### Execution update (2026-05-23)

This branch now has most of the structural pieces in place, but the remaining hard blockers are concentrated in folded orientation and vector-invariant closure:

- `compute_transport_velocities!` now writes explicit face-area transports for `SphericalShellGrid`, and remaining transport lifecycle work is validation depth and free-surface branch consistency.
- Fold-aware BC handlers currently cover scalar/velocity conventions from tripolar zipper heritage but do not yet provide full transport-orientation closure for all edge and corner paths required by the folded octahedral topology.
- WENO remains blocked by construction (`validate_momentum_advection` still throws for `WENOVectorInvariant` on `SphericalShellGrid`) until contamination/seam gates close.

The recommended next step is to treat the implementation as a strict phase graph:

1. close transport separation and seam-transform correctness,  
2. close tracer cosine-bell and diffusion/reconstruction gates,  
3. pass centered nonlinear vector-invariant identity/orthogonal-limit gates,  
4. unlock WENO only after all contamination and seam anomaly gates pass.

### Source-aligned integration checkpoints

Observed in-tree integration points for the 1st implementation pass:

- `SphericalShellGrid` is a concrete `AbstractHorizontallyCurvilinearGrid`, so geometry arrays are placed exactly like existing horizontal grids (`λ`, `φ`, `Δx`, `Δy`, `Az`, `volume`).
- `materialize_advection(m, grid)` is the allocation seam for scheme state; transport diagnostics that depend on model state must be refreshed via `compute_model`-time update calls in the scheme/materialized advection object.
- `HydrostaticFreeSurfaceModel` already keeps a separation between `model.velocities` and `model.transport_velocities`, with explicit conversion now implemented on `SphericalShellGrid`.
- Existing BC defaults dispatch on topology tags; `QuadFolded` is now the explicit tag for octahedral seam geometry.
- Seam treatment in this branch currently handles scalar and velocity boundary classes but not full transport/edge-orientation tensors; that is the hard physics/BC frontier.
- `model.transport_velocities` for HFSM on `SphericalShellGrid` is now updated from covariant velocities via the non-orthogonal Hodge map; residual work is in fold transport orientation.

### Source-anchored status ledger

| Design checkpoint | Status | Evidence |
|---|---|---|
| `SphericalShellGrid` construction & API parity | ✅ | `src/Grids/spherical_shell_grid.jl`, `test/test_spherical_shell_grid.jl` |
| `OrthogonalSphericalShellGrid` unchanged for existing workflows | ✅ | existing orthogonal test set; constructor defaults in `src/Grids` and `src/BoundaryConditions` |
| Non-orthogonal tracer continuity route through `div_Uc` | ✅ | `src/Advection/tracer_advection_operators.jl`, `src/Operators/divergence_operators.jl` |
| QuadFolded boundary-condition defaults | ✅ | `src/BoundaryConditions/field_boundary_conditions.jl`; tests in `test/test_spherical_shell_grid.jl` |
| WENO momentum blocked with explicit error message | ✅ | `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl` |
| `QuadFold` transport/orientation halo tables for vector fields | ⛔ | `src/BoundaryConditions/fill_halo_regions_*.jl` only cover existing tripolar zipper flavors |
| `NonOrthogonalScalarAdvection` and flux-centered transport diagnostics | ⛔ | no dedicated scheme exists yet in `src/Advection/` |
| Multi-run non-orthogonal tracer/wave benchmarks | ⛔ | current tests remain mostly one-step or low-order only |

Required source-anchored checks before moving forward:

1. `FieldBoundaryConditions` on `QuadFolded` must preserve API defaults for auxiliary and prognostic fields at each location.
2. `OrthogonalSphericalShellGrid` behavior and APIs remain unchanged.
3. The Hodge map and diagnostics path must run through existing `div_Uc`/`compute_w_from_continuity`/`vertical_vorticity` pathways for continuity and tracer consistency.

### Source-driven implementation matrix (required before Phase-3+)

1. **Transport diagnostics lifecycle**
   - File targets:
     - `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl`
     - `src/Models/HydrostaticFreeSurfaceModels/compute_w_from_continuity.jl`
   - Requirement:
  `compute_transport_velocities!` now converts covariant fields into explicit volume transports on `SphericalShellGrid`.
   - Metrics:
     one-step and multi-step tracer masses with `model.velocities ≠ model.transport_velocities`, with mass drift `≤ 1e-11` and seam ratio `≤ 2`.

2. **Seam transport orientation on QuadFolded**
   - File targets:
     - `src/BoundaryConditions/fill_halo_regions*.jl`
     - `src/BoundaryConditions/field_boundary_conditions.jl`
   - Requirement:
     complete scalar + vector + flux (transport) fold transforms for all halo operations.
   - Metrics:
     zero mismatch for seam/identity maps at `H = 1,2,3,6`, all test locations.

3. **Centered vector-invariant coupling**
   - File targets:
     - `src/Advection/vector_invariant_advection.jl`
     - `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_tendency_kernel_functions.jl`
   - Requirement:
     `compute_hydrostatic_free_surface_Gu!/Gv!` must consume non-orthogonal centered rotational blocks and energy-consistent reconstruction paths on `SphericalShellGrid`.
   - Metrics:
     split identity and orthogonal-limit residual `≤ 1e-12`.

4. **Non-orthogonal scalar advection benchmark**
   - File target:
     - `test/test_spherical_shell_grid.jl`
   - Requirement:
     expand from one-step checks to sustained cosine-bell and front-like runs.
   - Metrics:
     cosine-bell `ε_mass ≤ 1e-11`, `ε_L2 ≤ 5e-2` at release resolution (`N = 128`) after 100 rotations.

### Execution matrix for the requested feature set (2026-05-23)

Use this matrix as the practical implementation order for the requested goal:

1. **New grid + new octahealpix mapping**
   - Success criteria: `SphericalShellGrid`/`OctaHEALPixMapping` construction, topology, area closure, ring↔matrix bijections, and seam tables pass exactness checks.
2. **Non-orthogonal operators**
   - Success criteria: metric/Hodge kernels pass free-stream, adjointness, and orthogonal-limit gates with non-divergent tests.
3. **`model.transport_velocities` separation**
   - Success criteria: transport diagnostics are computed from model velocities on non-orthogonal grids, not copied; tracer and continuity paths remain coupled.
4. **QuadFolded transport orientation**
   - Success criteria: scalar + vector + transport halo semantics are verified for `H = 1,2,3,6` and all fold edge/corner tensor blocks.
5. **Tracer advection validation**
   - Success criteria: cosine-bell + front transport runs produce bounded mass drift, L2 error, and seam-ratio targets.
6. **Linear momentum validation**
   - Success criteria: `advection = nothing` tendency-level mass/energy tests and low-mode balance pass without regressions.
7. **Centered nonlinear vector invariant**
   - Success criteria: split-identity and orthogonal-limit residuals in release-gate range; no seam-drift in solid-body diagnostics.
8. **WENO vector invariant**
   - Success criteria: WENO validates relative-vorticity reconstruction and contamination/convergence thresholds before removing the explicit error block.

### Deep-dive status vs current implementation

- `src/Grids/spherical_shell_grid.jl`: `SphericalShellGrid`, `EquiangularGnomonicCubedSpherePanel`, `OctaHEALPixMapping`, `OctaHEALPixConnectivity`, and metric storage are in-tree.
- `src/BoundaryConditions/field_boundary_conditions.jl`: `QuadFolded` BC defaults are wired, but transport-orientation tables are not yet complete for all edge/corner tensor blocks.
- `src/Advection/tracer_advection_operators.jl`, `src/Operators/divergence_operators.jl`: non-orthogonal transport divergences are routed; tracer advection still depends on full transport orientation at folds for high-order runs.
- `src/Operators/nonorthogonal_metric_operators.jl`, `src/Operators/Operators.jl`: operator seams are present for covariant-to-contravariant conversion and rotational terms.
- `src/Models/HydrostaticFreeSurfaceModels/...`: model validation now supports `advection = nothing` and centered `VectorInvariant`, while explicitly rejecting `WENOVectorInvariant`.
- `test/test_spherical_shell_grid.jl`: API/geometry/operator smoke checks exist; long-run tracer/wave benchmarks remain work to be added.

### Source-grounded execution call-outs (critical before PR merge)

1. `src/Fields/field.jl` contains a special non-orthogonal seam halo fill for `SphericalShellGrid`:
   - scalar-only identity copy is already implemented for neighbors from `connectivity.ring_to_minus/plus_*_neighbor`;
   - it currently does not dispatch to location-aware folding transforms for `Face`/`Flux`/`Tensor` fields.
2. `QuadFolded` fold handling is therefore correct only for scalar continuity at present.
3. `src/BoundaryConditions/fill_halo_regions_upivotzipper.jl` and `fill_halo_regions_fpivotzipper.jl` are tripolar-specific (`UZBC` / `FZBC`) and are not yet sufficient to express octahedral fold tensor blocks.

### Quantitative next-step contract (next merge window)

For the next implementation cycle, close these in order:

1. **Scalar + vector + flux seam map generator**
   - target: `src/Fields/field.jl` (or dedicated `fill_halo_kernels` path)
   - gate: unit tests show exact scalar identity for `H=1,2,3,6`, and zero unmapped edge/corner fold cases.
2. **Orientation-aware vector transport**
   - target: `src/Grids/spherical_shell_grid.jl` metadata + `fill_halo` helper
   - gate: max orientation/sign mismatch on vector and transport folds `≤ 1e-13`.
3. **Fold-validated tracer advection**
   - target: `test/test_spherical_shell_grid.jl`
   - gate: sustained non-orthogonal transport mass and seam metrics in the same limits as the quantitative plan (`R_J`, mass drift, orthogonal-limit sanity checks).

No WENO unlock should be attempted until this seam contract is complete and the non-orthogonal tracer campaign in `oceananigans_nonorthogonal_spherical_cgrid_quantitative_plan.md` reaches release phase.

## 1.2 Source-first status (this branch)

### Implemented

- `SphericalShellGrid`, `EquiangularGnomonicCubedSpherePanel`, `OctaHEALPixMapping`, and `OctaHEALPixConnectivity` are implemented.
- `SphericalShellMetrics` currently stores the required non-orthogonal tensor objects at `ccc/fcc/cfc` and retains compatibility `λ/φ/Δx/Δy/Az` arrays.
- `BoundaryConditions` and `Models` now route `WENOVectorInvariant` rejection and `advection = nothing`/centered `VectorInvariant` handling on `SphericalShellGrid`.
- A focused test suite `test/test_spherical_shell_grid.jl` covers geometry formulas, mapping bijections, scalar diagnostics, and elementary algebraic non-orthogonal operators.

### Not yet complete (must remain tracked)

- `QuadFoldZipperBoundaryCondition` does not yet encode vector/flux transport transforms for all edge/corner tensor blocks.
- `NonOrthogonalScalarAdvection` and centered full nonlinear vector-invariant tendency workflows are not yet fully closed.
- WENO vector-invariant (`NonOrthogonalWENOVectorInvariant`) is intentionally blocked pending transport/orientation and reconstruction validation.

### Immediate blocking thresholds (quantified)

Until each threshold is closed, do not move to the next phase:

1. `QuadFoldZipperBoundaryCondition`
   - Seam scalar and vector halo mismatch count is zero for `H = 1..3`.
   - Transport orientation/sign mismatches are below `1e-13` in analytic solid-body rotation fold tests.
   - Vector/tensor fold maps are complete for all edge/corner blocks of `u`, `v`, and transport components.

2. `NonOrthogonalScalarAdvection`
   - Multi-step cosine-bell scalar advection at `N=64` and `N=128` reaches mass drift and seam ratio gates in the master plan.
   - Constant/linear analytic advection telescoping identities pass with residuals within phase-3 release gates.

3. Centered nonlinear vector-invariant tendencies
   - Orthogonal-limit centered tendency residual `≤ 1e-12`.
   - Split identity residual `≤ 1e-12` on at least one nonlinear benchmark.
   - Solid-body non-orthogonal run mass drift within phase-5 release envelope.

4. WENO unlock
   - `small-Rossby` contamination ratio at `|f|/|ζ|=10^4` must be `≤ 1.25` before any WENO merge.
   - WENO smooth convergence away from folds `p ≥ 4.5` at release resolution.

### Quantitative guardrail

Before merging each follow-up PR, verify:

1. all gates in `oceananigans_nonorthogonal_spherical_cgrid_quantitative_plan.md` for the corresponding phase;
2. no root dependency changes for OctaHEALPix validation (`Project.toml` unchanged);
3. full CI regression pass for `OrthogonalSphericalShellGrid` behavior.

Key source facts:

1. `OrthogonalSphericalShellGrid` is a concrete subtype of `AbstractHorizontallyCurvilinearGrid`. It stores horizontal coordinate and metric arrays directly on the grid at `cc`, `fc`, `cf`, and `ff` locations, plus `radius` and `conformal_mapping`.
2. Generic operators in `src/Operators/spacings_and_areas_and_volumes.jl` dispatch on names like `Δxᶠᶜᵃ`, `Δyᶜᶠᵃ`, `Azᶜᶜᵃ`, and `volume`. A new grid should preserve these interfaces where they are still meaningful.
3. Non-orthogonal dynamics cannot be represented by only `Δx`, `Δy`, `Az`, and `volume`. The Hodge map from covariant velocity to conservative transport needs metric-tensor data and cross-location interpolation or energy-symmetric blocks.
4. `HydrostaticFreeSurfaceModel` already has `model.velocities` and `model.transport_velocities`; tracer tendencies already use `model.transport_velocities`. This is the right integration seam.
5. `materialize_advection(advection, grid)` is the existing hook for allocating grid-dependent scheme state. It receives a scheme and grid, not a model, so model-dependent diagnostics need a separate update call before tendencies.
6. Existing zipper boundary conditions are tripolar, north-edge, pivot-specific implementations. OctaHEALPix requires a general grid-resident connectivity table for every folded edge, C-grid location, halo depth, and vector/transport orientation transform.
7. `AGENTS.md` forbids root dependency churn. SpeedyWeather/RingGrids is a validation reference, not a dependency.
8. Folded topologies currently require explicit boundary-condition dispatch handling (e.g. `QuadFolded`) in generic defaults so prognostic and auxiliary BC defaults remain well-defined on non-orthogonal shells.
9. Folded topologies require an octant-aware connectivity + halo table model in generic defaults so prognostic and auxiliary BC defaults are well-defined on non-orthogonal shells.

10. Current branch status: `OctaHEALPixConnectivity` uses matrix-based row/column/quadrant index transforms and explicit neighbor tables; active tests enforce seam continuity and seam halo consistency. Remaining work is completion of the full physical orientation machinery (edge/corner tensors and vector-component transforms) for seam transports.

Therefore the practical path is a side-by-side `SphericalShellGrid` implementation first. It should not rename or replace `OrthogonalSphericalShellGrid` until compatibility, docs, and downstream impact are understood.

## 1.2 Branch status (2026-05-23)

### Completed in-tree

- Thin-shell `SphericalShellGrid`, `EquiangularGnomonicCubedSpherePanel`, `OctaHEALPixMapping`, and `OctaHEALPixConnectivity` are present and on-architecture/`Adapt` compatible.
- Grid-level geometry and metric APIs are consistent with existing `OrthogonalSphericalShellGrid` style (`λ`, `φ`, `Δx`, `Δy`, `Az`, `volume`, `nodes`), with additional non-orthogonal metric storage (`J`, `g_ij`, `g^ij`, Hodge rows).
- Early HFSM routing is active: centered vector-invariant momentum is accepted, `WENOVectorInvariant` is rejected with `ArgumentError`, `div_Uc` and `vertical_vorticity` are routed through non-orthogonal metric paths.
- Focused tests now include OctaHEALPix formula/bijection checks, gnomonic analytic metric validation, basic non-orthogonal tracer divergence/transport checks, and momentum tendency hook tests.

### In progress

- `QuadFoldZipperBoundaryCondition` transport/velocity orientation tables (edge/corner tensor blocks) and vector/tensor halo fill semantics.
- Full transport diagnostics lifetime for the non-orthogonal centered vector-invariant operator (mass transport fields and divergence outputs).
- WENO/vector-invariant extensions and multi-rotation tracer/wave benchmarks.

### 1.3 External verification workflow (no hard dependency)

For independent cross-checks against an existing `OctaHEALPix` implementation without adding dependencies:

1. Clone SpeedyWeather or RingGrids locally when needed:

```bash
git clone https://github.com/CliMA/SpeedyWeather.jl
cd SpeedyWeather.jl
```

2. Use its native grid construction helpers to emit:
   - ring order `ring_to_r/c/q` tables,
   - latitude-ring populations,
   - latitude/longitude node arrays,
   - `matrix↔ring`/`ring↔xy` bijections (or equivalent).

3. Compare outputs against Oceananigans reference helpers in
   - `test/test_spherical_shell_grid.jl` (`test_octahealpix_indexing`, `test_octahealpix_reference_formulas`).

4. Keep checks one-way and CI-light:
   - no SpeedyWeather/RingGrids entries in `Project.toml` deps,
   - no runtime coupling, only offline scripts/tests for validation.

This is already reflected in the quantitative plan: SpeedyWeather/RingGrids references are treated as validation-only formulas, not a dependency path.

## 2. Guiding principles

### 2.1 One grid name, mapping-dependent geometry

Use a single grid type:

```julia
SphericalShellGrid(; mapping, radius, z, size, halo, topology, metrics, ...)
```

The mapping determines whether the horizontal coordinates are orthogonal, conformal, equiangular/gnomonic, or otherwise.

Examples:

```julia
SphericalShellGrid(; mapping = ConformalCubedSpherePanel(), ...)
SphericalShellGrid(; mapping = EquiangularGnomonicCubedSpherePanel(), ...)
SphericalShellGrid(; mapping = LatitudeLongitudeShell(), ...)
```

The type should preserve Oceananigans' structured-grid idiom: fixed index directions, C-grid field locations, halo regions, architecture adaptation, and metric operators dispatched by grid type.

### 2.2 Model velocity is distinct from transport velocity

For model integration, we keep the distinction:

```julia
model.velocities
model.transport_velocities      # HFSM-style
transport_velocities(model)     # Breeze-style accessor
```

`model.velocities` are the model's velocity variables used by momentum tendencies, diagnostics, closures, forcing, and output. On non-orthogonal C-grids, these may be covariant velocity components or another dynamically natural velocity representation.

`transport_velocities` are the velocity-like fields used to advect scalars. On a non-orthogonal finite-volume grid, these should be area-normal transport velocities satisfying

\[
A_i u_\perp^i = J u^i,
\]

so that scalar advection computes the correct volume transport through coordinate faces.

The conservative integrated transports are

\[
\mathcal V^i = J u^i, \qquad \mathcal U^i = J \rho u^i,
\]

where \(\mathcal V^i\) is volume transport and \(\mathcal U^i\) is mass transport.

### 2.3 Verbose code names, mathematical notation in local operators

Persistent objects, struct fields, and public functions should use clear names:

```julia
covariant_velocities
contravariant_velocities
transport_velocities
volume_transport
mass_transport
mass_transport_divergence
kinetic_energy
relative_vorticity
planetary_vorticity
absolute_vorticity
```

Compact mathematical symbols should be reserved for small internal formula functions, for example:

```julia
ζ₁₂ᶠᶠᶜ(...)
Z₁₂ᶠᶠᶜ(...)
Dᵢ𝒰ⁱᶜᶜᶜ(...)
```

This keeps persistent state readable while allowing kernels to resemble the derivation.

## 3. Coordinate and location notation

Oceananigans-style staggered locations should be used throughout documentation and code comments:

| Location | Meaning | Typical field |
|---|---|---|
| `ccc` | tracer / cell center | scalar, density, pressure |
| `fcc` | first-direction face | \(u_1\), \(u^1\), \(\mathcal V^1\) |
| `cfc` | second-direction face | \(u_2\), \(u^2\), \(\mathcal V^2\) |
| `ccf` | third-direction face | \(u_3\), \(u^3\), \(\mathcal V^3\) |
| `ffc` | horizontal edge / vertical vorticity location | \(\zeta_{12}\), \(Z_{12}\) |
| `fcf` | first-vertical edge | \(\zeta_{13}\), \(Z_{13}\) |
| `cff` | second-vertical edge | \(\zeta_{23}\), \(Z_{23}\) |

For the covariant velocity one-form,

\[
u_1^{\mathrm{fcc}}, \qquad
u_2^{\mathrm{cfc}}, \qquad
u_3^{\mathrm{ccf}}.
\]

The metric/Hodge conversion from covariant components to transport components must be defined at the target transport locations.

## 4. Equiangular gnomonic cubed-sphere panel

The first non-orthogonal spherical-shell target should be a single equiangular gnomonic cubed-sphere panel. This is structured, analytic, and strongly exercises the off-diagonal metric terms.

On a reference panel, define

\[
\phi_x, \phi_y \in [-1,1], \qquad
\alpha = \frac{\pi}{4}\phi_x, \qquad
\beta  = \frac{\pi}{4}\phi_y,
\]

\[
p = \tan \alpha, \qquad q = \tan \beta, \qquad D = 1+p^2+q^2.
\]

The unit-sphere map is

\[
\widehat{\boldsymbol x}(\alpha,\beta)
=
\frac{1}{\sqrt D}
\begin{pmatrix} p \\ q \\ 1 \end{pmatrix}.
\]

The spherical-shell map is

\[
\boldsymbol x(\alpha,\beta,r)
= r \widehat{\boldsymbol x}(\alpha,\beta),
\]

with panel rotations used to obtain the other five cubed-sphere faces.

The analytic horizontal metric components are

\[
g_{\alpha\alpha}
=
r^2\frac{(1+p^2)^2(1+q^2)}{D^2},
\]

\[
g_{\beta\beta}
=
r^2\frac{(1+q^2)^2(1+p^2)}{D^2},
\]

\[
g_{\alpha\beta}
=
-r^2\frac{pq(1+p^2)(1+q^2)}{D^2}.
\]

Thus \(g_{\alpha\beta}\ne0\) away from symmetry lines. This makes the grid a good first test for non-orthogonal C-grid numerics.

For the thin-shell case,

\[
g_{\alpha r}=g_{\beta r}=0, \qquad g_{rr}=1.
\]

For deep-atmosphere support, horizontal metric components should retain their radius dependence at each vertical location.

## 4.1 OctaHEALPix mapping reference

The global target is `OctaHEALPixMapping(N)` stored as a `2N × 2N` rectangular matrix. SpeedyWeather/RingGrids' `OctaHEALPixGrid` provides a useful independent reference for the indexing and equal-area geometry.

Reference facts to reproduce in Oceananigans tests:

```julia
npoints = 4N^2
matrix_size = (2N, 2N)
nlat = 2N - 1
nlon_per_ring(j) = min(4j, 8N - 4j)
solid_angle = 4π / npoints
```

For northern rings `j = 1:N`,

\[
z_j = 1 - \left(\frac{j}{N}\right)^2,
\qquad
\varphi_j = \arcsin z_j,
\]

and southern rings are obtained by antisymmetry. Longitudes on northern ring `j` are

\[
\lambda_i = \frac{\pi}{2j}\left(i - \frac12\right),
\qquad i = 1,\ldots,4j.
\]

The runtime storage does not need to use RingGrids' ring order. But the following bijections should be tested against equivalent reference logic:

```julia
ring_index <-> (row, column, quadrant)
matrix_index <-> (row, column, quadrant)
ring_index <-> matrix_index
```

For power-of-two `N`, nested-order round trips are an optional validation aid. They are not required for Oceananigans runtime storage.

## 5. Proposed grid object

The grid should follow the concrete, field-explicit style of `OrthogonalSphericalShellGrid` so that kernels see concrete fields and existing operator dispatch can be reused. A nested `metrics` object is still useful, but the compatibility geometry should remain directly accessible through familiar names.

A source-aligned concrete sketch:

```julia
struct SphericalShellGrid{FT, TX, TY, TZ, Z, Mapping, Metrics, Connectivity,
                          CC, FC, CF, FF, Arch, FT2} <:
       AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Z, Arch}
    architecture :: Arch

    Nx :: Int
    Ny :: Int
    Nz :: Int

    Hx :: Int
    Hy :: Int
    Hz :: Int

    radius :: FT2
    z :: Z

    # Compatibility coordinates at horizontal C-grid locations.
    λᶜᶜᵃ :: CC
    λᶠᶜᵃ :: FC
    λᶜᶠᵃ :: CF
    λᶠᶠᵃ :: FF
    φᶜᶜᵃ :: CC
    φᶠᶜᵃ :: FC
    φᶜᶠᵃ :: CF
    φᶠᶠᵃ :: FF

    # Compatibility metric arrays used by existing generic operators.
    Δxᶜᶜᵃ :: CC
    Δxᶠᶜᵃ :: FC
    Δxᶜᶠᵃ :: CF
    Δxᶠᶠᵃ :: FF
    Δyᶜᶜᵃ :: CC
    Δyᶠᶜᵃ :: FC
    Δyᶜᶠᵃ :: CF
    Δyᶠᶠᵃ :: FF
    Azᶜᶜᵃ :: CC
    Azᶠᶜᵃ :: FC
    Azᶜᶠᵃ :: CF
    Azᶠᶠᵃ :: FF

    mapping :: Mapping
    metrics :: Metrics
    connectivity :: Connectivity
end
```

The grid type should support:

```julia
architecture(grid)
topology(grid)
size(grid)
halo_size(grid)
with_halo(grid)
on_architecture(arch, grid)
Adapt.adapt_structure(to, grid)
```

and standard Oceananigans node/spacing/area/volume metric operator dispatch.

### 5.1 Migration rule

The first implementation should add `SphericalShellGrid` without changing `OrthogonalSphericalShellGrid`. Long term, the two can share constructors and helper functions, or `OrthogonalSphericalShellGrid` can become a mapping-restricted alias if that is still desirable. The quantitative gate for the first PR is simpler: all existing `OrthogonalSphericalShellGrid` tests must pass unchanged.

## 6. Proposed metrics object

The metrics object should store or lazily evaluate geometry at the locations where operators need it.

```julia
struct SphericalShellMetrics{Coordinates, Jacobians, Areas,
                             CovariantMetric, ContravariantMetric,
                             HodgeMap, EdgeGeometry}
    coordinates :: Coordinates
    jacobians :: Jacobians
    areas :: Areas
    covariant_metric :: CovariantMetric
    contravariant_metric :: ContravariantMetric
    hodge_map :: HodgeMap
    edge_geometry :: EdgeGeometry
end
```

This object is for non-orthogonal tensor and Hodge data. It does not replace the directly stored compatibility arrays (`λ`, `φ`, `Δx`, `Δy`, `Az`) unless and until generic Oceananigans operators are refactored to consume a metric object.

### 6.1 Coordinates

The coordinates may include both computational and physical/spherical representations:

```julia
coordinates = (; 
    alpha, beta, radius,
    longitude, latitude,
    cartesian_x, cartesian_y, cartesian_z,
)
```

At minimum, the grid should be able to provide coordinates at:

```text
ccc, fcc, cfc, ccf, ffc, fcf, cff, fff
```

as needed by C-grid operators and metric tests.

For the first thin-shell implementation, horizontal coordinates and metrics are 2D and independent of `k`; vertical coordinates still use Oceananigans' existing `z` discretization. Deep-shell support should add radius-dependent horizontal metrics without changing the public C-grid location conventions.

### 6.2 Metric placement

For robust staggered numerics, metric quantities should live where the operator evaluates them. The first-direction transport lives at `fcc`, so the quantities needed for \(\mathcal V^{1,\mathrm{fcc}}\) should be stored or evaluated at `fcc`.

For the horizontally non-orthogonal thin-shell case, the most important target-location metric data are:

```julia
contravariant_metric.g11_fcc
contravariant_metric.g12_fcc
contravariant_metric.g21_cfc
contravariant_metric.g22_cfc
jacobians.J_fcc
jacobians.J_cfc
areas.face_area_1_fcc
areas.face_area_2_cfc
```

More generally, the Hodge map should use

\[
G^{ij} = J g^{ij},
\]

at the appropriate staggered locations.

### 6.3 Connectivity and halo maps

`QuadFolded` should be a singleton topology tag, but the actual connectivity must be data on the grid:

```julia
struct QuadFoldConnectivity{ScalarMap, VelocityMap, TransportMap, MetricMap}
    scalar_halo_map :: ScalarMap
    velocity_halo_map :: VelocityMap
    transport_halo_map :: TransportMap
    metric_halo_map :: MetricMap
end
```

Each map entry should encode enough information for an allocation-free kernel:

```julia
(destination_i, destination_j,
 source_i, source_j,
 component_transform,
 orientation_sign)
```

The map must be location-specific because `ccc`, `fcc`, `cfc`, and `ffc` fold differently. It must also be halo-depth aware because high-order WENO requires wide halos. This is a generalization of the current tripolar `FPivotZipperBoundaryCondition` and `UPivotZipperBoundaryCondition`, which are useful precedents but not general enough for OctaHEALPix.

## 7. Staggered Hodge-map design

The central non-orthogonal C-grid operation is the Hodge map from covariant velocity to volume transport:

\[
\begin{pmatrix}
\mathcal V^1\\
\mathcal V^2\\
\mathcal V^3
\end{pmatrix}
=
\mathsf H
\begin{pmatrix}
u_1\\u_2\\u_3
\end{pmatrix}.
\]

At the first-direction transport location,

\[
\mathcal V^{1,\mathrm{fcc}}
=
\mathsf H^{11}u_1^{\mathrm{fcc}}
+
\mathsf H^{12}u_2^{\mathrm{cfc}}
+
\mathsf H^{13}u_3^{\mathrm{ccf}}.
\]

We should not assume a priori that one interpolation ordering is correct. Instead, we should implement and test several candidate Hodge maps.

### 7.1 Candidate A: target-metric Hodge

\[
\mathsf H^{12}_{\mathrm{target}} u_2
=
G^{12,\mathrm{fcc}}
\mathcal I_{\mathrm{cfc}\to\mathrm{fcc}} u_2.
\]

This evaluates the metric at the target transport location and interpolates the source velocity component there.

### 7.2 Candidate B: product-interpolated Hodge

\[
\mathsf H^{12}_{\mathrm{product}} u_2
=
\mathcal I_{\mathrm{cfc}\to\mathrm{fcc}}
\left(G^{12,\mathrm{cfc}}u_2^{\mathrm{cfc}}\right).
\]

This forms the metric-weighted product at the source location and interpolates the product to the target location.

### 7.3 Candidate C: energy-symmetric Hodge

Choose a local energy quadrature:

\[
E_h
=
\frac12\sum_q w_q G_q^{ij}
\left(\mathcal I_i^q u_i\right)
\left(\mathcal I_j^q u_j\right).
\]

Then define the Hodge map by differentiating the discrete kinetic energy:

\[
\mathcal V_i
=
W_i^{-1}\frac{\partial E_h}{\partial u_i},
\]

where \(W_i\) is the velocity-location quadrature/inner-product weight. This candidate is theoretically strongest because it guarantees a symmetric positive kinetic-energy form if the metric tensor and quadrature are positive.

## 8. Lessons from TRiSK

TRiSK does not answer the metric-staggering question by choosing “multiply first” or “interpolate first.” It constructs a metric-dependent map between staggered flux spaces and constrains the map by mimetic identities.

The key TRiSK operation maps normal edge fluxes to perpendicular/tangential fluxes:

\[
F_e^\perp
=
\frac{1}{d_e}
\sum_{e'\in ECP(e)}
w_{e,e'}\,l_{e'}F_{e'}.
\]

This is a flux/Hodge map, not a casual interpolation. The map is chosen so that a dual divergence of the mapped flux is an interpolation of the primal divergence:

\[
D_{\mathrm{dual}} M(F)
=
I\left(D_{\mathrm{primal}}F\right).
\]

This teaches the following design rule for `SphericalShellGrid`:

\[
\boxed{
\text{Choose staggered metric/Hodge maps by compatibility constraints, not by local taste.}
}
\]

The relevant compatibility targets are:

1. **free-stream preservation**: constant physical velocity should produce zero transport divergence;
2. **weighted adjointness**: off-diagonal Hodge blocks should be paired adjoints;
3. **positive kinetic energy**: the Hodge map should define a positive quadratic form;
4. **commuting transport/divergence behavior**: rotated or dual transports should not introduce divergence inconsistency;
5. **energy neutrality** of centered vorticity/Coriolis fluxes;
6. **PV consistency** when a mass measure collocated with absolute vorticity is defined.

TRiSK is originally built around locally orthogonal primal-dual meshes. Our equiangular gnomonic panel is non-orthogonal in the coordinate basis, so we cannot copy TRiSK formulas directly. But the operator-design philosophy is exactly the right precedent.

## 9. Transport and scalar advection

The grid/Hodge layer should support both volume and mass transport:

\[
\mathcal V^i = J u^i,
\qquad
\mathcal U^i = J\rho u^i.
\]

The volume-transport divergence is

\[
\mathcal D_{\mathcal V} = D_i\mathcal V^i,
\]

and the mass-transport divergence is

\[
\mathcal D_{\mathcal U} = D_i\mathcal U^i.
\]

For fixed-grid incompressible/hydrostatic Boussinesq dynamics,

\[
\mathcal D_{\mathcal V}=0.
\]

For fully compressible dynamics,

\[
\partial_t(J\rho)+\mathcal D_{\mathcal U}=0,
\]

so \(\mathcal D_{\mathcal U}\ne0\) in general.

For scalar advection, the preferred long-term operator is flux-based:

\[
\partial_t(J\rho c)
=
-D_i(\mathcal U^i c^{\uparrow}).
\]

However, Oceananigans HFSM already uses `transport_velocities`; thus a minimally invasive path is to diagnose `transport_velocities` such that existing scalar kernels compute the correct flux.

## 10. Relationship to momentum advection

The non-orthogonal vector-invariant scheme should use verbose, persistent fields:

```julia
struct NonOrthogonalVectorInvariant{...} <: AbstractAdvectionScheme{...}
    # reconstruction choices
    vorticity_reconstruction
    shear_vorticity_reconstruction
    planetary_vorticity_reconstruction
    divergence_reconstruction
    kinetic_energy_gradient_reconstruction
    transport_reconstruction
    upwinding

    # materialized diagnostic fields
    covariant_velocities
    contravariant_velocities
    transport_velocities
    volume_transport
    mass_transport
    mass_transport_divergence
    kinetic_energy
    relative_vorticity
    planetary_vorticity
    absolute_vorticity
end
```

The generic object is `NonOrthogonalVectorInvariant`. `NonOrthogonalWENOVectorInvariant` should be a convenience constructor that fills the generic object with WENO reconstructions.

Relative vorticity should be reconstructed with WENO, while planetary vorticity should be reconstructed smoothly or centered:

\[
\widehat Z_{ij}
=
\mathcal R_{\mathrm{WENO}}[\zeta_{ij}]
+
\mathcal R_{\mathrm{smooth}}[f_{ij}],
\]

where

\[
\zeta_{ij}=D_i u_j-D_j u_i,
\qquad
Z_{ij}=\zeta_{ij}+f_{ij}.
\]

This prevents smooth planetary vorticity from dominating the WENO smoothness indicators at small Rossby number.

## 11. Oceananigans integration points

### 11.1 Grid module

Add `SphericalShellGrid`, `SphericalShellMetrics`, mappings, and `QuadFolded` to `src/Grids` first. Export public names intentionally from `Grids.jl` and then from `Oceananigans.jl` if they are part of the user API. Source imports must be explicit.

Required methods:

```julia
architecture(grid)
topology(grid)
size(grid)
halo_size(grid)
nodes(grid, loc...)
λnode, φnode, xnode, ynode, znode
Az, volume
on_architecture(arch, grid)
Adapt.adapt_structure(to, grid)
```

### 11.2 Boundary conditions

Default field boundary conditions for `QuadFolded` directions should regularize to `QuadFoldZipperBoundaryCondition` for scalar-like fields, covariant velocity fields, transport fields, and vorticity-like edge fields. The implementation should use the grid-resident halo maps and follow the existing `fill_halo_regions!` kernel pattern.

### 11.3 Transport velocities and scalar advection

For `HydrostaticFreeSurfaceModel`, the least invasive path is:

```julia
model.velocities             # covariant model velocities
model.transport_velocities   # diagnosed area-normal transport velocities
```

Existing tracer tendencies call `div_Uc(..., model.transport_velocities, c)`. This works if `transport_velocities` are area-normal fields such that `Ax * u_transport` and `Ay * v_transport` equal the desired volume transports. The more robust long-term path is a `NonOrthogonalScalarAdvection` specialization that uses `volume_transport` and `mass_transport` directly.

### 11.4 Materialized momentum advection

`materialize_advection(advection, grid)` should allocate the diagnostic fields for `NonOrthogonalVectorInvariant`, but it cannot update them because it has no model. Add an explicit update hook before tendency kernels:

```julia
update_vector_invariant_diagnostics!(model.advection.momentum, model)
```

Kernels called by this update should receive fields, grid, density, and coriolis data. They should not receive the full model object.

### 11.5 HFSM staging

Initial validation should use HFSM with `Nz = 1`, fixed vertical coordinates, and either `advection = nothing` or centered non-orthogonal vector-invariant advection. Split-explicit free-surface and z-star support should remain out of scope until fixed-grid transport, pressure-gradient, and momentum gates pass.

## 12. Numerical test campaign for staggered Hodge maps

The Hodge-map design should be selected empirically by a formal test campaign.

### 12.1 Metric and geometry tests

Test analytic metric consistency:

\[
g_{ij}g^{jk}\approx\delta_i^k,
\qquad
g_{11}g_{22}-g_{12}^2>0.
\]

Test spherical area closure on one panel:

\[
\sum_{\mathrm{panel}} A
\approx
\frac{4\pi R^2}{6}.
\]

Test all required staggered metric locations: `ccc`, `fcc`, `cfc`, `ccf`, `ffc`, `fcf`, `cff`.

### 12.2 Hodge consistency tests

Given a known Cartesian vector field \(\boldsymbol u\), compute analytic covariant components

\[
u_i = \boldsymbol u\cdot\boldsymbol a_i,
\]

apply the candidate Hodge map, and compare with analytic volume transport:

\[
\mathcal V^i_{\mathrm{exact}} = J u^i.
\]

Fields to test:

1. constant Cartesian velocity,
2. solid-body rotation,
3. smooth manufactured vector field.

### 12.3 Weighted adjointness

For off-diagonal blocks, measure

\[
\epsilon_{\mathrm{adj}}
=
\frac{
\left\|W_1\mathsf H^{12}-(W_2\mathsf H^{21})^T\right\|
}{
\left\|W_1\mathsf H^{12}\right\|+
\left\|(W_2\mathsf H^{21})^T\right\|
}.
\]

The energy-symmetric candidate should be roundoff-small. Simpler candidates should either be roundoff-small or convergent with grid refinement.

### 12.4 Positive kinetic energy

Build

\[
E_h = \frac12 u^T W\mathsf H u.
\]

The symmetric part

\[
\frac12(W\mathsf H+\mathsf H^TW)
\]

must be positive definite on non-degenerate grids.

### 12.5 Free-stream preservation

For a constant Cartesian velocity, compute staggered covariant components, apply the Hodge map, then compute

\[
D_i\mathcal V^i.
\]

A uniform physical velocity must not produce spurious divergence. This is a red-line test.

### 12.6 Orthogonal limit

On a family of grids with \(g^{12}\to0\), verify that the non-orthogonal Hodge map reduces to the orthogonal formulas and agrees with existing Oceananigans operators at \(g^{12}=0\).

### 12.7 Linear mode tests

Use a periodic affine skew grid and a smoothly varying skew grid. Linearize simple shallow-water or compressible acoustic systems and inspect eigenvalues. The acceptable scheme should not exhibit growing grid-scale modes or checkerboard branches.

### 12.8 Centered conservation tests

With centered non-orthogonal vector-invariant advection and no forcing/diffusion, test:

1. tracer mass conservation,
2. tracer variance behavior,
3. kinetic-energy conservation or controlled energy residual,
4. vorticity/PV diagnostic consistency.

### 12.9 WENO tests

After centered tests pass, enable WENO and test:

1. WENO reconstruction of relative vorticity rather than absolute vorticity,
2. smooth-solution convergence,
3. vorticity filament robustness,
4. grid-noise spectra in decaying 2D turbulence.

## 13. Decision tree for Hodge-map selection

Use the simplest Hodge map that passes the essential tests.

1. If `TargetMetricHodge` passes free-stream preservation, positive energy, weighted adjointness, and orthogonal-limit tests, use it.
2. If `TargetMetricHodge` fails free-stream preservation but is otherwise stable, improve metric generation or use conservative metric identities.
3. If `ProductInterpolatedHodge` passes free-stream preservation but fails adjointness or positivity, use it only as a starting point and build the paired block by weighted transpose.
4. If neither simple candidate passes, use `EnergySymmetricHodge`.
5. If `EnergySymmetricHodge` is too expensive, derive a mass-lumped approximation and rerun the entire test suite.

The final choice should be documented with a test scorecard.

## 14. Suggested test layout

```text
test/Grids/test_spherical_shell_grid_metrics.jl
test/Grids/test_equiangular_gnomonic_panel.jl

test/Operators/test_hodge_consistency.jl
test/Operators/test_hodge_adjointness.jl
test/Operators/test_hodge_positivity.jl
test/Operators/test_free_stream_preservation.jl
test/Operators/test_orthogonal_limit.jl
test/Operators/test_metric_identity.jl

test/Advection/test_nonorthogonal_scalar_transport.jl
test/Advection/test_nonorthogonal_vector_invariant_centered.jl
test/Advection/test_nonorthogonal_weno_vorticity_reconstruction.jl
test/Advection/test_hydrostatic_split_identity.jl

test/Models/test_breeze_nonorthogonal_compressible.jl
test/Models/test_hfsm_nonorthogonal_rigid_lid.jl
```

## 15. Implementation roadmap

### PR 1: grid skeleton and source interfaces

Implement:

```julia
SphericalShellGrid
SphericalShellMetrics
AbstractSphericalShellMapping
QuadFolded
```

with `on_architecture`, `Adapt`, `nodes`, `Az`, `volume`, and explicit exports. Existing `OrthogonalSphericalShellGrid` tests must pass unchanged.

### PR 2: analytic panel mapping

Implement:

```julia
EquiangularGnomonicCubedSpherePanel
```

with analytic metrics at staggered locations and geometry tests.

### PR 3: OctaHEALPix mapping and reference tests

Implement:

```julia
OctaHEALPixMapping
```

with `2N × 2N` storage, equal areas, and SpeedyWeather/RingGrids reference checks for `npoints`, ring populations, latitudes, longitudes, and matrix-order bijections.

### PR 4: quad-fold connectivity and halos

Implement grid-resident scalar, velocity, transport, and metric halo maps plus `QuadFoldZipperBoundaryCondition`.

### PR 5: Hodge-map candidates

Implement:

```julia
AbstractSphericalShellHodgeMap
TargetMetricHodge
ProductInterpolatedHodge
EnergySymmetricHodge
```

and all pure-operator tests.

### PR 6: transport diagnostics

Implement:

```julia
compute_covariant_velocities!
compute_contravariant_velocities!
compute_transport_velocities!
compute_volume_transport!
compute_mass_transport!
compute_mass_transport_divergence!
```

### PR 7: scalar transport and centered vector-invariant tests

Implement non-orthogonal scalar advection and centered non-orthogonal vector-invariant diagnostics. Do not enable WENO until centered tests pass.

### PR 8: WENO vector-invariant construction

Implement:

```julia
NonOrthogonalVectorInvariant
NonOrthogonalWENOVectorInvariant
NonOrthogonalScalarAdvection
```

with WENO acting on relative vorticity.

### PR 9: Breeze first full model integration

Integrate with Breeze compressible dynamics first because it avoids HFSM free-surface coupling.

### PR 10: HFSM staged integration

Start with rigid-lid/fixed-grid HFSM. Add free-surface dynamics only after fixed-grid transport and momentum tests pass.

## 16. Open risks

### 16.1 Metric/Hodge instability

Risk: simple staggered metric choices create negative-energy or grid-scale modes.

Mitigation: Hodge positivity, adjointness, free-stream, and linear-mode tests before nonlinear simulations.

### 16.2 Free-surface coupling

Risk: HFSM free-surface code assumes horizontal transports behave like orthogonal-grid velocities.

Mitigation: defer split-explicit and z-star support until fixed-grid transport is verified.

### 16.3 WENO smoothness contamination by planetary vorticity

Risk: reconstructing \(Z=\zeta+f\) causes smooth \(f\) to dominate WENO smoothness indicators.

Mitigation: WENO reconstruct \(\zeta\), add smooth reconstruction of \(f\).

### 16.4 Cost of energy-symmetric Hodge

Risk: energy-symmetric Hodge is robust but too expensive.

Mitigation: implement it as a reference candidate, then derive and test mass-lumped simplifications.

## 17. Summary

`SphericalShellGrid` should be a single structured spherical-shell grid family whose mapping and metrics determine whether the grid is orthogonal or non-orthogonal. The equiangular gnomonic cubed-sphere panel is the first analytic non-orthogonal target; `OctaHEALPixMapping` is the first global target, validated against SpeedyWeather/RingGrids formulas without adding a dependency.

The central numerical problem is the staggered Hodge map from covariant C-grid velocities to volume and mass transports. This map should be selected by compatibility properties and empirical verification, not by an arbitrary choice between multiplying metrics before or after interpolation.

The development program should therefore prioritize source-compatible grid infrastructure, geometry, Hodge, and transport tests before full dynamics. Implement `SphericalShellGrid` side by side with the existing `OrthogonalSphericalShellGrid`, use HFSM's existing `transport_velocities` seam for early validation, and defer free-surface and z-star coupling until fixed-grid non-orthogonal transport is mature.
