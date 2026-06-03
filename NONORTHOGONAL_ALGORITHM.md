# Non-orthogonal staggered C-grid algorithm for `SphericalShellGrid`

This document specifies the discrete operators that make
`SphericalShellGrid` work as a finite-volume C-grid on a non-orthogonal
horizontal mapping (currently `EquiangularGnomonicCubedSpherePanel` and
`OctaHEALPixMapping`). It is the implementation contract: it states what
each operator computes, what its inputs and outputs are, and which
identities the discrete form is required to satisfy.

It is not the design document — `spherical_shell_grid_design.md` lays
out the higher-level choices. This document is the algorithm.

---

## 1. Continuous setup

The horizontal mapping
\(\mathbf{x}(\xi^1,\xi^2,r) = r\,\hat{\mathbf{x}}(\xi^1,\xi^2)\)
defines a thin spherical shell. Define:

- Covariant basis vectors \(\mathbf{a}_i = \partial\mathbf{x}/\partial\xi^i\).
- Covariant metric tensor \(g_{ij} = \mathbf{a}_i\cdot\mathbf{a}_j\).
- Contravariant metric \(g^{ij}\) such that \(g_{ik}g^{kj} = \delta_i^j\).
- Jacobian \(J = \sqrt{\det g_{ij}}\) (in the thin-shell limit
  \(g_{ij}\) is the 2D horizontal metric only).
- Density-weighted inverse metric \(G^{ij} \equiv J\,g^{ij}\).

A velocity field has three equivalent representations:

| Representation        | Symbol         | Live at           |
|-----------------------|----------------|-------------------|
| Covariant             | \(u_i\)        | fcc, cfc, ccf    |
| Contravariant         | \(u^i\)        | fcc, cfc, ccf    |
| Cartesian             | \(\mathbf{u}\) | ccc (diagnostic)  |

with \(u^i = g^{ij}u_j\) and \(\mathbf{u} = u^i\mathbf{a}_i\).

The conservative *volume transport* is \(\mathcal V^i = J u^i = G^{ij}u_j\).
The mass-transport divergence
\(D_i\mathcal V^i\) governs the continuity equation.

For the **thin shell**, vertical and horizontal directions are
orthogonal: \(g_{i3} = 0\) and \(g_{33} = 1\). All vertical operators
reduce to standard Oceananigans z-direction kernels and are not discussed
further here.

---

## 2. Grid storage

`SphericalShellGrid` keeps two parallel sets of horizontal arrays:

**Compatibility (used by generic Oceananigans operators):**

```
λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ      # longitudes at each C-grid location
φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ      # latitudes
Δxᶜᶜᵃ, Δxᶠᶜᵃ, …             # spacings (sqrt(Az) on equal-area grids)
Δyᶜᶜᵃ, …
Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ # cell areas
```

**Non-orthogonal metrics (`SphericalShellMetrics`):**

```
Jᶜᶜᵃ, Jᶠᶜᵃ, Jᶜᶠᵃ
g₁₁ᶜᶜᵃ, g₁₂ᶜᶜᵃ, g₂₂ᶜᶜᵃ
g¹¹ᶜᶜᵃ, g¹²ᶜᶜᵃ, g²²ᶜᶜᵃ
G¹¹ᶜᶜᵃ, G¹²ᶜᶜᵃ, G²²ᶜᶜᵃ        # G^ij at center
g¹¹ᶠᶜᵃ, g¹²ᶠᶜᵃ                # at fcc (target for 𝒱¹)
G¹¹ᶠᶜᵃ, G¹²ᶠᶜᵃ
g²¹ᶜᶠᵃ, g²²ᶜᶠᵃ                # at cfc (target for 𝒱²)
G²¹ᶜᶠᵃ, G²²ᶜᶠᵃ
eλxᶜᶜᵃ, eλyᶜᶜᵃ, eλzᶜᶜᵃ        # local east basis (Cartesian)
eφxᶜᶜᵃ, eφyᶜᶜᵃ, eφzᶜᶜᵃ        # local north basis
erxᶜᶜᵃ, eryᶜᶜᵃ, erzᶜᶜᵃ        # local up basis
xᶜᶜᵃ, yᶜᶜᵃ, zᶜᶜᵃ              # Cartesian cell-center positions
```

The compatibility arrays are written by
`fill_*_geometry!` and consumed by generic Oceananigans operators that
dispatch on `Δx*`, `Δy*`, `Az`, `volume`. They are equal-area on
OctaHEALPix and analytic on the gnomonic panel.

The metric arrays are written by `fill_spherical_shell_metrics!` and
consumed by the non-orthogonal kernels in
`src/Operators/nonorthogonal_metric_operators.jl`.

---

## 3. Hodge map: covariant → volume transport

### 3.1 Continuous

\[
\mathcal V^i \;=\; G^{ij}\,u_j,
\quad G^{ij} = J\,g^{ij}.
\]

### 3.2 Staggered discrete (Candidate A — target-metric Hodge)

The first-direction transport lives at fcc and the second-direction at
cfc. The implemented Hodge map is:

\[
\mathcal V^{1,\,\mathrm{fcc}}_{i,j,k}
\;=\;
G^{11}\!\big|_{\mathrm{fcc}}\, u_1^{\mathrm{fcc}}
\;+\;
G^{12}\!\big|_{\mathrm{fcc}} \cdot
\overline{u_2^{\mathrm{cfc}}}^{\,\mathrm{cfc}\to\mathrm{fcc}}_{i,j,k}
\]

\[
\mathcal V^{2,\,\mathrm{cfc}}_{i,j,k}
\;=\;
G^{22}\!\big|_{\mathrm{cfc}}\, u_2^{\mathrm{cfc}}
\;+\;
G^{21}\!\big|_{\mathrm{cfc}} \cdot
\overline{u_1^{\mathrm{fcc}}}^{\,\mathrm{fcc}\to\mathrm{cfc}}_{i,j,k}
\]

The interpolation \(\overline{\,\cdot\,}^{\,\mathrm{cfc}\to\mathrm{fcc}}\)
is a 4-point average of the four cfc-located neighbors of the target fcc
location:
\(\tfrac14\big(u_2[i{-}1,j] + u_2[i,j] + u_2[i{-}1,j{+}1] + u_2[i,j{+}1]\big)\).

Code:
- `covariant_to_volume_flux_uᶠᶜᶜ(i, j, k, grid, u, v)` — returns
  \(\mathcal V^{1,\mathrm{fcc}}\).
- `covariant_to_volume_flux_vᶜᶠᶜ(i, j, k, grid, u, v)` — returns
  \(\mathcal V^{2,\mathrm{cfc}}\).

### 3.3 Identities the discrete Hodge must satisfy

1. **Free-stream preservation.** For \(u_i = \hat{\mathbf{e}}\cdot\mathbf{a}_i\)
   with \(\hat{\mathbf e}\) a constant Cartesian field, the discrete
   horizontal divergence
   \(D_i\mathcal V^i_{i,j,k}\) must satisfy
   \(\max|D_i\mathcal V^i|\le 10^3\varepsilon\cdot\|A\hat e\|\).
   Pinned by `test_equiangular_gnomonic_nonzero_transport_conservation`.

2. **Orthogonal limit.** On a grid with \(g^{12}\equiv 0\), the
   off-diagonal term must vanish:
   \(\max|G^{12}|_\mathrm{fcc}\le 10^3\varepsilon\) ⇒
   \(\max|H^{12}\,u_2|\le 10^3\varepsilon\cdot\|u\|\).
   Implicit in metric tensor identity tests.

### 3.4 Identities the discrete Hodge **does not yet** satisfy

3. **Weighted adjointness** — convergent at \(O(h^2)\), not roundoff.
4. **Positive-definite KE form** — eigenvalues unverified.

If criteria 3 or 4 fail in tests, Candidate A must be replaced; see the
plan audit in `RUNNING_REVIEW.md` (Closure 1).

---

## 4. Discrete divergence and continuity

\[
D_i\mathcal V^i\big|_{\mathrm{ccc}}^{i,j,k}
\;=\;
\frac{\mathcal V^{1}_{i+1,j,k} - \mathcal V^{1}_{i,j,k}}{V_{\mathrm{ccc}}^{i,j,k}}
\;+\;
\frac{\mathcal V^{2}_{i,j+1,k} - \mathcal V^{2}_{i,j,k}}{V_{\mathrm{ccc}}^{i,j,k}}
\;+\;\text{(vertical term)}
\]

Code: `horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, k, grid, u, v)`.

The continuity equation for incompressible Boussinesq dynamics requires
\(D_i\mathcal V^i = 0\) pointwise after the pressure projection. The
vertical velocity \(w\) is diagnosed from this constraint by
`update_vertical_velocities!` integrating upward from the bottom.

---

## 5. Kinetic energy

\[
K^{\mathrm{ccc}}_{i,j,k} \;=\;
\tfrac12 \sum_{i',j' \in \{1,2\}}
g^{i'j'}\!\big|_{\mathrm{ccc}}\;
\overline{u_{i'}}\;\overline{u_{j'}},
\]
where the bar denotes interpolation from the velocity location to ccc
(2-point average in the appropriate direction). The two cross terms are
identical so the implementation collects them as
\(2 g^{12} \overline{u_1}\overline{u_2}\).

Code: `covariant_kinetic_energyᶜᶜᶜ(i, j, k, grid, u, v)`.

---

## 6. Bernoulli head (KE gradient)

\[
\partial_i K \;=\; \frac{K^{\mathrm{ccc}}_{i,j,k} - K^{\mathrm{ccc}}_{i-1,j,k}}{\Delta\xi^1}
\quad\text{at fcc,}
\]
and similarly for \(\partial_2 K\) at cfc. The Δξ here is the
**computational coordinate** spacing, not a Cartesian distance.

Code: `covariant_bernoulli_head_uᶠᶜᶜ`, `covariant_bernoulli_head_vᶜᶠᶜ`.

---

## 7. Vorticity and rotational advection

### 7.1 Discrete vertical vorticity component

\[
\zeta_{12}^{\mathrm{ffc}}_{i,j,k}
\;=\;
\frac{u_2[i,j,k] - u_2[i-1,j,k]}{\Delta\xi^1}
\;-\;
\frac{u_1[i,j,k] - u_1[i,j-1,k]}{\Delta\xi^2}.
\]
This is the *covariant* vorticity component. The physical vorticity is
\(\zeta^{\mathrm{phys}} = \zeta_{12}/J\). The discrete *circulation* is
\(\Gamma^{\mathrm{ffc}} = \Delta\xi^1\Delta\xi^2 \zeta_{12}\).

Code: `covariant_vertical_vorticity_componentᶠᶠᶜ`,
`covariant_vertical_circulationᶠᶠᶜ`,
`covariant_vertical_vorticityᶠᶠᶜ` (returns \(\Gamma / A_z\)).

### 7.2 Rotational advection

In the vector-invariant formulation, the horizontal momentum advection
splits into a rotational and a gradient piece. For the rotational part:

\[
\big(\mathbf{u}\cdot\nabla\big)_1 u
\;\sim\;
-\,\overline{\zeta_{12}}^{\,\mathrm{ffc}\to\mathrm{fcc}}\cdot
\overline{\mathcal V^{2}}^{\,\mathrm{cfc}\to\mathrm{fcc}}
\quad\text{at fcc,}
\]
with the analogous expression at cfc. The Coriolis term \(f\) is added
to \(\zeta_{12}\) before the multiplication to get the absolute vorticity.

Code: `covariant_rotational_advection_uᶠᶜᶜ`, `covariant_rotational_advection_vᶜᶠᶜ`.

### 7.3 Hooked into `VectorInvariant` advection

`bernoulli_head_U/V` and `horizontal_advection_U/V` in
`src/Advection/vector_invariant_advection.jl` dispatch on
`grid::SphericalShellGrid` and route through the covariant kernels. The
generic `VectorInvariantKEGradientEnergyConserving`,
`VectorInvariantEnergyConserving`, and
`VectorInvariantEnstrophyConserving` schemes all map to the same
covariant kernels — they do not differ on a non-orthogonal grid because
the vorticity-reconstruction choice and the KE-gradient form become
moot once the centered covariant operators are in place.

---

## 8. Transport velocities for tracer advection

Tracer advection uses `model.transport_velocities` rather than
`model.velocities`. On a non-orthogonal grid, the two are distinct:
`velocities` holds covariant components \(u_i\), but tracer advection
needs the area-normal volume transport \(\mathcal V^i\). Therefore,
`compute_transport_velocities!` calls
`convert_to_volume_flux_velocities!` on `SphericalShellGrid`, which
applies the Hodge map of §3.2 in place.

The vertical component is then re-diagnosed from continuity via
`update_vertical_velocities!`.

Code path (in `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl`):
```
compute_transport_velocities!(model, free_surface)
  └── if grid isa SphericalShellGrid:
        convert_to_volume_flux_velocities!(ũ, ṽ, grid, u, v)
      else:
        update_transport_velocities!(transport_velocities, velocities)  # identity
  └── update_vertical_velocities!(model.transport_velocities, grid, model)
```

The `ImplicitFreeSurface` and `SplitExplicitFreeSurfaces` paths each
also call `convert_to_volume_flux_velocities!` after their own
intermediate transport computation.

**Known hazard (2026-05-23):** the in-place call
`convert_to_volume_flux_velocities!(ũ, ṽ, grid, ũ, ṽ)` reads and writes
the same field, while the Hodge stencil interpolates from cross-located
neighbors. On GPU this is a race; on CPU it produces stale reads in a
row-major walk. A scratch buffer is needed (or explicit verification
that the stencil tolerates aliasing).

---

## 9. Geometry: OctaHEALPix matrix layout

The OctaHEALPix grid has \(4N^2\) cells laid out in a \(2N\times 2N\)
matrix. The matrix is *not* a flat torus. It is partitioned into four
\(N\times N\) blocks, each holding one quadrant of the sphere with a
specific 90° rotation:

| Matrix block | Quadrant | Rotation |
|--------------|----------|----------|
| \((1,1)\)    | \(q = 3\) | \(\rho = 2\) (180°) |
| \((1,2)\)    | \(q = 2\) | \(\rho = 1\) (90° CW) |
| \((2,1)\)    | \(q = 4\) | \(\rho = 3\) (270° CW) |
| \((2,2)\)    | \(q = 1\) | \(\rho = 0\) |

Cells in octant-local coordinates \((r, c)\) with \(r,c\in\{1,\ldots,N\}\)
map to matrix coordinates by:
\[
(i_{\mathrm{block}},j_{\mathrm{block}}) = \mathrm{Rot}_{\rho}(r, c),
\quad (i, j) = (i_{\mathrm{block}} + (R{-}1)N,\; j_{\mathrm{block}} + (C{-}1)N).
\]

with the rotations
\[
\mathrm{Rot}_0(r,c) = (r,c),\quad
\mathrm{Rot}_1(r,c) = (N{+}1{-}c,r),
\]
\[
\mathrm{Rot}_2(r,c) = (N{+}1{-}r, N{+}1{-}c),\quad
\mathrm{Rot}_3(r,c) = (c, N{+}1{-}r).
\]

(Implemented as `rotate_octahealpix_indices(r, c, N, Val{ρ}())` in
`src/Grids/spherical_shell_grid.jl`.)

### 9.1 Seam classification

There are 8 seams between blocks:
- 4 **internal seams** within the matrix (at \(i = N|N{+}1\) or
  \(j = N|N{+}1\)). Crossing an internal seam clockwise advances
  \(\Delta q = +1\).
- 4 **external seams** at the matrix boundary (matrix \(\bmod\) wrap at
  \(i = 1|2N\) or \(j = 1|2N\)). Crossing an external seam advances
  \(\Delta q = -1\).

### 9.2 Halo-fill semantics across a seam

A halo cell at \((0, j)\) is the geographic \(-i\) neighbor of \((1, j)\).
Computing its source cell requires:

1. Identify the block \((R, C)\) of \((1, j)\) and its quadrant \(q\)
   and rotation \(\rho\).
2. Classify the seam: \(-i\) from \((1, j)\) is an *external* seam
   crossing if \(R = 1\), or an *internal* seam crossing if \(R = 2\)
   (entering block \((1,C)\) from above).
3. The destination quadrant is \(q' = q + \Delta q\) with
   \(\Delta q = -1\) for external or \(\Delta q = +1\) for internal
   (clockwise convention).
4. Compute the octant-local \((r, c)\) of the source cell, apply
   \(\mathrm{Rot}_{\rho-\rho'}\) to get the octant-local position in the
   destination, and place it in the matrix.

The full per-seam table is not yet verified in the current implementation
— the existing connectivity uses \(\bmod 2N\) wrap, which collides with
the rotation structure. Building the correct table is the highest-priority
next implementation task.

### 9.3 Vector-component transforms across a seam

The covariant basis of the source octant is related to that of the
destination by a rotation of \(\Delta\rho \cdot 90°\). A covariant
vector transforms as

| \(\Delta\rho\) | \(u_1' = \cdot\) | \(u_2' = \cdot\) |
|----------------|------------------|------------------|
| \(0\)          | \(+u_1\)         | \(+u_2\)         |
| \(+1\) (90° CW) | \(+u_2\)        | \(-u_1\)         |
| \(+2\) (180°)  | \(-u_1\)         | \(-u_2\)         |
| \(+3\) (270° CW) | \(-u_2\)       | \(+u_1\)         |

Contravariant vectors and volume transports \(\mathcal V^i\) transform
with the *inverse* rotation (swap the sign of the off-diagonal entries).
Scalars (tracers, pressure, ccc-located KE) are invariant under all
\(\Delta\rho\). The covariant vorticity \(\zeta_{12}\) is a
pseudoscalar in 2D and transforms with \(\det\mathcal R\); since all 90°
rotations have \(\det = +1\), \(\zeta_{12}\) is invariant.

These transforms must be applied by `QuadFoldedZipperBoundaryCondition`
during halo fill — for each crossed seam, multiply the copied
covariant-velocity components by the matrix in the table above.

---

## 10. Boundary conditions

### 10.1 Default dispatch

For `(QuadFolded, QuadFolded, Bounded)` topology:

| Location | Direction | BC class                              |
|----------|-----------|---------------------------------------|
| Center   | x, y      | `NoFluxBoundaryCondition` (`ZFBC`)    |
| Center   | z         | `NoFluxBoundaryCondition` (`ZFBC`)    |
| Face     | x, y      | `ImpenetrableBoundaryCondition` (`OBC`) |
| Face     | z         | `ImpenetrableBoundaryCondition` (`OBC`) |
| Nothing  | any       | `nothing`                              |

Auxiliary BCs on `Face` directions resolve to `nothing` (halo filled by
the zipper); on `Center` directions they resolve to `ZFBC`.

### 10.2 `QuadFoldedZipperBoundaryCondition` (planned)

The default `ZFBC` mirrors the boundary cell value into the halo, which
is wrong for a folded topology. The correct behavior is:

1. Look up the connectivity-implied source cell (§9.2).
2. Apply the vector-component transform if the field is a covariant
   vector (§9.3).
3. Write the transformed value into the halo cell.

The current state: connectivity exists but uses naïve `mod1` wrap; the
halo-fill kernel happens to follow the same `mod1` path; both are
geographically wrong at seam crossings.

---

## 11. Test coverage and tolerances

### 11.1 Metric identities (gnomonic panel)

`test_equiangular_gnomonic_panel_metrics` enforces:
- \(g_{ij}g^{jk} = \delta_i^k\) to \(\le 100\varepsilon\)
- \(J^2 = \det g_{ij}\) to \(\le 100\varepsilon\)
- \(G^{ij} = J g^{ij}\) to \(\le 100\varepsilon\)

### 11.2 Single-panel free-stream preservation

`test_equiangular_gnomonic_nonzero_transport_conservation` reconstructs
covariant components from a known volume transport, then checks the
discrete divergence is roundoff-zero. Tolerance: \(10^3\varepsilon\cdot
\|\mathcal V\|\).

### 11.3 Vector-invariant primitives

`test_equiangular_gnomonic_vector_invariant_primitives` enforces, all
at \(10^3\varepsilon\cdot\text{scale}\):
- Contravariant velocity components match \(g^{ij}u_j\).
- Kinetic energy matches the analytic form.
- Bernoulli head matches \(\partial_i K\).
- Circulation matches \(\Delta\xi^1\Delta\xi^2(\partial_1 u_2 - \partial_2 u_1)\).
- Vorticity matches circulation / \(A_z\).
- Rotational advection matches \(-\zeta_{12}\overline{\mathcal V^2}\) at fcc.

### 11.4 OctaHEALPix seam continuity (currently failing)

`test_octahealpix_neighbor_geographic_continuity` (added 2026-05-22)
asserts that all four connectivity-pointed matrix neighbors of every
cell are within a great-circle arc of \(\pi/2\) on the unit sphere. The
*interior* portion (1 \(<\) i, j \(<\) 2N) passes. The *seam* portion
fails (max arc \(\approx 2.7\) rad for N = 4) and is marked
`@test_broken`. This will become a `@test` once §9.2 is implemented.

### 11.5 OctaHEALPix seam halo consistency (currently passing trivially)

`test_octahealpix_seam_halo_consistency` (added 2026-05-22) loads a
tracer with values equal to ring indices, fills halos, and asserts
each halo cell holds the connectivity-pointed neighbor's value. Passes
today *because* the halo fill happens to follow the same `mod1` wrap
that the connectivity uses; once §9.2 corrects the connectivity, the
halo-fill kernel must also be updated to keep this test green.

### 11.6 OctaHEALPix metric tensor vs Cartesian derivatives (REQUIRED — not yet written)

Every quantitative test on OctaHEALPix uses the stored `g_ij`, `J`,
`G^ij` arrays. Those arrays are currently filled by
`horizontal_spherical_shell_metric_tensor(φ, radius)`, which returns
the **(λ, φ)** metric components. On OctaHEALPix the computational
coordinates are matrix indices `(i, j)`, **not** longitude/latitude.
Until the stored metric is verified against finite-difference
derivatives of the actual `(i, j) → (x, y, z)` map, every other test
on the global grid is self-consistency, not physical correctness.

The required test (acceptance criterion: **red line**):

```julia
function test_octahealpix_metric_vs_cartesian_derivatives(FT, N; ε = 1e-3)
    mapping = OctaHEALPixMapping(N)
    grid = SphericalShellGrid(CPU(), FT; mapping, …)
    metrics = grid.metrics

    # Numerical (∂x/∂i, ∂x/∂j) from Cartesian positions at i±ε, j±ε.
    # For each interior (i, j):
    #   g₁₁_numeric = (∂x/∂i · ∂x/∂i)
    #   g₂₂_numeric = (∂x/∂j · ∂x/∂j)
    #   g₁₂_numeric = (∂x/∂i · ∂x/∂j)
    #   J_numeric   = √(g₁₁ g₂₂ − g₁₂²)
    # Compare against grid.metrics.g₁₁ᶜᶜᵃ[i, j] etc.

    @test maximum_error_g₁₁ ≤ 100 ε² · (grid.radius / N)²
    @test maximum_error_g₂₂ ≤ 100 ε² · (grid.radius / N)²
    @test maximum_error_g₁₂ ≤ 100 ε² · (grid.radius / N)²
    @test maximum_error_J  ≤ 100 ε² · (grid.radius / N)²
end
```

**Until this test passes, every OctaHEALPix tendency test is
characterizing the wrong metric.**

### 11.7 Cosine-bell solid-body advection (REQUIRED — gate for production)

Single-step tendency assertions like

```julia
expected_Gc = -div_Uc(i, j, k, grid, advection, transport_velocities, c)
@test computed_Gc ≈ expected_Gc rtol=1e-12
```

are **self-consistency checks**: both sides use the same metric and
the same Hodge map. They cannot catch:
- a metric formula that is wrong but consistent (§11.6),
- accumulated mass-conservation drift over many steps,
- L2 error growth from numerical dispersion or seam reflection,
- vorticity damping from Hodge-map non-adjointness.

**Long-time integration is the only test that exposes accumulated
error.** The canonical case is the Williamson et al. (1992) cosine-bell
solid-body rotation:

\[
c_0(\lambda, \varphi) = \begin{cases}
\tfrac{h_0}{2}\left(1 + \cos\frac{\pi r}{R}\right) & r < R, \\
0 & \text{otherwise,}
\end{cases}
\quad r = a \arccos(\sin\varphi_c \sin\varphi + \cos\varphi_c \cos\varphi \cos(\lambda - \lambda_c))
\]

with solid-body rotation velocity

\[
\mathbf{u}(\lambda, \varphi) = u_0 (\cos\alpha\cos\varphi + \sin\alpha\cos\lambda\sin\varphi)\,\hat\lambda
                              - u_0 \sin\alpha\sin\lambda \,\hat\varphi.
\]

Integrate for **one full revolution** (and ideally 12, 24, 100), then
measure:

| Metric | Symbol | Acceptance |
|---|---|---|
| Mass drift | \(\varepsilon_{\text{mass}} = \|\bar c - \bar c_0\| / \|\bar c_0\|\) | \(\le 10^{-11}\) over 1 rev; \(\le 10^{-9}\) over 100 revs |
| L2 error | \(\varepsilon_{L_2} = \|c - c_0\|_2 / \|c_0\|_2\) | \(\le 5\times 10^{-2}\) at N=128, α=π/4, 1 rev |
| Min/max bounds | \(\min c, \max c\) | within \([-0.01, h_0 + 0.01]\) — no spurious extrema |
| Seam ratio | error inside vs outside seam-touching cells | within 2× of each other |

The pole angle \(\alpha\) must be tested at three values to ensure the
seam-touching path is exercised:
- \(\alpha = 0\) (rotation about z-axis; tracer crosses the equator
  via the matrix interior),
- \(\alpha = \pi/4\) (tracer passes through both pole regions and
  several seams),
- \(\alpha = \pi/2\) (rotation about x-axis; tracer crosses through
  the polar singular regions and over the most seams).

**This test is the gate for any production claim.** Single-step
tendency tests can verify the discrete formula reproduces the
discrete divergence operator; only long-time integration can verify
the discrete formula reproduces the *physical* tracer evolution.
Without it, all the type-system bureaucracy around QCovZBC/QConZBC
and the partial/paired/normal dispatch machinery is unverified
scaffolding.

The design doc (`spherical_shell_grid_design.md` §15) names cosine-
bell as a future success criterion. Per §11.6 above, the metric
tensor test must pass first, otherwise the cosine-bell test only
characterizes how an incorrect metric distorts an analytic solution.

---

## 12. Implementation roadmap

This section refers to the roadmap in `spherical_shell_grid_design.md`
§15 with current status:

| Step | Subject                                | Status     |
|------|----------------------------------------|------------|
| 1    | Grid skeleton                          | done       |
| 2    | Gnomonic panel mapping                 | done       |
| 3    | OctaHEALPix mapping & reference tests  | done (matrix) |
| 4    | QuadFolded connectivity & halos        | **structural; table not verified** |
| 5    | Hodge candidates                       | A only; B/C not implemented |
| 6    | Transport diagnostics                  | done (target-metric Hodge) |
| 7    | Scalar + centered VI tests             | done on gnomonic panel; not on OctaHEALPix |
| 8    | WENO vector-invariant                  | not started |
| 9    | Breeze integration                     | not started |
| 10   | HFSM staged integration                | partial (centered VI accepted) |

The single highest-leverage next step is to complete §9.2 — the
octant-aware connectivity table and the `QuadFoldedZipperBoundaryCondition`
that drives off it. This unlocks step 7 on OctaHEALPix and is the gate
for everything else on the global grid.
