# Plan for non-orthogonal C-grid capabilities with `NonOrthogonalWENOVectorInvariant`

## 1. Purpose

This document describes the development plan for adding non-orthogonal C-grid capabilities to Oceananigans.jl and Breeze.jl, centered on a metric-compatible `SphericalShellGrid` and a new vector-invariant momentum advection family:

```julia
NonOrthogonalVectorInvariant
NonOrthogonalWENOVectorInvariant
NonOrthogonalScalarAdvection
```

The goal is to support structured non-orthogonal finite-volume C-grids, beginning with one panel of an equiangular / gnomonic cubed-sphere grid, and to build a 3D vector-invariant momentum scheme that can be used in both:

- Oceananigans `HydrostaticFreeSurfaceModel`, especially the hydrostatic-incompressible / Boussinesq case; and
- Breeze `AtmosphereModel`, especially fully compressible dynamics.

The plan emphasizes minimal but forward-looking code changes: reuse existing Oceananigans field-location, halo, metric, operator, and advection patterns where possible, but do not preserve old abstractions when they obscure the distinction between model velocity, transport velocity, and integrated transport.

## 1.1 Branch status and quantitative exit gates (2026-05-23)

| Objective | Branch status | Quantitative gate still required |
|---|---|---|
| `SphericalShellGrid` + `OctaHEALPixMapping` | ✅ Implemented | Full map/table coverage with seam orientation and edge-corner blocks remains blocked by boundary transport tests. |
| Non-orthogonal Hodge/operator stack (`Covariant ↔ contravariant/flux`) | ✅ Implemented | Adjointness/energy-positive campaign and long-range convergence metrics still pending release-level lock-down. |
| Tracer advection on non-orthogonal meshes | ✅ One-step analytic consistency tests present | Multi-step cosine-bell/rotated seam runs at `N=64,128` with mass drift, seam ratio, and shape metrics. |
| Linear momentum path (`advection = nothing`) | ✅ Hooked through `div_Uc` and `HydrostaticFreeSurfaceModel` dispatch | Full linear SWE tendency identities and gravity-wave benchmarks not yet executed in long integration. |
| Centered non-orthogonal vector-invariant (`VectorInvariant`) | ✅ Accepted and algebraic primitives tested | Full nonlinear centered shallow-water momentum/energy/seam gates, including orthogonal-limit and long-run benchmarks, still pending. |
| `WENOVectorInvariant` on `SphericalShellGrid` | ⛔ Not supported | `NonOrthogonalWENOVectorInvariant` constructor + update diagnostics + small-Rossby WENO tests still pending. |

Current block condition (must remain true until gate pass):

- `relative_vorticity` must be the reconstructed quantity in WENO flux reconstruction.
- `model.transport_velocities` on `SphericalShellGrid` must be non-orthogonally diagnostic, not a blind copy.
- `QuadFolded` transport orientation must be validated for vector and flux fields at `H = 1,2,3,6`.

Recommended acceptance rule: each gate must be reported with a scalar number, an explicit norm, and a CI/release threshold in `oceananigans_nonorthogonal_spherical_cgrid_quantitative_plan.md`.

## 1.2 Quantitative checkpoints before WENO unlock

Before enabling `NonOrthogonalWENOVectorInvariant` in HFSM, each checkpoint below must pass, with the same norms and gates used in the main quantitative plan:

This unlock is also coupled to source-level implementation milestones:

- `transport_velocities` and transport diagnostics in HFSM must be non-orthogonally separated
  (`hydrostatic_free_surface_model.jl`: `validate_momentum_advection`, `compute_transport_velocities!`).
- `QuadFolded` transport/flux orientation for vector and tensor fields must be complete in boundary machinery
  (`src/BoundaryConditions/field_boundary_conditions.jl`, `src/BoundaryConditions/fill_halo_regions_*.jl`).
- `NonOrthogonalScalarAdvection` must share the same transport state/flux object as the momentum path
  (not implemented yet; placeholder source point currently in `src/Advection/` under a new scalar advection scheme).
- WENO implementation must reconstruct **relative** vorticity (`ζ`) and then add planetary contribution (`f`) afterward in any mixed-path formulation; tests must reject absolute-vorticity-only reconstruction.

| Checkpoint | Required metric | CI gate | Release gate |
|---|---|---:|---:|
| Seam transport topology | Seam orientation identity completeness | zero unmapped transport faces at `H=1` | zero unmapped transport faces at `H=3` |
| Centered baseline | Centered nonlinear vector-invariant end-to-end residuals on `N=64` | no regressions > `100eps` in mass, energy residual, seam ratio `≤ 2` | same |
| Relative-vorticity reconstruction test | WENO vorticity input identification | exact | exact |
| Small-Rossby contamination | `error(noise metric) / error(baseline)`, `|f|/|ζ| = 10^4` | `≤ 2` | `≤ 1.25` |
| Adjacent seam consistency | Seam ratio on reconstructed fields (`u,v,ζ`) | `≤ 2` | `≤ 2` |
| WENO smooth convergence | Observed order for cosine-advection-like transport | `p ≥ 4.0` | `p ≥ 4.5` away from folds |

If any checkpoint misses release gate, keep WENO blocked and continue centered/transport/horizontal-metric hardening.

---

## 2. Guiding principles

### 2.1 Use one grid concept: `SphericalShellGrid`

Use the term and type name:

```julia
SphericalShellGrid
```

rather than separating the public API into `OrthogonalSphericalShellGrid` and `NonOrthogonalSphericalShellGrid`.

Orthogonality is a property of the mapping and Hodge/metric operators, not a different physical model concept. A spherical-shell grid may be constructed from different mappings:

```julia
SphericalShellGrid(; mapping = LatitudeLongitudeShell(), ...)
SphericalShellGrid(; mapping = ConformalCubedSpherePanel(), ...)
SphericalShellGrid(; mapping = EquiangularGnomonicCubedSpherePanel(), ...)
```

The mapping determines the metric tensor and whether off-diagonal terms such as `g¹²` vanish.

### 2.2 Separate model velocity from transport velocity

On an orthogonal grid, the same face field is often interpreted both as a model velocity and as a scalar-transport velocity. On a non-orthogonal grid, that conflation breaks down.

Use this convention:

```julia
model.velocities
```

means the model's velocity variables used by momentum tendencies, diagnostics, closures, forcing, and output.

```julia
model.transport_velocities      # HFSM-style field tuple
transport_velocities(model)     # Breeze-style accessor
```

means the velocity-like tuple used to advect tracers and thermodynamic variables.

```julia
transport_fluxes(model)
```

means the conservative integrated transport used by continuity, conservative scalar transport, and non-orthogonal vector-invariant momentum.

For simple cases:

```julia
transport_velocities(model) = model.velocities
```

or

```julia
model.transport_velocities === model.velocities
```

is acceptable.

For non-orthogonal grids, `transport_velocities` should generally differ from `model.velocities`.

### 2.3 Use verbose code names for persistent objects

Use clear semantic names for structs, fields, and public functions:

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

Reserve compact mathematical symbols for small internal operator functions and derivation-facing helpers:

```julia
ζ₁₂ᶠᶠᶜ(...)
Z₁₂ᶠᶠᶜ(...)
Dᵢ𝒰ⁱᶜᶜᶜ(...)
```

This keeps persistent state readable while allowing numerical kernels to resemble the mathematical derivation.

### 2.4 Treat metric conversion as a Hodge-map problem

Do not hard-code a single staggered interpolation convention until it has been tested. The conversion from covariant velocities to volume transports is a discrete Hodge map:

\[
\mathcal V^i = \mathsf H^i{}_j u_j .
\]

On non-orthogonal grids, the off-diagonal blocks of \(\mathsf H\) are the delicate part. Candidate Hodge maps must be compared empirically and algebraically.

---

## 3. Mathematical notation

### 3.1 Velocity and transport variables

Use:

\[
u_i = \boldsymbol u \cdot \boldsymbol a_i
\]

for covariant velocity components and

\[
u^i = \boldsymbol u \cdot \boldsymbol a^i = g^{ij}u_j
\]

for contravariant velocity components.

Define volume transport:

\[
\mathcal V^i = J u^i,
\]

and mass transport:

\[
\mathcal U^i = J\rho u^i = \rho \mathcal V^i.
\]

Define transport divergences:

\[
\mathcal D_{\mathcal V} = D_i\mathcal V^i,
\qquad
\mathcal D_{\mathcal U} = D_i\mathcal U^i.
\]

For hydrostatic-incompressible fixed-grid flow,

\[
\mathcal D_{\mathcal V}=0.
\]

For constant-density Boussinesq flow with \(\mathcal U^i=\rho_0\mathcal V^i\),

\[
\mathcal D_{\mathcal U}=0.
\]

For compressible dynamics,

\[
\partial_t(J\rho)+\mathcal D_{\mathcal U}=0,
\]

so \(\mathcal D_{\mathcal U}\neq0\) in general.

### 3.2 Vorticity notation

Use relative vorticity two-forms:

\[
\zeta_{ij}=D_i u_j-D_j u_i,
\]

planetary vorticity two-forms:

\[
f_{ij},
\]

and absolute vorticity two-forms:

\[
Z_{ij}=\zeta_{ij}+f_{ij}.
\]

When a normalized potential vorticity is needed, define

\[
q_{ij}=\frac{Z_{ij}}{\mu_{ij}},
\]

where \(\mu_{ij}\) is the compatible mass, layer-thickness, or density measure at the vorticity location. In the vector-invariant momentum equation, the object that multiplies transport is the absolute vorticity two-form \(Z_{ij}\), not necessarily a normalized PV.

### 3.3 WENO reconstruction of vorticity

For `NonOrthogonalWENOVectorInvariant`, WENO should reconstruct **relative vorticity**, not absolute vorticity:

\[
\widehat Z_{ij}
=
\mathcal R_{\rm WENO}[\zeta_{ij}]
+
\mathcal R_{\rm smooth}[f_{ij}].
\]

This avoids small-Rossby-number failure modes where the smooth, large planetary vorticity dominates WENO smoothness indicators and hides roughness in the dynamically important relative vorticity.

---

## 4. Oceananigans-style C-grid locations

Use Oceananigans-style location notation in documentation and code comments:

| Location | Meaning | Typical fields |
|---|---|---|
| `ccc` | cell center | tracers, density, pressure, kinetic energy |
| `fcc` | first-direction face | \(u_1\), \(u^1\), \(\mathcal V^1\), \(\mathcal U^1\) |
| `cfc` | second-direction face | \(u_2\), \(u^2\), \(\mathcal V^2\), \(\mathcal U^2\) |
| `ccf` | third-direction face | \(u_3\), \(u^3\), \(\mathcal V^3\), \(\mathcal U^3\) |
| `ffc` | horizontal edge | \(\zeta_{12}\), \(Z_{12}\) |
| `fcf` | first-vertical edge | \(\zeta_{13}\), \(Z_{13}\) |
| `cff` | second-vertical edge | \(\zeta_{23}\), \(Z_{23}\) |

Covariant velocities live at face locations:

\[
u_1^{\mathrm{fcc}},
\qquad
u_2^{\mathrm{cfc}},
\qquad
u_3^{\mathrm{ccf}}.
\]

The non-orthogonal Hodge map should produce volume transports at the corresponding transport locations:

\[
\mathcal V^{1,\mathrm{fcc}},
\qquad
\mathcal V^{2,\mathrm{cfc}},
\qquad
\mathcal V^{3,\mathrm{ccf}}.
\]

---

## 5. `SphericalShellGrid` development

### 5.1 Initial target: one equiangular gnomonic cubed-sphere panel

The first grid target is a single structured panel of the equiangular / gnomonic cubed sphere. On a reference panel, define

\[
\alpha,\beta\in[-\pi/4,\pi/4],
\qquad
p=\tan\alpha,
\qquad
q=\tan\beta,
\qquad
D=1+p^2+q^2.
\]

The unit-sphere map is

\[
\widehat{\boldsymbol x}(\alpha,\beta)
=\frac{1}{\sqrt D}
\begin{pmatrix}
p\\ q\\ 1
\end{pmatrix}.
\]

For shell radius \(r\),

\[
\boldsymbol x(\alpha,\beta,r)=r\widehat{\boldsymbol x}(\alpha,\beta).
\]

The horizontal metric has nonzero off-diagonal term:

\[
g_{\alpha\beta}
= -r^2\frac{pq(1+p^2)(1+q^2)}{D^2},
\]

which makes this grid an ideal first non-orthogonal C-grid test case.

### 5.2 Thin-shell first, deep-shell later

The first implementation should use a thin-atmosphere / shallow-shell approximation:

- horizontal metrics are evaluated at a representative radius;
- the vertical/radial coordinate is orthogonal to horizontal surfaces;
- \(g_{13}=g_{23}=0\);
- only horizontal off-diagonal metrics \(g_{12},g^{12}\) are active.

Relaxing the shallow-shell approximation later means:

- horizontal metrics vary with radius;
- volume and area factors depend on radius;
- metric terms in momentum may include full spherical-shell geometric effects;
- Coriolis and gravity may need a full deep-atmosphere representation rather than shallow-atmosphere approximations.

This is more than just a grid-storage issue because momentum equations, Coriolis, and hydrostatic balance may need deep-atmosphere forms. However, the first non-orthogonal C-grid challenge is already present in the thin-shell limit, so start there.

### 5.3 Metric placement

Metrics and Hodge coefficients should be stored or evaluated at the locations where they are used. For example, any coefficient used to produce \(\mathcal V^{1,\mathrm{fcc}}\) should live at `fcc` or be generated at `fcc`.

This suggests storing or evaluating at least:

```julia
jacobian_center             # Jᶜᶜᶜ
face_area_1                 # A₁ᶠᶜᶜ
face_area_2                 # A₂ᶜᶠᶜ
face_area_3                 # A₃ᶜᶜᶠ
contravariant_metric_11_fcc
contravariant_metric_12_fcc
contravariant_metric_21_cfc
contravariant_metric_22_cfc
```

The exact storage layout should be chosen after the Hodge-map test campaign. Some metrics can be evaluated on the fly if analytic expressions are cheap; others may be stored to avoid repeated work and to guarantee halo consistency.

---

## 6. Staggered Hodge-map candidates

The key numerical issue is how to compute cross-basis contributions such as the \(u_2\) contribution to \(\mathcal V^1\) at `fcc`.

The continuous expression is unambiguous:

\[
\mathcal V^1 = Jg^{11}u_1 + Jg^{12}u_2 + Jg^{13}u_3.
\]

But on the C-grid, \(u_1\), \(u_2\), and \(u_3\) live at different staggered locations, so we must define a discrete Hodge map:

\[
\mathcal V^i = \mathsf H^i{}_j u_j.
\]

We should not choose a stencil by taste. Implement at least the following candidates and select empirically.

### 6.1 `TargetMetricHodge`

Interpolate the source velocity to the target transport location, then multiply by the metric coefficient at the target location:

\[
\left(\mathsf H^{12}_{\rm target}u_2\right)^{\mathrm{fcc}}
=
G^{12,\mathrm{fcc}}
\mathcal I_{\mathrm{cfc}\to\mathrm{fcc}}u_2,
\qquad
G^{ij}=Jg^{ij}.
\]

This is simple and aligns with the principle that metrics live where the operator is evaluated.

### 6.2 `ProductInterpolatedHodge`

Form the metric-weighted product at the source location, then interpolate the product:

\[
\left(\mathsf H^{12}_{\rm product}u_2\right)^{\mathrm{fcc}}
=
\mathcal I_{\mathrm{cfc}\to\mathrm{fcc}}
\left(G^{12,\mathrm{cfc}}u_2^{\mathrm{cfc}}\right).
\]

This may better preserve some metric identities and may be closer to flux-first thinking.

### 6.3 `EnergySymmetricHodge`

Define the discrete kinetic energy first:

\[
E_h
=
\frac12\sum_q w_q G^{ij}_q
\left(\mathcal I_i^q u_i\right)
\left(\mathcal I_j^q u_j\right),
\]

then derive the Hodge map by differentiating:

\[
\mathcal V_i
=
W_i^{-1}\frac{\partial E_h}{\partial u_i}.
\]

This guarantees a symmetric positive kinetic-energy quadratic form if the metric tensor and quadrature weights are positive. It is the safest theoretical candidate, though it may be more expensive or less local.

---

## 7. TRiSK lessons for our structured grid

TRiSK does not answer the question “multiply first or interpolate first” directly. Instead, it turns staggered metric interpolation into a constrained operator-design problem.

TRiSK uses a primal/dual C-grid where normal velocity or flux lives on primal edges. The vector-invariant/PV term requires a perpendicular or tangential flux that is not prognosed. TRiSK constructs a mapping from normal flux to perpendicular flux:

\[
F_e^\perp
=
\frac{1}{d_e}
\sum_{e'} w_{e,e'}\,l_{e'}F_{e'}.
\]

The geometry factors are part of the map. The map is constrained so that the divergence of the mapped flux on the dual mesh is an interpolation of the divergence of the original flux on the primal mesh:

\[
D_{\rm dual}M(F)=I(D_{\rm primal}F).
\]

TRiSK also imposes energy neutrality of the PV/Coriolis flux through antisymmetric edge couplings and symmetric PV factors. The lesson for our structured grid is:

\[
\boxed{\text{Select Hodge and rotation maps by mimetic properties, not by local stencil aesthetics.}}
\]

Our analogue should test whether candidate Hodge maps satisfy:

\[
D_{\rm dual}\mathsf R(\mathcal V)
\approx
I(D_{\rm primal}\mathcal V),
\]

whether the vorticity/PV flux operator is energy neutral in the centered case,

\[
\langle u,\mathsf C_Zu\rangle_h=0,
\]

and whether a constant PV-like state remains constant under inviscid unforced dynamics.

---

## 8. `NonOrthogonalVectorInvariant` design

### 8.1 Generic reconstruction-agnostic operator

Implement `NonOrthogonalVectorInvariant` as the generic operator. It should be configurable by reconstruction choices:

```julia
NonOrthogonalVectorInvariant(;
    vorticity_reconstruction = Centered(order = 2),
    shear_vorticity_reconstruction = Centered(order = 2),
    planetary_vorticity_reconstruction = Centered(order = 2),
    divergence_reconstruction = Centered(order = 2),
    kinetic_energy_gradient_reconstruction = Centered(order = 2),
    transport_reconstruction = Centered(order = 2),
    hodge_map = EnergySymmetricHodge(),
    upwinding = nothing,
)
```

The generic object separates the geometric vector-invariant form from the reconstruction method. This allows centered, WENO, mixed, and future reconstructions to share the same non-orthogonal metric/transport machinery.

### 8.2 Materialized top-level fields

After materialization with a grid, the object should own its metric-compatible diagnostic fields explicitly at top level:

```julia
struct NonOrthogonalVectorInvariant{...} <: AbstractAdvectionScheme{N, FT}
    # Reconstruction choices
    vorticity_reconstruction
    shear_vorticity_reconstruction
    planetary_vorticity_reconstruction
    divergence_reconstruction
    kinetic_energy_gradient_reconstruction
    transport_reconstruction
    hodge_map
    upwinding

    # Diagnostic state
    covariant_velocities
    contravariant_velocities
    transport_velocities
    volume_transport
    mass_transport
    volume_transport_divergence
    mass_transport_divergence
    kinetic_energy
    relative_vorticity
    planetary_vorticity
    absolute_vorticity
end
```

This is a deliberate departure from current `WENOVectorInvariant`, which stores reconstruction configuration but no runtime diagnostic fields. The non-orthogonal operator needs these fields to keep scalar transport, continuity, kinetic energy, and vector-invariant momentum consistent.

### 8.3 Update lifecycle

Each model tendency evaluation should update the non-orthogonal diagnostics in a fixed order:

```julia
update_vector_invariant_diagnostics!(model.advection.momentum, model)
```

Internally:

```julia
compute_covariant_velocities!(advection.covariant_velocities, model.momentum, density, grid)
compute_contravariant_velocities!(advection.contravariant_velocities,
                                  advection.covariant_velocities,
                                  grid,
                                  advection.hodge_map)
compute_transport_velocities!(advection.transport_velocities,
                              advection.contravariant_velocities,
                              grid)
compute_volume_transport!(advection.volume_transport,
                          advection.contravariant_velocities,
                          grid)
compute_mass_transport!(advection.mass_transport,
                        density,
                        advection.volume_transport,
                        grid)
compute_transport_divergence!(advection.volume_transport_divergence,
                              advection.volume_transport,
                              grid)
compute_transport_divergence!(advection.mass_transport_divergence,
                              advection.mass_transport,
                              grid)
compute_kinetic_energy!(advection.kinetic_energy,
                        advection.covariant_velocities,
                        advection.contravariant_velocities,
                        grid)
compute_relative_vorticity!(advection.relative_vorticity,
                            advection.covariant_velocities,
                            grid)
compute_planetary_vorticity!(advection.planetary_vorticity,
                             grid,
                             coriolis)
compute_absolute_vorticity!(advection.absolute_vorticity,
                            advection.relative_vorticity,
                            advection.planetary_vorticity)
```

---

## 9. `NonOrthogonalWENOVectorInvariant` design

### 9.1 Convenience constructor

`NonOrthogonalWENOVectorInvariant` should be a convenience constructor for `NonOrthogonalVectorInvariant`:

```julia
NonOrthogonalWENOVectorInvariant(FT = Oceananigans.defaults.FloatType;
    vorticity_order = 9,
    shear_vorticity_order = 5,
    divergence_order = 5,
    kinetic_energy_gradient_order = 5,
    transport_order = 5,
    planetary_vorticity_reconstruction = Centered(order = 4),
    hodge_map = EnergySymmetricHodge(),
    upwinding = nothing,
    minimum_buffer_upwind_order = 1,
    weno_kw...,
)
```

It constructs WENO sub-reconstructions and returns a `NonOrthogonalVectorInvariant`.

### 9.2 Relative-vorticity WENO

The WENO vorticity reconstructions should act on relative vorticity:

```julia
vorticity_reconstruction          # reconstructs ζ12
shear_vorticity_reconstruction    # reconstructs ζ13, ζ23
```

The planetary vorticity should be centered, exact, or smoothly reconstructed:

```julia
planetary_vorticity_reconstruction = Centered(order = 4)
```

Absolute vorticity is assembled from reconstructed pieces:

```julia
absolute_vorticity_reconstruction =
    reconstruct(vorticity_reconstruction, relative_vorticity) +
    reconstruct(planetary_vorticity_reconstruction, planetary_vorticity)
```

### 9.3 Hydrostatic split versus full 3D form

The full compressible vector-invariant conservative tendency is

\[
\partial_t(J\rho u_i)
=
\mathcal U^jZ_{ij}
-
J\rho D_iK
-
u_i\mathcal D_{\mathcal U}
+
\cdots.
\]

Existing WENO vector-invariant hydrostatic logic rewrites the vertical contribution to horizontal momentum as a vertical momentum-flux divergence plus a horizontal divergence flux:

\[
D_3(\mathcal U^3u_i)+u_i\mathcal D_{h,\mathcal U}
=
\mathcal U^3D_3u_i+u_i\mathcal D_{\mathcal U}.
\]

These are alternative arrangements, not additive corrections. Implementation must avoid double-counting the full divergence term and the hydrostatic split.

For Breeze fully compressible dynamics, start with the full 3D form.

For HFSM hydrostatic-incompressible dynamics, start with the hydrostatic split form using volume transports.

---

## 10. Scalar advection design

Implement:

```julia
NonOrthogonalScalarAdvection
```

as a wrapper that references the materialized `NonOrthogonalVectorInvariant` object, so all scalars use the same transport fields.

Example:

```julia
struct NonOrthogonalScalarAdvection{R, M}
    scalar_reconstruction :: R
    momentum_advection :: M
end
```

Then scalar advection can use either:

1. `transport_velocities` for compatibility with existing velocity-based finite-volume operators; or
2. `volume_transport` / `mass_transport` directly for a cleaner conservative flux form.

Long-term, flux-based scalar advection is preferred:

\[
\partial_t(J\rho c) = -D_i(\mathcal U^i c^\uparrow).
\]

For hydrostatic-incompressible HFSM tracer advection, use volume transport:

\[
\partial_t(Jc) = -D_i(\mathcal V^i c^\uparrow).
\]

---

## 11. Oceananigans and Breeze integration plan

### 11.1 Start in Oceananigans infrastructure

Begin with reusable infrastructure:

1. `SphericalShellGrid` and equiangular gnomonic panel mapping.
2. Metric storage/evaluation at Oceananigans locations.
3. Hodge-map candidate implementations.
4. Transport diagnostics: `compute_volume_transport!`, `compute_mass_transport!`, transport divergences.
5. `NonOrthogonalVectorInvariant`, `NonOrthogonalWENOVectorInvariant`, and `NonOrthogonalScalarAdvection` abstractions.

This should be model-independent.

### 11.2 Use Breeze as first full dynamics target

Breeze is the better first full-model target because it avoids HFSM free-surface coupling. The compressible path needs non-orthogonal mass transport in continuity:

\[
\partial_t(J\rho)=-D_i\mathcal U^i.
\]

Breeze already has the conceptual hook:

```julia
transport_velocities(model)
advecting_momentum(model)
```

The non-orthogonal extension should add:

```julia
transport_fluxes(model)
```

and specialize density, scalar, and momentum tendencies to use the materialized non-orthogonal advection object.

### 11.3 Add HFSM in stages

HFSM already distinguishes `model.velocities` from `model.transport_velocities`, and tracers are advected with `model.transport_velocities`. This is the right abstraction, but free-surface dynamics touches additional code paths.

Stage HFSM integration as follows:

1. Rigid lid / no free surface on fixed grids.
2. Static vertical coordinate with explicit or implicit free surface.
3. Split-explicit free surface.
4. z-star moving vertical coordinate.

Do not start with split-explicit or z-star support. These require barotropic transport, vertical velocity reconstruction, and pressure correction to be metric-compatible.

---

## 12. Empirical verification campaign for staggered stencils

The implementation should include a formal Hodge-map test campaign comparing:

```julia
TargetMetricHodge
ProductInterpolatedHodge
EnergySymmetricHodge
```

The goal is to select the simplest Hodge map that passes mimetic, stability, and conservation tests.

### 12.1 Local metric and Hodge consistency

Tests:

- analytic metric inversion: \(g_{ij}g^{jk}=\delta_i^k\);
- positive metric determinant;
- one-panel spherical area sum;
- known-vector Hodge consistency for constant Cartesian velocity, solid-body rotation, and smooth manufactured vector fields.

Expected outcome: Hodge error converges under refinement.

### 12.2 Weighted adjointness

For cross Hodge blocks, test:

\[
\langle u_1,\mathsf H^{12}u_2\rangle_{\mathrm{fcc}}
=
\langle \mathsf H^{21}u_1,u_2\rangle_{\mathrm{cfc}}.
\]

Measure:

\[
\epsilon_{\rm adj}
=
\frac{\|W_1\mathsf H^{12}-(W_2\mathsf H^{21})^T\|}
{\|W_1\mathsf H^{12}\|+\|(W_2\mathsf H^{21})^T\|}.
\]

This should be roundoff-small for `EnergySymmetricHodge` and small/convergent for acceptable approximations.

### 12.3 Positive kinetic energy

Build the kinetic-energy matrix:

\[
E_h=\frac12u^TW\mathsf H u.
\]

The symmetric part must be positive definite:

\[
\lambda_{\min}\left(\frac12(W\mathsf H+\mathsf H^TW)\right)>0.
\]

Reject Hodge maps with negative kinetic-energy modes.

### 12.4 Free-stream preservation

For constant Cartesian velocity \(\boldsymbol u_0\), compute analytic covariant components and then volume transport:

\[
\mathcal V^i=\mathsf H^i{}_ju_j.
\]

Check:

\[
D_i\mathcal V^i\approx0.
\]

Also test constant scalar advection:

\[
D_i(\mathcal V^i c_0)\approx0.
\]

This is a red-line test. If it fails, the model will generate grid-induced tracer noise.

### 12.5 Orthogonal limit

Construct a family of grids with skewness parameter \(\epsilon\), so \(g^{12}=O(\epsilon)\). As \(\epsilon\to0\), the non-orthogonal algorithm should reduce to existing orthogonal Oceananigans behavior.

### 12.6 Manufactured convergence

Use smooth analytic fields on:

1. an affine skew periodic plane;
2. a smoothly varying periodic curvilinear plane;
3. one equiangular gnomonic spherical panel.

Measure convergence of:

```julia
volume_transport
mass_transport
transport_divergence
relative_vorticity
absolute_vorticity
kinetic_energy
```

### 12.7 Linear mode and computational-mode analysis

Build discrete linear operators for shallow-water-like or acoustic systems on affine and smoothly varying skew grids. Inspect eigenvalues for:

- positive real parts;
- checkerboard pressure modes;
- near-zero-energy velocity modes;
- large grid-scale eigenvalue outliers.

### 12.8 Conservation tests with centered advection

Before enabling WENO, centered non-orthogonal vector-invariant advection should pass:

- tracer mass conservation;
- tracer variance behavior under divergence-free transport;
- kinetic-energy conservation in inviscid incompressible fixed-grid tests;
- consistency of vertical split versus full 3D identity.

### 12.9 WENO-specific tests

After centered tests pass:

- verify WENO reconstructs relative vorticity, not absolute vorticity;
- test small-Rossby flows with \(|f|\gg|\zeta|\);
- confirm high-order smooth convergence;
- run sharp tracer/vorticity filament tests;
- inspect grid-scale spectrum in decaying turbulence.

---

## 13. Decision tree for Hodge-map selection

Use the following decision rule:

1. If `TargetMetricHodge` passes free-stream preservation, weighted adjointness, positivity, and orthogonal-limit tests, use it.
2. If `TargetMetricHodge` fails free-stream preservation but is otherwise acceptable, improve metric generation and conservative metric identities before abandoning the target-location philosophy.
3. If `ProductInterpolatedHodge` passes free-stream preservation but fails adjointness or positivity, use it only as a guide to construct a weighted-adjoint paired block.
4. If neither simple candidate passes, use `EnergySymmetricHodge`.
5. If `EnergySymmetricHodge` is too expensive, derive a mass-lumped approximation and rerun the full suite.

The final choice must be justified by tests, not by aesthetic preference.

---

## 14. Suggested implementation milestones

### Milestone 1: geometry and metrics

- Add `SphericalShellGrid` skeleton.
- Add `EquiangularGnomonicCubedSpherePanel` mapping.
- Add analytic metric evaluation at `ccc`, `fcc`, `cfc`, `ccf`, `ffc`, `fcf`, `cff` as needed.
- Add one-panel geometry tests.

### Milestone 2: Hodge maps and transport diagnostics

- Implement Hodge-map candidates.
- Implement `compute_contravariant_velocities!`.
- Implement `compute_transport_velocities!`.
- Implement `compute_volume_transport!` and `compute_mass_transport!`.
- Implement divergence of transports.
- Run the Hodge-map test campaign.

### Milestone 3: centered `NonOrthogonalVectorInvariant`

- Implement generic reconstruction-agnostic object.
- Implement centered vorticity, kinetic energy, divergence, and transport operators.
- Verify hydrostatic split identity and full 3D identity.
- Run conservation and orthogonal-limit tests.

### Milestone 4: `NonOrthogonalWENOVectorInvariant`

- Add WENO convenience constructor.
- Reconstruct relative vorticity with WENO.
- Add smooth/centered planetary vorticity reconstruction.
- Add WENO divergence and kinetic-energy-gradient reconstructions.
- Run WENO-specific tests.

### Milestone 5: scalar advection

- Add `NonOrthogonalScalarAdvection`.
- Use shared transport diagnostics from materialized momentum advection.
- Test passive tracer conservation and free-stream preservation.

### Milestone 6: Breeze integration

- Specialize `transport_velocities(model)` and `transport_fluxes(model)`.
- Replace compressible continuity with divergence of `mass_transport`.
- Use full 3D non-orthogonal vector-invariant momentum form.
- Run compressible test cases.

### Milestone 7: HFSM rigid-lid integration

- Use volume transport and hydrostatic split form.
- Avoid free-surface complications initially.
- Test hydrostatic-incompressible divergence constraint and turbulence cases.

### Milestone 8: HFSM free-surface integration

- Static vertical coordinate first.
- Then split-explicit free surface.
- Then z-star.
- Audit barotropic transport, pressure correction, and vertical velocity reconstruction.

### Milestone 9: six-panel connectivity

- Add panel-edge halo exchange.
- Rotate vector/covariant components consistently across panel boundaries.
- Verify global area, divergence, tracer conservation, and solid-body rotation.

---

## 15. Suggested test-file layout

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

---

## 16. Major risks and mitigations

### Risk: bad Hodge map creates numerical noise or computational modes

Mitigation: require free-stream preservation, Hodge positivity, weighted adjointness, linear mode tests, and centered energy tests before nonlinear simulations.

### Risk: scalar advection accidentally uses model velocities instead of transport velocities

Mitigation: enforce interface separation:

```julia
model.velocities
transport_velocities(model)
transport_fluxes(model)
```

Add tests that fail if scalar advection on a non-orthogonal grid uses covariant velocities directly.

### Risk: WENO reconstruction acts on smooth planetary vorticity instead of relative vorticity

Mitigation: implement WENO on `relative_vorticity` only; add small-Rossby reconstruction tests.

### Risk: HFSM free-surface paths require deeper changes than expected

Mitigation: delay free-surface integration until fixed-grid transport and Breeze dynamics are working. Stage HFSM support from rigid lid to static free surface to split-explicit to z-star.

### Risk: code churn from materialized advection fields

Mitigation: keep user-facing constructors familiar; allocate diagnostics only in `materialize_advection`; share one materialized momentum-advection object with all non-orthogonal scalar advection wrappers.

---

## 17. Success criteria

The program is successful when:

1. `SphericalShellGrid` supports a one-panel equiangular gnomonic cubed-sphere grid with verified metrics.
2. Candidate Hodge maps are compared and one is selected by documented tests.
3. The chosen Hodge map preserves free streams, has a positive kinetic-energy form, and passes orthogonal-limit tests.
4. Centered `NonOrthogonalVectorInvariant` passes conservation and consistency tests.
5. `NonOrthogonalWENOVectorInvariant` reconstructs relative vorticity correctly and behaves well in small-Rossby and turbulence tests.
6. Breeze compressible dynamics uses non-orthogonal mass transport consistently in continuity, scalar advection, and momentum.
7. HFSM rigid-lid support works with volume transport and hydrostatic split dynamics.
8. The system eventually supports free-surface and six-panel configurations without grid-induced tracer noise or spurious velocity modes.
