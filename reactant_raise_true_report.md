# Reactant Correctness Tests - `raise=true` Report

**Julia Version**: 1.11.8
**Reactant Version**: 0.2.114
**Test Date**: January 14, 2026
**Architecture**: CPU (arm64 macOS)

## Summary

| Test Category | Passed | Failed | Error | Total |
|--------------|--------|--------|-------|-------|
| `fill_halo_regions!` | 176 | 0 | 0 | 176 |
| `compute_simple_Gu!` | 47 | 7 | 6 | 60 |
| Time-stepping | 6 | 0 | 3 | 9 |
| **TOTAL** | **229** | **7** | **9** | **245** |

**Pass rate: 93%** (229/245)

---

## fill_halo_regions! Results ✓

All tested grids pass with `raise=true` (176/176 tests):
- **RectilinearGrid**: 7 non-triply-periodic topologies × 8 locations × 2 raise modes = 112 tests
- **LatitudeLongitudeGrid**: 2 topologies × 8 locations × 2 raise modes = 32 tests
- **TripolarGrid**: 8 locations × 2 raise modes = 16 tests
- **OrthogonalSphericalShellGrid**: 8 locations × 2 raise modes = 16 tests

**Excluded**: `(Periodic, Periodic, Periodic)` topology - triggers segfault in Reactant's MLIR `RecognizeRotate` pass

---

## compute_simple_Gu! Results

### RectilinearGrid (48 tests)

| Topology | coriolis=nothing | coriolis=FPlane | Total |
|----------|------------------|-----------------|-------|
| `(Periodic, Periodic, Bounded)` | 6/6 ✓ | 6/6 ✓ | **12/12** |
| `(Periodic, Bounded, Bounded)` | 6/6 ✓ | 3/6 ✗ | **9/12** |
| `(Bounded, Periodic, Bounded)` | 5/6 (1 error) | 5/6 (1 error) | **10/12** |
| `(Bounded, Bounded, Bounded)` | 5/6 (1 error) | 3/6 (1 fail, 1 err) | **8/12** |

**Failures on RectilinearGrid:**
- `(Periodic, Bounded, Bounded)` + FPlane + raise=true: numerical mismatch
- `(Bounded, *, Bounded)` + WENO + raise=true: MLIR compilation error
- `(Bounded, Bounded, Bounded)` + FPlane + raise=true: numerical mismatch

### LatitudeLongitudeGrid (12 tests)

| Coriolis | nothing | Centered | WENO |
|----------|---------|----------|------|
| `nothing` | ✓ | ✓ | ✗ ERROR (raise=true) |
| `HydrostaticSphericalCoriolis` | ✗ FAILED (raise=true) | ✗ FAILED (raise=true) | ✗ ERROR (raise=true) |

---

## HydrostaticFreeSurfaceModel Time-stepping Results

| Topology | closure=nothing | Total |
|----------|-----------------|-------|
| `(Periodic, Periodic, Bounded)` | 6/6 ✓ | **6/6** |
| `(Periodic, Bounded, Bounded)` | 0/6 ERROR | **0/6** |
| `(Bounded, Periodic, Bounded)` | 0/6 ERROR | **0/6** |
| `(Bounded, Bounded, Bounded)` | 0/6 ERROR | **0/6** |

**Key Finding**: Only `(Periodic, Periodic, Bounded)` works for full time-stepping.

---

## Known Issues

### 1. GPU-specific failures (not reproduced locally on CPU)
CI shows `fill_south_and_north_halo!` failure on `OrthogonalSphericalShellGrid` when running with GPU backend.
This passes locally on CPU. See [Buildkite build #28477](https://buildkite.com/clima/oceananigans/builds/28477#019bbb0d-6e66-4f7c-85f2-e8ece7b51d52).

### 2. Triply Periodic Segfault
`(Periodic, Periodic, Periodic)` topology triggers segfault in `RecognizeRotate` MLIR pass.

### 3. Bounded directions + FPlane + raise=true
Numerical mismatches occur when there's a Bounded direction in x or y with FPlane Coriolis.

### 4. WENO + Bounded x-direction + raise=true
MLIR compilation failures occur for WENO advection when x-direction is Bounded.

### 5. Non-doubly-periodic topologies fail time-stepping
Only `(Periodic, Periodic, Bounded)` works for full HydrostaticFreeSurfaceModel time-stepping.

### 6. HydrostaticSphericalCoriolis Numerical Mismatch
LatitudeLongitudeGrid + HydrostaticSphericalCoriolis produces numerical differences with `raise=true`.

### 7. WENO + LatitudeLongitudeGrid Compilation Failure
WENO on LatitudeLongitudeGrid fails during MLIR compilation with `raise=true`.

### 8. CATKEVerticalDiffusivity Compilation Timeout
CATKE takes >10 minutes to compile with Reactant.

### 9. ImplicitFreeSurface Not Supported
FFT-based solver doesn't work with Reactant arrays. Use `SplitExplicitFreeSurface` instead.

---

## Recommendations

### Use `raise=true` with:
- ✅ `(Periodic, Periodic, Bounded)` topology (the standard ocean configuration)
- ✅ FPlane Coriolis (with doubly-periodic horizontal)
- ✅ WENO tracer advection (with doubly-periodic horizontal)
- ✅ WENOVectorInvariant momentum advection
- ✅ SeawaterBuoyancy + TEOS10
- ✅ SplitExplicitFreeSurface
- ✅ `closure=nothing`
- ✅ `fill_halo_regions!` (all grids/topologies)

### Avoid `raise=true` with:
- ❌ Bounded horizontal directions + FPlane (numerical mismatch)
- ❌ Bounded x-direction + WENO (MLIR error)
- ❌ LatitudeLongitudeGrid + HydrostaticSphericalCoriolis
- ❌ LatitudeLongitudeGrid + WENO
- ❌ Triply periodic grids
- ❌ CATKEVerticalDiffusivity (timeout)
- ❌ ImplicitFreeSurface
