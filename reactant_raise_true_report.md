# Reactant Correctness Tests - `raise=true` Report

**Julia Version**: 1.11.8  
**Reactant Version**: 0.2.114  
**Test Date**: January 13, 2026  
**Architecture**: CPU (arm64 macOS)

## Summary

Most tested configurations produce **machine-precision** results when comparing vanilla Oceananigans vs ReactantState.

| Test Category | Passed | Failed | Error | Total |
|--------------|--------|--------|-------|-------|
| `fill_halo_regions!` | 176 | 0 | 0 | 176 |
| `compute_simple_Gu!` | 20 | 2 | 2 | 24 |
| Time-stepping | 6 | 0 | 0 | 6 |
| **TOTAL** | **202** | **2** | **2** | **206** |

**Pass rate: 98%** (202/206)

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

### RectilinearGrid - All Pass ✓ (12/12 tests)

| Coriolis | Advection | raise=false | raise=true |
|----------|-----------|-------------|------------|
| `nothing` | `nothing` | ✓ | ✓ |
| `nothing` | `Centered` | ✓ | ✓ |
| `nothing` | `WENO` | ✓ | ✓ |
| `FPlane` | `nothing` | ✓ | ✓ |
| `FPlane` | `Centered` | ✓ | ✓ |
| `FPlane` | `WENO` | ✓ | ✓ |

### LatitudeLongitudeGrid - Partial Pass (8/12 tests)

| Coriolis | Advection | raise=false | raise=true |
|----------|-----------|-------------|------------|
| `nothing` | `nothing` | ✓ | ✓ |
| `nothing` | `Centered` | ✓ | ✓ |
| `nothing` | `WENO` | ✓ | ✗ ERROR (MLIR) |
| `HydrostaticSphericalCoriolis` | `nothing` | ✓ | ✗ FAILED (δ~1e-6) |
| `HydrostaticSphericalCoriolis` | `Centered` | ✓ | ✗ FAILED (δ~1e-6) |
| `HydrostaticSphericalCoriolis` | `WENO` | ✓ | ✗ ERROR (MLIR) |

---

## Time-stepping Tests ✓

### HydrostaticFreeSurfaceModel with RectilinearGrid

Full configuration tested and **PASSED** with machine precision (δ ~ 1e-15):

| Component | Value |
|-----------|-------|
| Grid | RectilinearGrid (8×8×4, halo=6,6,3) |
| Coriolis | FPlane(f=1e-4) |
| Tracer Advection | WENO |
| Momentum Advection | WENOVectorInvariant |
| Equation of State | TEOS10 |
| Buoyancy | SeawaterBuoyancy |
| Tracers | T, S |
| Free Surface | SplitExplicitFreeSurface (10 substeps) |
| Closure | `nothing` |
| Time steps | 3 × 60s |

**Results after 3 time steps:**
- u: max|δ| = 3.3e-15 ✓
- v: max|δ| = 4.3e-15 ✓
- T: max|δ| = 3.6e-14 ✓
- S: max|δ| = 3.8e-13 ✓

### Ablation Test Summary

All components tested individually - all pass with machine precision:

| Configuration | Result |
|--------------|--------|
| Minimal (no physics) | ✓ PASSED (δ=0) |
| + FPlane coriolis | ✓ PASSED (δ=0) |
| + tracer with WENO | ✓ PASSED (δ=0) |
| + WENOVectorInvariant momentum | ✓ PASSED (δ~1e-17) |
| + BuoyancyTracer | ✓ PASSED (δ~1e-17) |
| + SeawaterBuoyancy + LinearEOS | ✓ PASSED (δ~1e-16) |
| + SeawaterBuoyancy + TEOS10 | ✓ PASSED (δ~1e-15) |

---

## Known Issues

### 1. GPU-specific failures (not reproduced locally on CPU)
CI shows `fill_south_and_north_halo!` failure on `OrthogonalSphericalShellGrid` when running with GPU backend.
This passes locally on CPU. See [Buildkite build #28477](https://buildkite.com/clima/oceananigans/builds/28477#019bbb0d-6e66-4f7c-85f2-e8ece7b51d52).

### 2. Triply Periodic Segfault
`(Periodic, Periodic, Periodic)` topology triggers segfault in `RecognizeRotate` MLIR pass.

### 2. HydrostaticSphericalCoriolis Numerical Mismatch
LatitudeLongitudeGrid + HydrostaticSphericalCoriolis produces δ ≈ 8.6e-06 at grid point (1,1,1).

### 3. WENO + LatitudeLongitudeGrid Compilation Failure
WENO on LatitudeLongitudeGrid fails during MLIR compilation.

### 4. CATKEVerticalDiffusivity Compilation Timeout
CATKE takes >10 minutes to compile with Reactant.

### 5. ImplicitFreeSurface Not Supported
FFT-based solver doesn't work with Reactant arrays. Use `SplitExplicitFreeSurface` instead.

---

## Requirements

- **CUDA.jl must be loaded**: Even on CPU, `using CUDA` is required for `raise=true`

---

## Recommendations

### Use `raise=true` with:
- ✅ RectilinearGrid (non-triply-periodic)
- ✅ FPlane Coriolis
- ✅ WENO tracer advection
- ✅ WENOVectorInvariant momentum advection
- ✅ SeawaterBuoyancy + TEOS10
- ✅ SplitExplicitFreeSurface
- ✅ `closure=nothing`
- ✅ `fill_halo_regions!`

### Avoid `raise=true` with:
- ❌ LatitudeLongitudeGrid + HydrostaticSphericalCoriolis
- ❌ LatitudeLongitudeGrid + WENO
- ❌ Triply periodic grids
- ❌ CATKEVerticalDiffusivity (timeout)
- ❌ ImplicitFreeSurface
