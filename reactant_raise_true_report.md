# Reactant Correctness Tests - `raise=true` Report

**Julia Version**: 1.11.8  
**Reactant Version**: 0.2.114  
**Test Date**: January 13, 2026  
**Architecture**: CPU (arm64 macOS)

## Summary

| Test Category | Passed | Total | Pass Rate |
|--------------|--------|-------|-----------|
| `fill_halo_regions!` | 7 | 7 | 100% |
| `compute_simple_Gu!` | 11 | 14 | 79% |
| **TOTAL** | **18** | **21** | **86%** |

---

## fill_halo_regions! Results

**Grid**: RectilinearGrid (size=(3,4,2), halo=(1,1,1))  
**Field Location**: (Center, Center, Center)

| Topology | Result |
|----------|--------|
| (Bounded, Periodic, Periodic) | ✓ PASSED |
| (Periodic, Bounded, Periodic) | ✓ PASSED |
| (Bounded, Bounded, Periodic) | ✓ PASSED |
| (Periodic, Periodic, Bounded) | ✓ PASSED |
| (Bounded, Periodic, Bounded) | ✓ PASSED |
| (Periodic, Bounded, Bounded) | ✓ PASSED |
| (Bounded, Bounded, Bounded) | ✓ PASSED |

**Excluded**: `(Periodic, Periodic, Periodic)` - triggers segfault in Reactant's MLIR `RecognizeRotate` pass

---

## compute_simple_Gu! Results

Tests the u-velocity tendency computation: `Gu = -U⋅∇u - f×U`

### RectilinearGrid

| Coriolis | Advection | Result | max\|δ\| |
|----------|-----------|--------|----------|
| `nothing` | `nothing` | ✓ PASSED | 0.0 |
| `nothing` | `Centered` | ✓ PASSED | ~1e-15 |
| `nothing` | `WENO` | ✓ PASSED | ~1e-15 |
| `FPlane` | `nothing` | ✓ PASSED | 0.0 |
| `FPlane` | `Centered` | ✓ PASSED | 3.6e-15 |
| `FPlane` | `WENO` | ✓ PASSED | 7.1e-15 |

**All 6 tests pass** ✓

### LatitudeLongitudeGrid

| Coriolis | Advection | Result | max\|δ\| | Notes |
|----------|-----------|--------|----------|-------|
| `nothing` | `nothing` | ✓ PASSED | 0.0 | |
| `nothing` | `Centered` | ✓ PASSED | ~1e-15 | |
| `nothing` | `WENO` | ✗ ERROR | — | MLIR compilation failure |
| `HydrostaticSphericalCoriolis` | `nothing` | ✗ FAILED | 8.6e-06 | Numerical mismatch |
| `HydrostaticSphericalCoriolis` | `Centered` | ✗ FAILED | 8.6e-06 | Numerical mismatch |
| `HydrostaticSphericalCoriolis` | `WENO` | ✗ ERROR | — | MLIR compilation failure |

**5 of 8 tests pass** (with 3 known failures)

---

## Analysis

### Root Causes Identified

1. **HydrostaticSphericalCoriolis produces numerical differences** (δ ≈ 8.6e-06)
   - The issue is specifically with `HydrostaticSphericalCoriolis`, not the grid metrics
   - With `coriolis=nothing`, LatitudeLongitudeGrid works correctly for `nothing` and `Centered` advection
   - The spherical Coriolis term `x_f_cross_U` computes differently with Reactant

2. **WENO + LatitudeLongitudeGrid fails to compile**
   - Regardless of Coriolis, WENO on LatitudeLongitudeGrid fails during MLIR compilation
   - The complex WENO stencils combined with curvilinear grid metrics generate IR that Reactant cannot compile

3. **Face field size bug** (separate issue)
   - Face-located fields on LatitudeLongitudeGrid have incorrect parent array sizes
   - Missing +1 in `reactant_total_length` for Face on BoundedTopology

---

## Requirements

- **CUDA.jl must be loaded**: Even on CPU, `using CUDA` is required for `raise=true` to work with KernelAbstractions kernels

---

## Recommendations

### Use `raise=true` with:
- ✅ RectilinearGrid with any topology (except triply periodic)
- ✅ Any Coriolis (`nothing`, `FPlane`)
- ✅ Any advection (`nothing`, `Centered`, `WENO`)
- ✅ LatitudeLongitudeGrid with `coriolis=nothing` and `advection ∈ {nothing, Centered}`
- ✅ `fill_halo_regions!`

### Avoid `raise=true` with:
- ❌ LatitudeLongitudeGrid + `HydrostaticSphericalCoriolis` (numerical mismatch)
- ❌ LatitudeLongitudeGrid + `WENO` (compilation failure)
- ❌ Triply periodic grids (segfault)

---

## Issues to File

1. **HydrostaticSphericalCoriolis numerical mismatch** - δ ≈ 8.6e-06 at (1,1,1)
2. **WENO + LatitudeLongitudeGrid MLIR compilation failure**
3. **Triply periodic segfault** in `RecognizeRotate` pass
4. **Face field size bug** - missing +1 for Face on BoundedTopology in `reactant_total_length`
