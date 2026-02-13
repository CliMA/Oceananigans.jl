# WENO Momentum Kernel Optimization — Full Summary

**Branch:** `ss/optimize-weno`
**GPU:** Titan V (SM 7.0, 65,536 regs/SM, 2048 max threads/SM)
**Benchmark grid:** 500×200×60, ImmersedBoundaryGrid with GridFittedBottom
**Config:** WENOVectorInvariant (WENO{5} vorticity, WENO{3} vert/div/KE), CATKE, SplitExplicitFreeSurface(substeps=70), TEOS10, tracers (T, S)
**Profiled:** 15-18 kernel invocations per type via nsys

---

## Overview

Two optimizations applied to reduce GPU register pressure in the monolithic momentum tendency kernels:

1. **Reduced-order WENO optimization** (committed) — Replace recursive `buffer_scheme` dispatch with `ifelse(red_order == N, ...)` chains, eliminating redundant FMAs for immersed boundary cells.

2. **Kernel fission** (this session) — Split each monolithic Gu/Gv kernel into 4 sub-kernels, each containing at most one WENO interpolation, giving independent register allocation per sub-kernel.

---

## Register Counts (cuobjdump)

| Kernel | main | reduced-order | split | Occupancy (main → split) |
|--------|:---:|:---:|:---:|:---:|
| **Gu (monolithic)** | 255 | 255 | — | 12.5% |
| **Gv (monolithic)** | 255 | 255 | — | 12.5% |
| u_horizontal (WENO{5} vorticity) | — | — | **128** | → **25.0%** (2 blocks/SM) |
| u_vertical (WENO{3} div + vert) | — | — | **113** | → **25.0%** (2 blocks/SM) |
| u_bernoulli (WENO{3} KE gradient) | — | — | **57** | → **50.0%** (4 blocks/SM) |
| u_nonadvection (no WENO) | — | — | **55** | → **50.0%** (4 blocks/SM) |
| v_horizontal (WENO{5} vorticity) | — | — | **128** | → **25.0%** (2 blocks/SM) |
| v_vertical (WENO{3} div + vert) | — | — | **106** | → **25.0%** (2 blocks/SM) |
| v_bernoulli (WENO{3} KE gradient) | — | — | **48** | → **62.5%** (5 blocks/SM) |
| v_nonadvection (no WENO) | — | — | **55** | → **50.0%** (4 blocks/SM) |

All sub-kernels achieved ≤128 registers. The heaviest (horizontal advection with WENO{5} vorticity) hits exactly 128 — the threshold for 2 blocks/SM on Titan V. Stack usage is 32-56 bytes (Julia error handling frames only, no register spills).

---

## Benchmark Results (median kernel times, ns)

### Individual Kernel Timings

| Kernel | main | reduced-order | split | main → split | optim → split |
|--------|:---:|:---:|:---:|:---:|:---:|
| **Gu (monolithic)** | 3,896,968 | 3,031,962 | — | — | — |
| **Gv (monolithic)** | 3,441,594 | 2,884,171 | — | — | — |
| u_horizontal | — | — | 1,303,912 | | |
| u_vertical | — | — | 1,206,232 | | |
| u_bernoulli | — | — | 321,518 | | |
| u_nonadvect | — | — | 469,150 | | |
| **u_total (split)** | — | — | **3,300,812** | **−15.3%** | **+8.9%** |
| v_horizontal | — | — | 717,819 | | |
| v_vertical | — | — | 1,071,850 | | |
| v_bernoulli | — | — | 287,182 | | |
| v_nonadvect | — | — | 466,701 | | |
| **v_total (split)** | — | — | **2,543,552** | **−26.1%** | **−11.8%** |

### Aggregate Momentum Tendency Time (per invocation, median)

| Configuration | Gu time | Gv time | Total momentum | vs main | vs optim |
|:---|:---:|:---:|:---:|:---:|:---:|
| **main** | 3,896,968 | 3,441,594 | **7,338,562** | — | — |
| **reduced-order** | 3,031,962 | 2,884,171 | **5,916,133** | **−19.4%** | — |
| **split** | 3,300,812 | 2,543,552 | **5,844,364** | **−20.3%** | **−1.2%** |

### Non-Momentum Kernels (unchanged)

| Kernel | main | reduced-order | split | Notes |
|--------|:---:|:---:|:---:|:---|
| Gc (T, median of 3) | ~2,866,000 | ~2,573,000 | ~2,568,000 | −10.4% vs main |
| CATKE diffusivities | 2,154,675 | 2,159,137 | 2,153,187 | Unchanged |
| TKE diffusivity | 1,673,974 | 1,676,340 | 1,669,046 | Unchanged |

---

## Analysis

### What the split achieved

1. **Register pressure eliminated:** All 8 sub-kernels at ≤128 regs (from 255), achieving 2-5 blocks/SM (from 1).
2. **Gv improved significantly:** −11.8% vs reduced-order monolithic. The v-direction stencils benefit most from doubled occupancy (better latency hiding on memory accesses).
3. **Net momentum improvement vs main:** −20.3%, combining the reduced-order ifelse optimization (−19.4%) with the kernel fission benefit.
4. **Non-WENO paths unaffected:** Dispatch on `VectorInvariantUpwindVorticity` means EnergyConserving/EnstrophyConserving schemes use the original monolithic path.

### Why Gu didn't improve (and got slightly slower)

The u-direction split sub-kernels sum to 3,300,812 ns vs the monolithic 3,031,962 ns (+8.9%). Two factors:
1. **Extra memory traffic:** Each accumulation sub-kernel reads and writes Gu (an extra ~460MB of global memory traffic across 3 accumulation kernels).
2. **Kernel launch overhead:** 4 launches instead of 1 adds ~15-20 μs.
3. **u_vertical is heavy:** 1,206,232 ns for vertical advection (113 regs, 2 blocks/SM) — this contains TWO WENO{3} interpolations (divergence flux + vertical momentum flux), which is significant.

The occupancy improvement (12.5% → 25%) helps hide memory latency, but the extra memory traffic from the accumulation pattern (`Gu[i,j,k] += ...`) partially offsets this gain for the u-direction.

### Why Gv improved but Gu didn't

The asymmetry comes from the horizontal advection kernel:
- **u_horizontal:** 1,303,912 ns — reconstructs vorticity in the y-direction
- **v_horizontal:** 717,819 ns — reconstructs vorticity in the x-direction

The y-direction WENO stencil accesses memory in a non-contiguous pattern (strided across the y-dimension), while x-direction accesses are more cache-friendly. At 2 blocks/SM, the v-direction sub-kernels can overlap memory latency much more effectively.

---

## Per-Timestep Impact

For a single timestep (18 invocations of each momentum kernel):

| | main | reduced-order | split | Δ (main→split) |
|:---|:---:|:---:|:---:|:---:|
| Momentum total | 132.1 ms | 106.5 ms | 105.2 ms | **−26.9 ms (−20.3%)** |
| Gc total (3×15) | 128.9 ms | 115.5 ms | 115.1 ms | **−13.8 ms (−10.7%)** |
| Other kernels | ~389 ms | ~389 ms | ~389 ms | ~0 |
| **Total GPU time** | **~650 ms** | **~611 ms** | **~609 ms** | **~−41 ms (−6.3%)** |

The reduced-order optimization delivers most of the speedup (−39 ms). Kernel fission adds a modest further −1.3 ms from the Gv improvement.

---

## Files Modified

### Committed (reduced-order optimization)
Already on branch from previous sessions — ifelse-based dynamic order reduction in WENO stencil code.

### New (kernel fission + infrastructure fixes)

| File | Changes |
|------|---------|
| `src/Models/HydrostaticFreeSurfaceModels/compute_hydrostatic_free_surface_tendencies.jl` | Refactored `compute_hydrostatic_momentum_tendencies!` to dispatch on advection type via `_compute_hydrostatic_momentum_tendencies!` |
| `src/Models/HydrostaticFreeSurfaceModels/split_hydrostatic_momentum_tendencies.jl` | **New file** — 8 sub-kernels, 2 nonadvection inline functions, split dispatch for `VectorInvariantUpwindVorticity` |
| `src/Models/HydrostaticFreeSurfaceModels/HydrostaticFreeSurfaceModels.jl` | Added `include("split_hydrostatic_momentum_tendencies.jl")` after line 174 |
| `src/Advection/flux_form_advection.jl` | Added `on_architecture` method for `FluxFormAdvection` (infrastructure fix) |
| `src/Models/NonhydrostaticModels/nonhydrostatic_model.jl` | Added `on_architecture` import + call for advection in constructor |
| `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl` | Added `on_architecture` import + call for advection in constructor |

### Reverted (LUT optimization — post-mortem)
The LUT optimization (replacing coefficient ifelse cascades with CuDeviceVector lookup tables) was implemented, tested, benchmarked, and found to be a net regression for momentum kernels (+5-10% slower) despite eliminating 70% of selp instructions. Root cause: `selp` is ~4 cycles (register-register) vs `ld.global` ~28 cycles; at 1 block/SM there's no latency hiding. Reverted in this session.

---

## Architecture

### Kernel fission dispatch flow

```
compute_momentum_tendencies!(model, ...)
  └─ compute_hydrostatic_momentum_tendencies!(model, ...)
       └─ _compute_hydrostatic_momentum_tendencies!(advection, model, ...)
            ├─ [VectorInvariantUpwindVorticity]  → 8 sub-kernels (split path)
            └─ [fallback]                         → 2 monolithic kernels (original)
```

### Sub-kernel execution order (per velocity component)

```
Gu[i,j,k]  = -horizontal_advection_U(...)   # kernel 1 (=, WENO{5} vorticity)
Gu[i,j,k] -= vertical_advection_U(...)      # kernel 2 (-=, WENO{3} div+vert)
Gu[i,j,k] -= bernoulli_head_U(...)          # kernel 3 (-=, WENO{3} KE)
Gu[i,j,k] += nonadvection(...)              # kernel 4 (+=, pressure+coriolis+diffusion+forcing)
```

Kernel 1 uses `=` (overwrite); kernels 2-4 use `-=`/`+=` (accumulate). GPU in-order stream execution guarantees correctness without explicit synchronization.

---

## Potential Further Optimizations

1. **Fuse bernoulli + nonadvection:** These two sub-kernels are cheap (287-321 μs and 466-469 μs). Fusing them would eliminate one kernel launch and one Gu read-write cycle, saving ~10-20 μs.

2. **Split vertical advection further:** The vertical advection kernel contains TWO WENO{3} interpolations (divergence flux and vertical momentum flux) and uses 106-113 regs. Splitting into two sub-kernels might bring each below 64 regs → 4 blocks/SM.

3. **Memory access optimization:** The u_horizontal_advection kernel is 1.8× slower than v_horizontal due to y-direction stride patterns. Shared memory tiling could improve this.

4. **Explore `maxregs=64`:** With 4+ blocks/SM, the warp scheduler has enough concurrent warps to fully hide memory latency. Forcing lower register counts with controlled spilling might be net positive at high occupancy.
