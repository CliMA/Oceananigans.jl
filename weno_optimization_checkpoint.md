# WENO Register Optimization — Checkpoint

**Branch:** `ss/optimize-weno`
**Hardware:** NVIDIA V100 (SM70), 65536 regs/SM, 255 max regs/thread
**Date:** 2026-02-11

---

## 1. Current State

The dynamic reduced-order WENO optimization (replacing recursive `buffer_scheme` with `ifelse`-based `red_order` selection) is implemented and working. Benchmarks show **22% Gu speedup, 16% Gv speedup, 10-12% Gc speedup** — but register pressure increased from 216→255 (Gu) and 200→254 (Gv) vs `main`.

**No spills.** Occupancy unchanged (both at 1 block/SM = 12.5%).

---

## 2. Baseline Register Counts

### Physical registers (from `cuobjdump --dump-resource-usage`):

| Kernel | `main` branch | `ss/optimize-weno` | Delta |
|--------|:---:|:---:|:---:|
| **Gu** (u-momentum, WENO{5} VectorInvariant) | 216 | 255 | +39 |
| **Gv** (v-momentum, WENO{5} VectorInvariant) | 200 | 254 | +54 |
| **Gc** (tracer, WENO{4} order=7) | ~104 | 104 | ~0 |

### PTX virtual registers (Gu, main entry point):

| Register | `ss/optimize-weno` |
|----------|:---:|
| `.f64 %fd` | 2981 |
| `.b64 %rd` | 1387 |
| `.pred %p` | 268 |
| `.f32 %f` | 80 |
| `.b32 %r` | 32 |

### Key instruction counts (Gu kernel):

| Instruction | `main` | `ss/optimize-weno` | Delta |
|-------------|:---:|:---:|:---:|
| `fma.rn.f64` | 1,049 | 640 | **-39%** |
| `selp` (total) | 271 | 1,031 | **+281%** |
| `setp` | 129 | 171 | +33% |
| `ld.global` | 246 | 258 | +5% |
| `bra` | 10 | 10 | Same |
| `call.uni` | 0 | 0 | Same |
| `st.local` / `ld.local` | 0 / 0 | 0 / 0 | **No spills** |

### File sizes:

| Kernel | PTX (bytes) | cubin (bytes) |
|--------|:---:|:---:|
| Gu | 314,215 | 433,192 |
| Gv | 310,835 | 425,128 |
| Gc | 916,589 | 492,200 |

---

## 3. PTX `selp` Breakdown (Gu kernel, ~967 selps in main entry)

| Category | Count | % of total | Description |
|----------|:---:|:---:|-------------|
| **Literal constant pair selps** | 419 | 43% | `selp.f64 %fd, 0d<hex>, 0d<hex>, %p` — coefficient cascades selecting between two f64 literal constants. These are reconstruction_coefficients, smoothness_coefficients, and C★ optimal weights. |
| **Zero-clipping selps** | 316 | 33% | `selp.f64 %fd, 0d0000..., %fd_computed, %p` — selecting between 0.0 and a computed value. Used for immersed boundary masking and zeroing inactive stencils. |
| **Register-register selps** | 226 | 23% | `selp.f64 %fd, %fd_a, %fd_b, %p` — selecting between two computed values. Used for stencil extraction (LeftBias/RightBias selection in S₀₅, S₁₅, etc.) and intermediate cascade results. |
| **Int32 selps** | 6 | 1% | Smoothness coefficient selections (Int32 values) |
| **Total** | ~967 | 100% | |

### Key insight: ~76% of selps (735/967) are coefficient-related

The 419 literal-constant selps + 316 zero-clipping selps are the direct cost of the `ifelse(red_order == N, ...)` cascades in:
- `reconstruction_coefficients()` (weno_coefficients.jl lines 92-205)
- `smoothness_coefficients()` (weno_smoothness.jl lines 126-238)
- `C★()` optimal weights (weno_coefficients.jl lines 239-267)

### Cascade structure in PTX

For each coefficient tuple element, the ifelse chain generates a cascade of `selp.f64` instructions:

```ptx
// Level 1: selp between two literal constants (red_order == 4 vs 5)
selp.f64 %fd1597, 0d3FCFFFFFFFFFFFFC, 0d3FDCCCCCCCCCCCCD, %p95;

// Level 2: selp with previous result + new literal (red_order == 3 vs above)
selp.f64 %fd1601, 0d3FFD555555555556, %fd1598, %p94;

// Level 3: selp with previous result + new literal (red_order == 2 vs above)
selp.f64 %fd1605, %fd1601, <value>, %p93;
```

Each cascade level adds an intermediate register that must stay live until consumed. For WENO{5}:
- 4 cascade levels per coefficient element
- 5 elements per coefficient tuple
- 5 stencils per WENO reconstruction
- = up to 100 simultaneous live intermediate registers during peak pressure

**Estimated register consumption from coefficient cascades: ~100-120 of 255 physical registers.**

---

## 4. Why Coefficients Were Cheap in Old Approach (main branch)

In the old `buffer_scheme` approach:
- Each `WENO{N}` specialization had nested buffer schemes: `WENO{5}` → `WENO{4}` → `WENO{3}` → ...
- Near boundaries, the code dispatched to `scheme.buffer_scheme` (a separate compiled specialization)
- Each specialization knew its coefficients **at compile time**: `stencil_coefficients(FT, 50, stencil, ...)` was evaluated during `@eval` and the result was a literal constant in the generated code
- **No ifelse chains needed** — the dispatch to the right buffer scheme was done via Julia's method dispatch (before kernel launch or through separate kernel paths)
- The compiler embedded coefficients as **immediate operands** in FMA instructions, consuming **zero registers**
- This is what the user refers to as "unified registers" — compile-time constants are either immediates or in constant memory, shared across all threads with no per-thread register cost

In the new approach:
- A single `WENO{5}` handles all orders dynamically via `red_order::Int`
- ALL possible coefficient values for ALL orders must be available simultaneously
- `ifelse(red_order == N, coeff_A, coeff_B)` → `selp.f64` chains
- Each literal constant becomes a per-thread register value during the cascade
- **The same data that was "free" (embedded immediates) now consumes ~100-120 physical registers**

The trade-off was worthwhile (22% speedup from -39% FMAs outweighs the register cost), but the register ceiling is now a concern.

---

## 5. Failed Optimization: Fused Raw-Stencil (Opportunity 1)

### What was implemented
Replaced the N-substencil tuple materialization with on-demand extraction from a raw stencil S:
- Added `weno_raw_stencil_x/y/z` (loads raw 2*buffer values)
- Added `fused_beta_from_raw` and `fused_reconstruction_from_raw` (extract substencils on-demand)
- Modified all 4 `biased_interpolate_*` methods

### Results

| Metric | Pre-fused | Fused | Delta |
|--------|:---:|:---:|:---:|
| Gu physical regs | 255 | 255 | **0** |
| Gv physical regs | 254 | 255 | **+1** |
| Gc physical regs | 104 | 102 | **-2** |
| Gu PTX `.f64` vregs | 2981 | 2838 | -143 (-4.8%) |
| Gu PTX `.b64` vregs | 1387 | 1153 | -234 (-16.9%) |
| Gu PTX size | 314 KB | 465 KB | **+48%** |
| Gu fma.rn.f64 | 640 | 640 | 0 |
| Gu selp | 1031 | 1032 | +1 |
| Gu call.uni | 0 | 8 | +8 (error paths only) |
| Gu st.local | 0 | 15 | +15 (error paths only) |
| Gu STACK | 0 | 32 | +32 B (call frames) |

### Why it failed
The ptxas register allocator was already effectively collapsing the old virtual registers. The substencil tuple storage was never the physical bottleneck — it was a virtual register namespace issue, not a live-range issue. The compiler could reuse registers across sequential stencil computations regardless of how substencils were organized.

### Side effects
- PTX size increased ~48% (functions inlined but generated more code)
- 8 `call.uni` instructions introduced (bounds-check error paths: `julia_throw_boundserror`, `vprintf`, `gpu_report_exception`)
- 15 `st.local` stores (register saves around error-path calls)
- All side effects were on unreachable error paths — no impact on normal execution

### Decision: **REVERTED** — no benefit, increased code size.

---

## 6. Optimization Opportunities (Updated)

### NOT recommended:
| # | Opportunity | Why not |
|---|-------------|---------|
| 1 | Fused raw-stencil | **Failed** — doesn't reduce physical registers (ptxas already optimal) |
| 2 | Direct index-into-S reconstruction | Same mechanism as #1, likely same result |
| 4 | Full-order fast path (branch) | User advises against it — branch divergence is significant near immersed boundaries |

### Recommended (in priority order):

#### A. Coefficient Lookup Tables (NEW — replaces Opportunities 2 & 5)
**Replace ifelse coefficient cascades with indexed memory loads.**

Currently, `reconstruction_coefficients`, `smoothness_coefficients`, and `C★` use nested ifelse chains that generate ~735 selps consuming ~100-120 physical registers.

**Approach:** Store all coefficient values in flat lookup tables (NTuple or CuArray), indexed by `(red_order, stencil)`. Replace:
```julia
ifelse(red_order == 1, coeff_A,
ifelse(red_order == 2, coeff_B,
ifelse(red_order == 3, coeff_C,
                       coeff_D)))
```
with:
```julia
coeff_table[red_order]  # Single indexed access
```

**Implementation options:**
1. **NTuple-of-NTuple in const (force local memory):** Large tuples (>~8 elements) indexed by runtime value get stored in local memory by LLVM. A flat `NTuple{125, Float64}` indexed by computed offset would generate `ld.local` instead of selp cascades. Local memory on V100 is L1-cached (~28 cycles) — slower than selp (~4 cycles) per access, but eliminates ~100-120 registers.

2. **CuArray in global memory:** Pre-allocate a small coefficient table (~3KB) as a `CuArray{Float64}`, store the device pointer in the WENO struct. All threads access the same data (warp-broadcast from L1 cache). Requires `Adapt.jl` support.

3. **Constant memory via CUDA.jl:** Use `CUDA.CuStaticSharedArray` or mark arrays as `__constant__`. 64KB constant memory limit is more than enough. Requires CUDA.jl API.

4. **Shared memory:** Load coefficient tables into shared memory at block start. Fast broadcast to all threads. Requires kernel restructuring.

**Storage needed for WENO{5}:**
- Reconstruction: 5 stencils × 5 red_orders × 5 coefficients × 8B = 1000 B
- Smoothness: 5 stencils × 5 red_orders × 15 coefficients × 4B = 1500 B
- Optimal weights: 5 stencils × 5 red_orders × 8B = 200 B
- **Total: ~2.7 KB** (easily fits in L1/constant/shared memory)

**Note:** Current kernel args already use 4072 B of CONSTANT[0] (close to 4KB limit), so adding to the struct may exceed constant memory for kernel args. CuArray approach (passing a device pointer) avoids this.

**Estimated savings:** ~100-120 physical registers (eliminates selp cascade intermediates). This alone could bring Gu from 255 to ~140-155 registers. Combined with Opportunity 3, possibly reaching 128 (2 blocks/SM threshold).

**Risk:** Memory latency replaces register-based computation. On V100, L1 cache hit = ~28 cycles vs selp = ~4 cycles. But the GPU can hide this latency through warp scheduling if occupancy improves. Net performance depends on the occupancy trade-off.

#### B. Reload Grid Metrics (Opportunity 3)
Grid metrics loaded at kernel entry persist through the entire WENO computation but are only used at the beginning and end. Reloading from global memory before final assembly frees ~7 f64 (~14 physical regs) during peak WENO phases.

**Estimated savings:** ~14 physical registers
**Risk:** Compiler may re-hoist loads.

#### C. Float32 Smoothness (Opportunity 5)
Compute β smoothness indicators in Float32. Only ratios matter (normalized to ω), so Float32 precision (~7 digits) is sufficient.

**Estimated savings:** ~10 physical registers
**Risk:** Edge cases with extreme β ratios near sharp discontinuities.

#### D. Fuse IB Mask (Opportunity 6)
Fuse immersed boundary masking into the smoothness/reconstruction compute instead of materializing masked stencils.

**Estimated savings:** ~16 physical registers
**Complexity:** High — deeply integrated into stencil loading pipeline.

---

## 7. Path to 2 Blocks/SM (128 registers)

| Step | Optimization | Registers after | Blocks/SM |
|------|-------------|:---:|:---:|
| Current | (baseline) | 255 | 1 (12.5%) |
| 1 | Coefficient lookup tables | ~140-155 | 1 (12.5%) |
| 2 | + Reload grid metrics | ~130-140 | 1 (12.5%) |
| 3 | + Float32 smoothness | ~125-135 | **potentially 2 (25%)** |

Reaching exactly 128 registers requires all three optimizations and may need additional tuning. The coefficient lookup table is by far the biggest lever.

---

## 8. Key Files

| File | Role |
|------|------|
| `src/Advection/weno_interpolants.jl` | Main WENO pipeline: stencil loading → smoothness → weights → reconstruction |
| `src/Advection/weno_coefficients.jl` | Reconstruction coefficients (RS*** tuples) + C★ optimal weights, all via ifelse chains |
| `src/Advection/weno_smoothness.jl` | Smoothness coefficients (SS*** tuples) + global_smoothness_indicator, all via ifelse chains |
| `src/Advection/weno_stencils.jl` | Substencil extraction functions (S₀₂ through S₅₆) |
| `src/Advection/weno_reconstruction.jl` | WENO struct definition, constructors, Adapt methods |
| `extract_ptx.jl` | Script for PTX/cubin extraction + register analysis |
| `ptx_optimized/` | PTX/cubin files from `ss/optimize-weno` (pre-fused baseline) |
| `ptx_fused/` | PTX/cubin files from failed fused optimization (can be deleted) |

---

## 9. WENO Struct (Current)

```julia
struct WENO{N, FT, FT2, PP, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT}
    bounds :: PP                       # Bounds for MPS-WENO (usually nothing)
    advecting_velocity_scheme :: SI    # Centered scheme for advecting velocity
end
```

Kernel argument size: CONSTANT[0] = 4072 bytes (close to 4KB limit).
Adding coefficient tables directly to the struct would likely exceed the constant memory limit.

---

## 10. Coefficient Scale (for WENO{5}, the VectorInvariant momentum scheme)

### Reconstruction coefficients (Float64):
- 5 stencils × 5 possible red_orders × 5 coefficients per stencil = 125 Float64 values
- Many are zero (lower-order stencils have zero-padded coefficients)
- Currently: 5 functions with 4-level-deep ifelse chains returning NTuple{5, Float64}

### Smoothness coefficients (Int32):
- 5 stencils × 5 possible red_orders × 15 coefficients per stencil = 375 Int32 values
- Stored as flattened upper-triangular matrix
- Many are zero (lower-order stencils have zero-padded coefficients)
- Currently: 5 functions with 4-level-deep ifelse chains returning NTuple{15, Int32}

### Optimal weights C★ (Float64):
- 5 stencils × 5 possible red_orders × 1 weight = 25 Float64 values
- Currently: 5 functions with 4-level-deep ifelse chains returning Float64 scalars

### Total selp budget per WENO interpolation call:
- Reconstruction: 5 stencils × 5 coefficients × 4 selps = 100 selp.f64
- Smoothness: 5 stencils × 15 coefficients × 4 selps = 300 selp (Int32→f64 promotion optimizes some away)
- Optimal weights: 5 stencils × 1 × 4 selps = 20 selp.f64
- **~420 selps per interpolation** (matches the 419 literal-constant selps observed)

The Gu kernel calls WENO interpolation ~6-8 times (for different velocity components and directions), but the compiler may share some coefficient computations across calls if the inputs are the same.
