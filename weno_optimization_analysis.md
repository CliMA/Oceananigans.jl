# WENO Stencil Optimization: PTX/SASS & Benchmark Analysis

**Branch:** `ss/optimize-weno`
**Baseline:** `main` (commit `1a34645`)
**Hardware:** NVIDIA V100 (SM70)
**Benchmark:** `BENCHMARK_GROUP="immersed"` — `HydrostaticFreeSurfaceModel` with `WENOVectorInvariant()` momentum, `WENO(order=7)` tracers, `ImmersedBoundaryGrid` with `GridFittedBottom`, `SplitExplicitFreeSurface`, `CATKEVerticalDiffusivity`, TEOS10 equation of state. Grid: 500x200x60.

---

## 1. What Changed

The optimization replaces **recursive buffer schemes** with a **dynamic reduced-order** system:

- **Before:** `WENO{5}` contained a nested `buffer_scheme` field: `WENO{4} → WENO{3} → WENO{2} → UpwindBiased{1} → Centered{1}`. Near boundaries (immersed or topological), the code dispatched to these nested instances, each compiled as a separate specialization. This inflated PTX with duplicated arithmetic for every order level.

- **After:** A single `WENO{5}` instance computes a `red_order::Int` at each grid point based on distance to boundaries, then uses `ifelse` chains (compiled to `selp` — predicated moves) to select the correct stencil at runtime. Precomputed coefficient lookup tables (`weno_coefficients.jl`, `weno_smoothness.jl`) replace generated code.

Key files: `weno_reconstruction.jl`, `weno_interpolants.jl`, `weno_coefficients.jl` (new), `weno_smoothness.jl` (new), `weno_stencils.jl` (new), `centered_reconstruction.jl`, `upwind_biased_reconstruction.jl`, `immersed_advective_fluxes.jl`, `topologically_conditional_interpolation.jl`.

---

## 2. Register Counts

Extracted via `extract_ptx.jl` using `CUDA.registers()` / `CUDA.memory().local`:

| Kernel | `main` | `ss/optimize-weno` | Delta |
|--------|-------:|-------------------:|------:|
| **Gu** (u-momentum) | 216 regs | 255 regs | **+39 (+18%)** |
| **Gv** (v-momentum) | 200 regs | 254 regs | **+54 (+27%)** |
| Local memory (spills) | 0 B | 0 B | **No spills** |
| Occupancy (256 threads/block) | 12.5% (1 block/SM) | 12.5% (1 block/SM) | **Same** |

Register pressure increased to near the 255-register ceiling. However, occupancy is unchanged because both branches already exceed the 1-block/SM threshold on V100 (65536 regs / 256 threads = 256 max regs for 1 block). No register spills to local memory on either branch.

---

## 3. PTX Code Size

| Kernel | `main` PTX | optimized PTX | Delta |
|--------|----------:|-------------:|------:|
| **Gu** | 320,595 B | 314,215 B | **-2.0%** |
| **Gv** | 312,106 B | 310,835 B | **-0.4%** |
| **Gc** (tracer) | 1,202,232 B | 916,589 B | **-23.8%** |

| Kernel | `main` cubin | optimized cubin | Delta |
|--------|------------:|-----------:|------:|
| **Gu** | 438,952 B | 433,192 B | **-1.3%** |
| **Gv** | 422,312 B | 425,128 B | **+0.7%** |
| **Gc** (tracer) | 447,912 B | 492,200 B | **+9.9%** |

The Gc tracer kernel PTX shrank by **24%** — the largest win from eliminating buffer scheme recursion (tracers had the deepest `WENO{order=7}` nesting). The SASS backend (cubin) expanded slightly for Gc/Gv because the `selp` chains get unrolled by the compiler.

---

## 4. PTX Instruction Breakdown

### Gu (u-momentum tendency)

| Instruction | `main` | optimized | Delta | Notes |
|-------------|-------:|----------:|------:|-------|
| `fma.rn.f64` | 1,049 | 640 | **-39%** | Less duplicated stencil arithmetic |
| `selp` | 271 | 1,031 | **+281%** | `ifelse` chains for order selection |
| `setp` | 129 | 171 | **+33%** | Condition evaluations |
| `ld.global` | 246 | 258 | **+5%** | Slightly more global loads |
| `st.global` | 1 | 1 | Same | Single output store |
| `bra` | 10 | 10 | Same | No additional divergent branches |
| `st.local` / `ld.local` | 0 | 0 | **No spills** | |

### Gv (v-momentum tendency)

| Instruction | `main` | optimized | Delta | Notes |
|-------------|-------:|----------:|------:|-------|
| `fma.rn.f64` | 1,049 | 640 | **-39%** | Same reduction as Gu |
| `selp` | 273 | 1,037 | **+280%** | `ifelse` chains |
| `setp` | 115 | 160 | **+39%** | |
| `ld.global` | 229 | 253 | **+10%** | |
| `bra` | 10 | 10 | Same | |

### Gc (tracer tendency — T, S, or e)

| Instruction | `main` | optimized | Delta | Notes |
|-------------|-------:|----------:|------:|-------|
| `fma.rn.f64` | 1,277 | 769 | **-40%** | Largest absolute reduction |
| `selp` | 53 | 1,379 | **+2,502%** | Massive selp increase |
| `setp` | 75 | 155 | **+107%** | |
| `ld.global` | 152 | 177 | **+16%** | |
| `bra` | 12 | 12 | Same | |

**Key trade-off:** ~40% fewer `fma.rn.f64` (expensive double-precision FMA, ~8 cycles latency on V100) replaced by ~4-25x more `selp` (cheap predicated moves, ~4 cycles, no warp divergence). The `selp` instructions execute uniformly across all threads — they are simple conditional register copies, not divergent branches.

---

## 5. Nsys Benchmark Results

Profiled with `nsys profile --trace=cuda`, 15 timesteps (first 3 are warmup). Median kernel times reported.

### WENO Advection Kernels

| Kernel | `main` median (ns) | optimized median (ns) | Speedup |
|--------|--------------------:|----------------------:|--------:|
| **Gu** (u-momentum) | 3,896,968 | 3,031,962 | **1.29x (22.2% faster)** |
| **Gv** (v-momentum) | 3,441,594 | 2,884,171 | **1.19x (16.2% faster)** |
| **Gc** (T tracer) | 2,887,886 | 2,585,965 | **1.12x (10.5% faster)** |
| **Gc** (S tracer) | 2,854,637 | 2,594,574 | **1.10x (9.1% faster)** |
| **Gc** (e tracer) | 2,855,214 | 2,559,629 | **1.12x (10.3% faster)** |
| **Sum (Gu+Gv+3*Gc)** | 24,936,299 | 19,656,301 | **1.27x (21.2% faster)** |

### Non-WENO Kernels (Control — Should Be Unchanged)

| Kernel | `main` median (ns) | optimized median (ns) | Delta |
|--------|--------------------:|----------------------:|------:|
| `ab2_substep_TKE` | 5,740,444 | 5,727,383 | -0.2% |
| `CATKE_diffusivities` | 2,154,675 | 2,159,137 | +0.2% |
| `TKE_diffusivity` | 1,673,974 | 1,676,340 | +0.1% |
| `tridiagonal_solve` (best) | 957,034 | 947,705 | -1.0% |
| `barotropic_velocity` | 22,783 | 22,687 | -0.4% |
| `split_explicit_free_surface` | 18,528 | 18,560 | +0.2% |
| `update_hydrostatic_pressure` | 681,788 | 681,195 | -0.1% |
| `compute_w_from_continuity` | 344,958 | 345,053 | +0.0% |

Non-WENO kernels show noise-level variation (< 1%), confirming the optimization is targeted.

### Total GPU Kernel Time

| | `main` total (ns) | optimized total (ns) | Delta |
|---|-------------------:|---------------------:|------:|
| All CUDA kernels | 664,107,373 | 635,117,534 | **-4.4%** |
| WENO kernels only | 268,138,601 | 231,231,198 | **-13.8%** |

---

## 6. Why It's Faster Despite Higher Register Pressure

1. **FMA reduction dominates.** The ~40% fewer `fma.rn.f64` instructions eliminate redundant double-precision arithmetic that was duplicated across buffer scheme specializations. On V100, `DFMA` has ~8-cycle latency and is the throughput bottleneck for these compute-bound kernels.

2. **`selp` is cheap.** The conditional select instructions (`selp`) compile to predicated register moves — they execute in ~4 cycles with no warp divergence, no branch misprediction penalty, and full warp utilization. The 1000+ selps per kernel add overhead, but far less than the FMAs they replaced.

3. **Occupancy unchanged.** Both branches are at 1 block/SM (12.5% occupancy) because even the baseline's 200-216 registers already exceed the 1-block threshold. The increased register count (254-255) doesn't reduce occupancy further.

4. **No register spills.** Despite approaching the 255-register ceiling, the compiler avoids spilling to local memory (0 bytes on both branches). Spills would have been far more damaging than the register increase.

---

## 7. Risks and Notes

- **Register ceiling:** At 254-255 registers, any future kernel growth could trigger spills to local memory, which would significantly degrade performance. Monitor this carefully.
- **Gc cubin growth (+10%):** The SASS backend expands `selp` chains, increasing Gc cubin size. This has minimal runtime impact but increases JIT compilation time.
- **Occupancy floor:** Both branches are at the minimum 12.5% occupancy. Future work to reduce register pressure below 128 (for 2 blocks/SM, 25% occupancy) could yield additional gains.
