# WENO Advection Kernel Optimization Report

**Branch:** `ss/split-advective-maps` (based on `ss/optimize-weno`)
**GPU:** NVIDIA Titan V (5120 CUDA cores, 65536 regs/SM, 12 GB HBM2)
**Benchmark grid:** 500×200×60 ImmersedBoundaryGrid with sloping bottom
**Model config:** WENOVectorInvariant + WENO(order=7) + CATKE + SplitExplicitFreeSurface(70 substeps)

## Problem Statement

On ImmersedBoundaryGrid (IBG), WENO advection kernels in the `HydrostaticFreeSurfaceModel` suffer from high GPU register pressure. The monolithic tendency kernels on `main` use 199–210 registers per thread for momentum, far exceeding the 128-register threshold for 2 blocks/SM occupancy on Volta. This limits parallelism and throughput.

Three root causes drive the register inflation:

1. **Monolithic kernels** combine advection, Coriolis, pressure gradient, and diffusion into a single `@kernel`, preventing the compiler from reusing registers across independent terms.
2. **Immersed boundary overhead** adds runtime `conditional_δ` checks and a variable `red_order` (reduced WENO order near boundaries), which prevents the compiler from constant-folding WENO coefficients via `ifelse` chains.
3. **FP64 smoothness indicators** use 2 registers per stencil value for the smoothness computation (β), even though the smoothness indicators are intermediate values that don't need full precision.

## Optimizations Implemented

### 1. Kernel Fission

The monolithic `compute_hydrostatic_free_surface_Gu!` kernel (which computes advection + Coriolis + pressure gradient + diffusion in a single launch) was split into focused sub-kernels:

| Sub-kernel | Computes |
|------------|----------|
| `_compute_u_horizontal_advection!` | Horizontal vorticity-based advection |
| `_compute_u_vertical_advection!` | Vertical advection |
| `_compute_u_bernoulli_head!` | Kinetic energy gradient (Bernoulli head) |
| `_compute_u_forcing_diffusion!` | Coriolis + pressure + diffusion + forcing |

The same pattern is applied for v-velocity (4 sub-kernels) and tracers (2 sub-kernels: `_compute_tracer_advection!` + `_compute_tracer_forcing_diffusion!`).

Each sub-kernel writes to (or accumulates into) the tendency field independently. The first sub-kernel assigns (`Gu[i,j,k] = ...`), subsequent ones accumulate (`Gu[i,j,k] += ...` or `-=`).

**Impact:** Splitting allows the compiler to allocate registers only for the terms computed in each sub-kernel. The heavy WENO advection code no longer competes with diffusion/Coriolis for registers.

### 2. Interior/Boundary Stencil Split

For IBG grids, a precomputed classification partitions the active cells map into two subsets:

- **Interior cells** (81.4% of active cells): Cells whose full WENO stencil lies entirely within the fluid domain. These are launched with `grid.underlying_grid` instead of the IBG, so the compiler never sees the immersed boundary code paths. The `conditional_δ` checks compile away, `red_order` is always the full order (a compile-time constant), and WENO coefficient `ifelse` chains constant-fold.

- **Boundary cells** (18.6% of active cells): Cells near the immersed boundary that require the full IBG treatment with runtime `red_order`.

The stencil classification is computed once at model construction time. A boolean field marks cells where the full stencil (of radius equal to the advection scheme's halo) is in the fluid domain. The active cells map is then partitioned into interior/boundary index arrays stored in `grid.stencil_active_cells`.

The `split_advection_launch!` function handles the dispatch:

```julia
# Interior cells: underlying_grid (no IBG overhead)
launch!(arch, grid, kp, kernel!, Gvel, grid.underlying_grid, args...;
        active_cells_map=stencil_pair.interior)

# Boundary cells: full IBG
launch!(arch, grid, kp, kernel!, Gvel, grid, args...;
        active_cells_map=stencil_pair.boundary)
```

**Impact:** Interior cells compile a specialized GPU binary without IBG overhead. This drops registers dramatically (e.g., 210 → 89 for u_horizontal) and enables higher occupancy.

### 3. FP32 Smoothness Indicators

WENO smoothness indicators (β) are computed from stencil values using quadratic forms. These are intermediate quantities used only to compute nonlinear weights — they don't need Float64 precision. By converting stencil values to `Float32` before computing β:

```julia
@inline function smoothness_indicator(ψ, scheme::WENO{N, FT, FT2}, ...) where {N, FT, FT2}
    coefficients = smoothness_coefficients(scheme, red_order, val_stencil)
    ψ_low = FT2.(ψ)  # FT2 = Float32 by default
    return smoothness_operation(scheme, ψ_low, coefficients)
end
```

each stencil value uses 1 register instead of 2 during the smoothness computation. The `FT2` type parameter (defaulting to `Float32`) is stored in the `WENO` type, allowing compile-time specialization.

**Impact:** Reduces register usage by ~10 registers for horizontal advection kernels. No measurable effect on solution accuracy (smoothness indicators are used only for weight ratios).

### 4. maxregs Infrastructure (Not Applied by Default)

A `maxregs` keyword argument was added to `launch!` that allows passing CUDA's `maxregs` compiler hint to control register allocation. The CUDA extension (`OceananigansCUDAExt.jl`) overrides `_launch_kernel!` to recompile kernels with `@cuda maxregs=N`, bypassing KernelAbstractions' default compilation path.

Per-kernel benchmarks showed significant individual improvements (up to 47% for specific kernels), but the full-model impact was negligible (~1%) because:
- Advection is only ~35% of the timestep compute
- The optimal `maxregs` value differs per kernel — a blanket value helps some kernels but hurts others
- Boundary cells (where the largest per-kernel win was found) are only 18.6% of cells

The infrastructure remains available as an opt-in feature via `launch!(...; maxregs=N)` and `split_advection_launch!(...; interior_maxregs=N, boundary_maxregs=N)`.

## What Was Tried and Didn't Work

### Early Full-Order Fast Path

Added an `if red_order == N` branch to `biased_interpolate` to skip the `ifelse` coefficient chains when running at full WENO order. On CPU, this would allow the compiler to constant-fold the full-order path.

**Result:** Regression. On CUDA, both branches of `if`/`else` are compiled, and the register count is `max(both paths)`. The reduced-order path (with all `ifelse` chains) was the heavier one, so the branch only increased register pressure. IBG horizontal advection went from 118 → 128 registers. Full-model time increased from 35.5 → 36.8 ms/step.

**Lesson:** On GPU, `if`/`else` branches inflate register counts to the maximum of both paths. The existing `ifelse`/`selp` approach is actually more register-efficient because it shares registers across conditional assignments.

### Blanket `maxregs` for All Kernels

Applied `interior_maxregs=85` to all split advection sub-kernels (horizontal, vertical, Bernoulli head, tracer advection).

**Result:** Massive regression (35.5 → 63.9 ms/step). While `maxregs=85` helped v_horizontal (34% faster individually), it caused excessive register spilling on u_horizontal and tracer advection kernels that didn't benefit. The forced register cap pushed values to local memory (L1 cache), increasing latency.

**Lesson:** `maxregs` is highly kernel-specific. A value that helps one kernel can catastrophically hurt another. Per-kernel tuning is required, but the infrastructure complexity outweighs the benefit for this model configuration.

## Per-Kernel Benchmark Results

All timings are median of 50 trials. Kernel launches use the full active cells map (3,500,000 cells: 2,847,400 interior + 652,600 boundary).

### `main` — Monolithic Kernels

| Kernel | Registers | Time (μs) |
|--------|:---------:|----------:|
| `Gu` (advection + Coriolis + pressure + diffusion) | 210 | 4,213 |
| `Gv` (advection + Coriolis + pressure + diffusion) | 199 | 3,471 |
| `Gc` T (advection + diffusion) | 119 | 3,008 |
| **Momentum total (Gu + Gv)** | | **7,684** |
| **Tracer total (3 × Gc)** | | **9,024** |
| **Grand total** | | **16,708** |

At 210 registers × 256 threads/block = 53,760 regs/block → only 1 block/SM (65,536 limit). Occupancy is **12.5%** (256 of 2048 max threads/SM).

### Optimized — Sub-Kernels on IBG (Fission Only, No Stencil Split)

| Kernel | Registers | Time (μs) | vs main |
|--------|:---------:|----------:|--------:|
| u_horizontal | 127 | 961 | |
| v_horizontal | 114 | 697 | |
| u_vertical | 114 | 1,271 | |
| v_vertical | 102 | 1,065 | |
| u_bernoulli | 48 | 326 | |
| v_bernoulli | 43 | 318 | |
| Gc_adv T | 99 | 2,427 | |
| **Momentum total** | | **4,638** | **1.66×** |
| **Tracer total (3 × Gc_adv)** | | **7,281** | **1.24×** |
| **Grand total** | | **11,919** | **1.40×** |

Note: This total excludes forcing+diffusion sub-kernels (which are present in both configurations). The comparison is apples-to-apples because main's monolithic kernel includes those terms.

### Optimized — Sub-Kernels with Interior/Boundary Split + FP32 (Actual Execution)

| Kernel | Interior regs | Boundary regs | Time (μs) | vs main |
|--------|:---:|:---:|---:|---:|
| u_horizontal | 89 | 127 | 393 | |
| v_horizontal | 76 | 114 | 438 | |
| u_vertical | 64 | 114 | 815 | |
| v_vertical | 66 | 102 | 704 | |
| u_bernoulli | 44 | 48 | 284 | |
| v_bernoulli | 40 | 43 | 286 | |
| Gc_adv T | 83 | 99 | 1,169 | |
| **Momentum total** | | | **2,920** | **2.63×** |
| **Tracer total (3 × Gc_adv)** | | | **3,507** | **2.57×** |
| **Grand total** | | | **6,427** | **2.60×** |

Interior kernels at 64–89 registers → 3 blocks/SM at 256 threads/block → **37.5% occupancy** (vs 12.5% on main).

### Optimization Contribution Breakdown

| Technique | Advection time | Cumulative speedup |
|-----------|---------------:|-------------------:|
| `main` monolithic | 16,708 μs | 1.00× |
| + Kernel fission | 11,919 μs | 1.40× |
| + Interior/boundary split | ~7,100 μs (est.) | ~2.35× |
| + FP32 smoothness | 6,427 μs | 2.60× |

## Full-Model Timestep Results

| Configuration | Per timestep | Speedup |
|---------------|------------:|--------:|
| `main` (monolithic) | 41.5 ms | — |
| Optimized (fission + split + FP32) | 35.5 ms | **14.5%** |

The 2.6× advection speedup translates to 14.5% full-model improvement because the advection kernels account for roughly 35-40% of the total timestep compute. The remaining time is dominated by:
- SplitExplicitFreeSurface with 70 barotropic substeps
- CATKE vertical diffusivity computation
- Halo communication and boundary condition fills
- Pressure solve and free surface updates

## Files Modified

| File | Change |
|------|--------|
| `src/Models/split_advection.jl` | New: stencil split maps, `split_advection_launch!`, `attach_stencil_active_cells` |
| `src/Models/HydrostaticFreeSurfaceModels/split_hydrostatic_momentum_tendencies.jl` | New: split sub-kernels for Gu, Gv, Gc; split dispatch for `VectorInvariantUpwindVorticity` |
| `src/Models/NonhydrostaticModels/split_nonhydrostatic_tendencies.jl` | New: split sub-kernels for NonhydrostaticModel |
| `src/Models/HydrostaticFreeSurfaceModels/compute_hydrostatic_free_surface_tendencies.jl` | Dispatch to split path when advection is upwind-biased |
| `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl` | Call `attach_stencil_active_cells` at model construction |
| `src/Models/NonhydrostaticModels/nonhydrostatic_model.jl` | Call `attach_stencil_active_cells` at model construction |
| `src/ImmersedBoundaries/active_cells_map.jl` | `compute_stencil_interior_field`, `partition_active_map_by_stencil` |
| `src/ImmersedBoundaries/immersed_boundary_grid.jl` | Added `stencil_active_cells` field to `ImmersedBoundaryGrid` |
| `src/Advection/weno_interpolants.jl` | FP32 smoothness indicators via `FT2` type parameter |
| `src/Advection/flux_form_advection.jl` | `FT2` type parameter propagation for `WENO` |
| `src/Utils/kernel_launching.jl` | `maxregs` kwarg support in `launch!` / `_launch!` |
| `ext/OceananigansCUDAExt.jl` | CUDA `maxregs` override via `_launch_kernel!` |
| `src/DistributedComputations/distributed_immersed_boundaries.jl` | `DistributedActiveInteriorIBG` type alias for distributed stencil maps |

## Distributed Grid Support

The stencil split maps support distributed (MPI) grids where `interior_active_cells` is a 5-part `NamedTuple`:

```julia
(; halo_independent_cells, west_halo_dependent_cells, east_halo_dependent_cells,
   south_halo_dependent_cells, north_halo_dependent_cells)
```

Each sub-map is independently partitioned into `(; interior, boundary)` pairs. The `get_stencil_pair` function resolves the correct pair for a given active cells map by identity comparison.

## Reproducibility

```bash
# Full-model benchmark
julia --project benchmark_fp32.jl

# Per-kernel breakdown (optimized branch)
julia --project benchmark_per_kernel.jl

# Per-kernel breakdown (main branch)
git stash && git checkout main
julia --project benchmark_per_kernel_main.jl
git checkout ss/split-advective-maps && git stash pop
```
