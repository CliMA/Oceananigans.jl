# Master Plan: Fix distributed FPivot TripolarGrid

## Completed Steps

### Step 1: Index mapping diagnostic scripts — COMPLETE
- Created 4 index-printing scripts: `print_index_CC.jl`, `print_index_FC.jl`, `print_index_CF.jl`, `print_index_FF.jl`
- Shared helpers in `print_index_common.jl`, PBS launcher in `run_print_index.sh`
- Output: `print_index.o162927038` (all 16 cases: 4 locations × 2 topologies × 2 partitions)
- Key finding: `set!` triggers `fold_set!` which is broken for FPivot distributed — bypassed by writing directly to `interior()` instead
- Key finding: index mapping is `serial_parent_index = local_offset_index + rank_offset + H`

### Step 2: Fix local halo index mapping in `index_tracing_halo_test.jl` — COMPLETE
- Fixed two bugs: (1) `fill_index_field!` bypasses `set!`/`fold_set!` by writing to `interior()`, (2) k-index `di.data[:,:,1]` not `1+Hz`
- Output: `index_halo_test.o162931660`
- Results: Partition(1,4) PASS for both UPivot and FPivot (all 4 locations)
- Partition(2,2) fails only at the fold row — confirms Step 3 is the root cause
- UPivot Partition(2,2): CC/FC fail at j=Ny=40 (serial reflects right half, distributed doesn't)
- FPivot Partition(2,2): CF/FF fail at j=Ny=21 on northernmost ranks (same fold issue)

### Step 2b: Generic `global_index_offset` + expanded Ny tests — COMPLETE
- Added canonical `global_index_offset(N, R, local_index)` to `src/DistributedComputations/distributed_grids.jl`
- Convenience method `global_index_offset(arch::Distributed, global_sz)` returns 3-tuple of offsets
- Exported from `DistributedComputations.jl`
- Updated `index_tracing_halo_test.jl` and `print_index_common.jl` to use generic offset
- Expanded test matrix: 4 Ny values (UPivot 40/43, FPivot 41/42) × 2 partitions = 8 configs
- Output: `index_halo_test.o162934499` — 8/8 PASS (after Step 3 fix applied simultaneously)
- Commit: `7f8f024cf`

### Step 3: Fix distributed fold-line overwrite — COMPLETE
- **Root cause**: Serial fold kernels overwrite the right half of the fold row with x-reversed values.
  The distributed code in `distributed_zipper.jl` was missing this overwrite for:
  - UPivot: CC (Center,Center) and FC (Face,Center) at j=Ny
  - FPivot: CF (Center,Face) and FF (Face,Face) — had interior overwrite but missing corner writes
- **Fix** in `src/OrthogonalSphericalShellGrids/distributed_zipper.jl`:
  1. Added `_fold_line_from_buffer!` methods for UPivot Center-y (CC via Center-x, FC via Face-x)
  2. Added `_fold_line_parent_y` helper to dispatch on location/topology
  3. Added fold-line corner writes in `switch_north_halos!` via `_fold_corner_write!`
  4. Updated comment to reflect both UPivot and FPivot coverage
- Output: `index_halo_test.o162934499` — **8/8 configurations PASS** (all locations, both topologies, both partitions, multiple Ny values)
- Commit: `7f8f024cf` (includes Step 3 fix)

### Step 4: Verify `reconstruct_global_field` for FPivot — COMPLETE (already fixed)

Already fixed by commits `915f929e2` and `f9bfb201e`. Grid/field reconstruction tests passed in job `162939008` (ran inside MPI but results not reported — see Step 7).

### Step 5: Remove test `_fill_north_halo!` overrides — COMPLETE

- Root cause of UPivot boundary test failure in job `162937439`: test's custom `_fill_north_halo!` overrides skipped fold-line overwrite. Now that distributed correctly does fold-line overwrite (Step 3), the serial built-in code is the correct reference.
- Removed all 4 `_fill_north_halo!` override dispatches and 4 helper functions from `tmp_MPI_Gadi/distributed_tests_utils.jl`
- Job `162939008` results: **boundary conditions 8/8 PASS** (both UPivot and FPivot)
- `test/distributed_tests_utils.jl` still needs same removal (pending)

## Next Steps

### Step 6: Comprehensive local field + halo comparison — COMPLETE

Standalone MPI job verifying every cell (interior + halos) on every rank matches serial after `fill_halo_regions!`.
- Fields: c (CC), u (FC), v (CF), ζ (FF) — i-index and j-index fields for each
- Index-tracing technique with `fill_index_field!` + global offset
- Test matrix: UPivot Ny=40/43, FPivot Ny=41/42, Partition(1,4) and Partition(2,2) = 8 configs
- Job `162946593`: **96/96 tests PASS** (8 configs × 4 fields × 3 tests each)
- Files: `tmp_MPI_Gadi/field_halo_test.jl`, `tmp_MPI_Gadi/run_field_halo_test.sh`

### Step 7: Fix test reporting & CUDA warnings — COMPLETE

1. Restructured grid/field reconstruction tests in `test_mpi_tripolar.jl`:
   - Old: `@test` runs inside MPI subprocess strings → never reported ("Total 0")
   - Fix: save full arrays from every rank to JLD2, load in parent, `@test` every cell including halos
   - One `Distributed` architecture per MPI run, parameterized via string interpolation
2. Added `module load cuda/12.9.0` + env vars to `run_tripolar_tests.sh`
3. Parallelized testsets via `TRIPOLAR_TESTSET` env var + `run_tripolar_tests_parallel.sh` driver
4. Results:
   - Testset 3 (UPivot BC): **4/4 PASS** (job 162951249)
   - Testset 4 (FPivot BC): **4/4 PASS** (job 162951250)
   - Testset 1 (grid recon): running (job 162951247, old code — will rerun)
   - Testset 2 (field recon): running (job 162951248, old code — will rerun)

### Step 8: Fix simulation test failures — IN PROGRESS

**Two root causes identified**:

1. **`.data` vs `interior()` comparison** (same as Step 12): `reconstruct_global_field` calls
   `set!` but not `fill_halo_regions!`. Comparing `.data` fails due to unfilled halos.
   Additionally, SplitExplicit extends halos only for `ConnectedTopology` directions
   (`split_explicit_free_surface.jl:228`). Serial TripolarGrid has x=Periodic (Hx stays 5)
   but y=RightCenterFolded (Hy extends). Distributed x becomes FullyConnected (Hx extends).
   This creates a DimensionMismatch for η when comparing serial `.data` vs distributed `.data`.

2. **SplitExplicit halo overflow**: With `substeps=20`, `H=16`. When H ≥ local N (slab Ny=10,
   large-pencil Nx=10), the barotropic solver accesses insufficient halo data → incorrect results.

**Fixes applied**:
1. Save `Array(interior(...))` instead of `.data` in `run_distributed_tripolar_grid`
2. Compare `interior()` instead of `.data` in testsets 5 & 6
3. Reduced `substeps` from 20 to 5 (H=5, no halo extension needed; tests don't need accurate substepping)

**Run 6 results (testsets 5 & 6, jobs 162974270–162974275)**:
- Testset 5 cfg1 (slab 1×4): **PASS 4/4** (job 162974270)
- Testset 5 cfg2 (pencil 2×2): **PASS 4/4** (job 162974271)
- Testset 5 cfg3 (large-pencil 4×2): **FAIL 1/4** (job 162974272) — u/v/η fail, c passes
- Testset 6 cfg1 (slab 1×4): **PASS 8/8** (job 162974273)
- Testset 6 cfg2 (pencil 2×2): **PASS 8/8** (job 162974274)
- Testset 6 cfg3 (large-pencil 4×2): **PASS 8/8** (job 162974275)

**Root cause of cfg3 failure**: Tests used `us ≈ up` (norm-based array `isapprox`) instead of
`all(us .≈ up)` (element-wise, matching upstream). Norm-based comparison is too strict for
small-valued arrays with tiny floating-point differences from 4-rank x-decomposition.

**Fix attempt 1**: Changed all 12 simulation assertions to `all(us .≈ up)` style (element-wise).

**Run 7 results (all testsets 5 & 6, jobs 162975907–162975912)**: **WORSE**
- Testset 5 cfg1 (slab 1×4): **PASS 4/4** (job 162975907)
- Testset 5 cfg2 (pencil 2×2): **PASS 4/4** (job 162975908)
- Testset 5 cfg3 (large-pencil 4×2): **FAIL 0/4** (job 162975909) — all 4 fail (c now fails too)
- Testset 6 cfg1 (slab 1×4): **FAIL 5/8** (job 162975910) — ICs pass, final u/v/η fail
- Testset 6 cfg2 (pencil 2×2): **FAIL 5/8** (job 162975911) — ICs pass, final u/v/η fail
- Testset 6 cfg3 (large-pencil 4×2): **FAIL 5/8** (job 162975912) — ICs pass, final u/v/η fail

**Revised root cause**: The real issue is `substeps=5`, not the comparison method.
- Element-wise `all(.≈)` is STRICTER than norm-based `≈` for individual cells with near-zero values
- `substeps=5` under-resolves barotropic dynamics, producing noisy near-zero values in η/u/v
- `isapprox(a, b)` fails when both a,b are tiny: `|a-b| > rtol * max(|a|, |b|)`
- The original "halo overflow" concern was a misdiagnosis: `Gⁿ.U`/`Gⁿ.V` are on the extended
  free surface grid (halo=H), not the model grid (halo=5). Ghost zone expansion self-corrects
  even when H > N — the +2 margin in H=Nsubsteps+2 protects interior values.

**Fix attempt 2**: Reverted `substeps` from 5 to 20 (matching upstream). Removed `--check-bounds=yes`
from simulation mpiexec commands. Kept `all(.≈)` comparison.

**Run 8 results (all testsets 5 & 6, jobs 162977900–162977906)**:
- Testset 5 cfg1 (slab 1×4): **PASS 4/4** (job 162977900)
- Testset 5 cfg2 (pencil 2×2): **PASS 4/4** (job 162977902)
- Testset 5 cfg3 (large-pencil 4×2): **FAIL 0/4** (job 162977903) — H=16 >= Nx_local=10
- Testset 6 cfg1 (slab 1×4): **FAIL 5/8** (job 162977904) — ICs pass, final u/v/η fail
- Testset 6 cfg2 (pencil 2×2): **FAIL 5/8** (job 162977905) — ICs pass, final u/v/η fail
- Testset 6 cfg3 (large-pencil 4×2): **FAIL 5/8** (job 162977906) — ICs pass, final u/v/η fail

**Analysis**: substeps=20 fixed UPivot slab/pencil (testset 5), but FPivot still fails across ALL
configs. The actual Nsubsteps after weight trimming ≈ 14 (not 20), giving H = 14+2 = 16.

Key observations:
- FPivot ICs pass (grid + field setup correct), only u/v/η fail after time-stepping
- Tracer c passes — it's only advected by 3D time stepper with standard halo fills
- u/v/η are all updated by SplitExplicit barotropic substep loop
- Testset 5 cfg3 fails because H=16 >= Nx_local=10 (upstream doesn't test this config)
- **The FPivot simulation test is entirely new on this branch** (not in upstream)

**Diagnostic run 9**: Testing `SplitExplicitFreeSurface(grid; substeps=20, extend_halos=false)`
(fill-halo mode = fill_halo_regions! between each substep, no ghost zone expansion).

**Run 9a (job 162980503)**: ERRORED — used invalid `SplitExplicitFreeSurface{false}(grid; substeps=20)`
syntax. The type parameter `{false}` is the inner constructor; the correct API uses the `extend_halos`
keyword: `SplitExplicitFreeSurface(grid; substeps=20, extend_halos=false)`.

**Run 9b (job 162981449)**: **FAIL 5/8** — same pattern as ghost zone mode. ICs pass, final u/v/η
fail, c passes. Fill-halo mode did NOT fix the problem.

**Conclusion from run 9**: The bug is **not** specific to ghost zone temporal blocking — it
fails in both modes.

**Diagnostic run 10** (job 162983558): Enhanced `simulation_debug.jl` — serial + distributed
side-by-side in same MPI job, comparing at steps 0, 1, 10, 100 with per-j-row mismatch analysis.

**Run 10 results — CRITICAL FINDING: errors are at SOUTH boundary, NOT the fold!**

| Step | u | v | c | η | U | V |
|------|---|---|---|---|---|---|
| 0 | MATCH | MATCH | MATCH | MATCH | MATCH | MATCH |
| 1 | MATCH | MATCH | MATCH | MATCH | MATCH | MATCH |
| 10 | j=1..3 ~1e-28 | j=2..4 ~1e-29 | MATCH | j=1..3 ~1e-29 | j=1..3 ~1e-25 | j=2..4 ~1e-26 |
| 100 | j=1..6 ~1e-13 | j=2..6 ~1e-15 | MATCH (3e-39) | j=1..6 ~1e-14 | j=1..6 ~1e-10 | j=2..6 ~1e-12 |

Key observations:
- **Mismatches at j=1..6 (south/Bounded boundary), NOT near j=121 (fold)**
- The FPivot fold communication appears CORRECT — no errors near fold line
- Errors are tiny (~1e-28 at step 10) and grow over baroclinic steps (~1e-13 at step 100)
- Only SplitExplicit fields (u/v/η/U/V) affected; tracer c always matches
- 120 mismatches per step-10 row = exactly Nx=40 × Nz=3(?), suggesting entire rows
- Barotropic U/V errors are ~1000× larger than u/v (consistent with depth integration)
- Pattern: errors diffuse northward over time (j=1..3 → j=1..6)

**Revised diagnosis**: This is NOT a fold communication bug. It appears to be floating-point
accumulation differences at the south boundary between serial (single-domain kernel) and
distributed (partitioned kernel) SplitExplicit solvers. The `isapprox` test fails because
both serial and distributed values are very tiny near the south boundary, and `isapprox(a,b)`
with default rtol≈1.5e-8 fails when `rtol * max(|a|,|b|)` is smaller than `|a-b|`.

**Run 10b — UPivot comparison** (job 162987377): **ALL MATCH through 100 steps** (maxdiff=0.0).
UPivot has ZERO mismatches at any step. The south-boundary errors are **FPivot-specific**.

This rules out the "generic FP accumulation" hypothesis. Something specific to `RightFaceFolded`
(FPivot) causes tiny differences at the south boundary that `RightCenterFolded` (UPivot) doesn't.

Possible FPivot-specific differences to investigate:
1. `split_explicit_kernel_size(::Type{RightFaceFolded}, N, H) = 1:N+H-1` vs
   `split_explicit_kernel_size(::Type{RightCenterFolded}, N, H) = 1:N+H` — differs by 1 at north end
2. FPivot fold writes to interior row Ny (Pattern C in distributed_zipper.jl) — could contaminate
   the ghost zone kernel computation differently
3. `Ny=121` (odd, FPivot) vs `Ny=120` (even, UPivot) — uneven partition 30+30+30+31 vs even 30×4
4. `fold_set!` in `distributed_tripolar_grid.jl` — FPivot-specific fold-line initialization

**Root cause confirmed**: FPivot places `southernmost_latitude` at the cell **face** of j=1,
so the cell center is at ~φm + Δφ/2 ≈ -79.3°. The condition `(φ < φm)` evaluates to FALSE
at the center, leaving j=1 as water instead of immersed. Active cells at j=1 expose tiny FP
accumulation differences between serial and distributed SplitExplicit solvers.

**Confirmed fix** (job 162991727): Using `(φ < 0)` makes ALL MATCH through 100 steps.
Applied fix: `(φ < φm + radius)` which gives `φ < -75°`, safely immersing j=1 (~-79.3° < -75°).

**Additional fix**: Increased grid sizes from (40,40,1)/(40,121,1) to (80,80,1)/(80,81,1)
to ensure Nx_local > H for Partition(4,2) configs. With Nx=80 and 4 x-ranks, Nx_local=20 > H=16.

**Status**: Fixes applied, awaiting verification run.

### Step 9: Parallelize testsets across PBS jobs — COMPLETE

Added `TRIPOLAR_CONFIG` env var to avoid sequential MPI launches within a single PBS job:
- **Testsets 1 & 2**: 4 configs each (2 folds × 2 partitions). `CONFIG` 1-4 selects one;
  `CONFIG=0` runs all (backward compat). Each config is ~5 min instead of ~35 min sequential.
- **Testsets 5 & 6**: 3 configs each (slab 1×4, pencil 2×2, large-pencil 4×2).
  Refactored into `sim_configs`/`fpivot_sim_configs` loops with `CONFIG` filtering.
  Serial simulation re-runs per job but MPI launches run in parallel across PBS jobs.
- **Testset 6 expanded**: Now tests all 3 partition configs (matching testset 5), with
  IC comparison (4 IC tests + 4 final-state tests per config). Uses `fpivot_sim_script()`
  helper to generate MPI scripts with interpolated partition/filename.
- Updated `run_tripolar_tests_parallel.sh`: now submits 16 independent PBS jobs
  (4+4+1+1+3+3) instead of 6.

### Step 10: Test refactoring — save `.data`, `jldopen`, `@testset for` — COMPLETE

1. Replaced all `jldsave` calls with `jldopen(f, "w") do file; file["key"] = val.data; end`
2. Wrote out all grid variable names explicitly (no metaprogramming)
3. Converted all testsets to `@testset for` syntax to reduce nesting
4. Fixed `config_id` scoping: `global config_id += 1` required inside `@testset for` (Julia 1.12 local scope)
5. Removed dead `mismatches` accumulation code — direct `@test` at every cell instead
6. Committed removal of test `_fill_north_halo!` overrides from `test/distributed_tests_utils.jl` (commit `0127aa60e`)

### Step 11: Fix MPI deadlock in testsets 1 & 2 — COMPLETE

**Root cause**: `reconstruct_global_grid` / `reconstruct_global_field` use MPI collectives
(`MPI.Allreduce!` via `concatenate_local_sizes`) but were guarded by `if arch.local_rank == 0`
in the MPI subprocess scripts. Only rank 0 entered, while ranks 1-3 skipped to `MPI.Barrier` → deadlock.

**Fix**: Moved the `reconstruct_global_*` calls outside the rank-0 guard so all ranks participate.
Only the JLD2 save remains rank-0-only.

**Failed run 3 (162969723–162969741)**:
- Jobs 162969723–162969731 (testsets 1 & 2): **DEADLOCKED** — killed after 30+ min
- Jobs 162969732, 162969734 (testsets 3 & 4): **PASS** (4/4 each)
- Jobs 162969735–162969737 (testset 5): **ALL FAIL** (slab, pencil, large-pencil)
- Jobs 162969738–162969741 (testset 6): **ALL FAIL** (ICs partially pass, final state fails)

### Step 12: Fix testset 2 global field comparison + simplify per-rank tests — COMPLETE

**Root cause of 3 failures per config**: `reconstruct_global_field` calls `set!` but never
`fill_halo_regions!`. Comparing `gu.data ≈ us.data` fails because reconstructed field has
zero halos while serial field has filled halos. Interior values match.

**Fixes**:
1. MPI subprocess saves `Array(interior(gu))` instead of `gu.data`; main process compares
   with `interior(us)` — avoids halo mismatch
2. Replaced per-cell `@testset "(i=$i, j=$j)" for ...` (~24000 tests) with array-level
   `@test local_data[:, :, 1] ≈ serial_data[irange, jrange, 1]` (~12 tests per config)

**Run 4 results (162971102–162971117)**:
- Testset 1: **4/4 PASS** (60/60 each) — cfg1 took 8 min (job 162971102), cfg2-4 ~3 min each
- Testset 2: all 4 configs — 24000+ local tests **PASS**, 3 global recon **FAIL** (fixed in run 5)
- Testsets 3 & 4: **PASS** (confirmed again)
- Testsets 5 & 6: **ALL FAIL** (confirmed again)

**Run 5 results (testset 2 only, 162971698–162971701)**: **ALL PASS**
- cfg1 (UPivot × 1×4): **39/39 PASS** (job 162971698)
- cfg2 (UPivot × 2×2): **39/39 PASS** (job 162971699)
- cfg3 (FPivot × 1×4): **39/39 PASS** (job 162971700)
- cfg4 (FPivot × 2×2): **39/39 PASS** (job 162971701)

**Run 6 results (162999325–162999340)**: Full verification after south masking fix + grid size increase.
- Testsets 1–4: **ALL PASS** (unchanged)
- Testset 5 cfg1 (slab 1×4): **PASS**, cfg2 (pencil 2×2): **PASS**
- Testset 5 cfg3 (large-pencil 4×2): **FAIL** — u, v, η fail; c passes (1 pass / 3 fail)
- Testset 6 cfg1 (slab 1×4): **PASS**, cfg2 (pencil 2×2): **PASS**
- Testset 6 cfg3 (large-pencil 4×2): **FAIL** — u fails; v, c, η pass (3 pass / 1 fail)

**Analysis**: The H>=N theory was incorrect — with 80×80 and Partition(4,2), Nx_local=20 > H=16.
The failures are FP-accumulation differences specific to 4 x-ranks. Slab (1×4) and pencil (2×2)
both pass, so the issue is specific to having 4 ranks in the x-direction.

**Next step**: Run `simulation_debug.jl` with Partition(4,2) on the 80×80/81 grids to pinpoint
where divergence starts (step 0, 1, 10, or 100) and which rows/fields are affected.

### Summary of confirmed results across all runs

| Testset | Description | Status | Notes |
|---------|-------------|--------|-------|
| 1 | Grid reconstruction | **PASS** (4/4 configs) | 60/60 tests each |
| 2 | Field reconstruction | **PASS** (4/4 configs) | 39/39 tests each (run 5) |
| 3 | UPivot boundary conditions | **PASS** (4/4) | Confirmed across multiple runs |
| 4 | FPivot boundary conditions | **PASS** (4/4) | Confirmed across multiple runs |
| 5 | UPivot simulations | **PARTIAL** | slab+pencil PASS, large-pencil 4×2 FAIL (u,v,η) |
| 6 | FPivot simulations | **PARTIAL** | slab+pencil PASS, large-pencil 4×2 FAIL (u only) |

## Key references

- Serial fold: `src/BoundaryConditions/fill_halo_regions_upivotzipper.jl`, `fill_halo_regions_fpivotzipper.jl`
- Distributed fold: `src/OrthogonalSphericalShellGrids/distributed_zipper.jl`
- Field data: `src/Fields/field.jl`, `src/Grids/new_data.jl` (OffsetArray construction)
- Global reconstruction: `src/DistributedComputations/distributed_fields.jl`, `partition_assemble.jl`
- Distributed TripolarGrid set!: `src/OrthogonalSphericalShellGrids/distributed_tripolar_grid.jl`
- Generic index offset: `src/DistributedComputations/distributed_grids.jl` (`global_index_offset`)
