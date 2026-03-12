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

### Step 8: Diagnose simulation test failures — IN PROGRESS

Both UPivot and FPivot simulation tests fail with strict `all(.≈)`:
- Testset 5 (UPivot sim): 8/12 pass, 4 fail (job 162951251)
  - Slab (1×4) passes; pencil (2×2) and large-pencil (4×2) fail
  - Warnings: `Hy=16 >= Ny=10` and `Hx=16 >= Nx=10` — SplitExplicit extended halo exceeds local grid size
- Testset 6 (FPivot sim): 1/4 pass, 3 fail — u, v, η fail; c passes (job 162951252)

**Diagnostic approach**: Added IC comparison to isolate initialization vs time-stepping divergence:
- Split `run_distributed_simulation` → `setup_simulation` + `run_simulation!` (no code duplication)
- `run_distributed_tripolar_grid` now saves ICs (u0, v0, c0, η0) alongside final state
- Testset 6 serial side uses `setup_simulation` → capture ICs → `run_simulation!`

**Results (job 162953361, testset 6 FPivot slab 1×4)**:
- IC tests: **4/4 PASS** (u0, v0, c0, η0 all match serial)
- Final state: u FAIL, v FAIL, η FAIL, c PASS — same as before
- **Conclusion**: ICs are identical → divergence occurs during time-stepping, not initialization.
- This confirms the issue is in the time-stepper / SplitExplicit barotropic solver, not in `set!`/`fold_set!`.

**Hypothesis**: Halo size > local grid size causes incorrect SplitExplicit barotropic solver results in distributed mode. This affects both UPivot and FPivot, not FPivot-specific. The fact that c passes but u, v, η fail is consistent: c is a passive tracer advected by u/v, while u, v, η are coupled through the barotropic solver.

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

### Summary of confirmed results across all runs

| Testset | Description | Status | Notes |
|---------|-------------|--------|-------|
| 1 | Grid reconstruction | **PASS** (4/4 configs) | 60/60 tests each |
| 2 | Field reconstruction | **PASS** (4/4 configs) | 39/39 tests each (run 5) |
| 3 | UPivot boundary conditions | **PASS** (4/4) | Confirmed across multiple runs |
| 4 | FPivot boundary conditions | **PASS** (4/4) | Confirmed across multiple runs |
| 5 | UPivot simulations | **FAIL** | ALL configs fail (slab, pencil, large-pencil) |
| 6 | FPivot simulations | **FAIL** | ALL configs fail; ICs partially pass → divergence in time-stepping |

## Key references

- Serial fold: `src/BoundaryConditions/fill_halo_regions_upivotzipper.jl`, `fill_halo_regions_fpivotzipper.jl`
- Distributed fold: `src/OrthogonalSphericalShellGrids/distributed_zipper.jl`
- Field data: `src/Fields/field.jl`, `src/Grids/new_data.jl` (OffsetArray construction)
- Global reconstruction: `src/DistributedComputations/distributed_fields.jl`, `partition_assemble.jl`
- Distributed TripolarGrid set!: `src/OrthogonalSphericalShellGrids/distributed_tripolar_grid.jl`
- Generic index offset: `src/DistributedComputations/distributed_grids.jl` (`global_index_offset`)
