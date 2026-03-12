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

## Next Steps

### Step 4: Fix `reconstruct_global_field` for FPivot (may be resolved)

Previous issue: FPivot Partition(1,4) showed zeros from row 11+. But job 162931660 shows FPivot Partition(1,4) PASSES interior check. Possibly already fixed by commit 915f929e2.
- Recheck if still needed after running full test suite.
- Files: `src/DistributedComputations/partition_assemble.jl`, `distributed_fields.jl`

### Step 5: Fix `set!` crash for FPivot Partition(2,2)

Crash in `fold_set!` during `set!` on distributed FPivot field:
```
fold_set! at distributed_tripolar_grid.jl:71
fold_set! at distributed_tripolar_grid.jl:49
set! at distributed_tripolar_grid.jl:41
```
- Files: `src/OrthogonalSphericalShellGrids/distributed_tripolar_grid.jl`

### Step 6: Fix FPivot simulation test tolerance

PBS job 162913183: FPivot simulation 3/4 fail (u, v, η) with `all(.≈)` too strict for ~1e-13 diffs after 100 time steps. May resolve after fixing steps 3-5.

## Key references

- Serial fold: `src/BoundaryConditions/fill_halo_regions_upivotzipper.jl`, `fill_halo_regions_fpivotzipper.jl`
- Distributed fold: `src/OrthogonalSphericalShellGrids/distributed_zipper.jl`
- Field data: `src/Fields/field.jl`, `src/Grids/new_data.jl` (OffsetArray construction)
- Global reconstruction: `src/DistributedComputations/distributed_fields.jl`, `partition_assemble.jl`
- Distributed TripolarGrid set!: `src/OrthogonalSphericalShellGrids/distributed_tripolar_grid.jl`
- Generic index offset: `src/DistributedComputations/distributed_grids.jl` (`global_index_offset`)
