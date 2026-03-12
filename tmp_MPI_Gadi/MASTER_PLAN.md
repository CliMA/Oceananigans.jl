# Master Plan: Fix distributed FPivot TripolarGrid

## Completed Steps

### Step 1: Index mapping diagnostic scripts — COMPLETE
- Created 4 index-printing scripts: `print_index_CC.jl`, `print_index_FC.jl`, `print_index_CF.jl`, `print_index_FF.jl`
- Shared helpers in `print_index_common.jl`, PBS launcher in `run_print_index.sh`
- Output: `print_index.o162927038` (all 16 cases: 4 locations × 2 topologies × 2 partitions)
- Key finding: `set!` triggers `fold_set!` which is broken for FPivot distributed — bypassed by writing directly to `interior()` instead
- Key finding: index mapping is `serial_parent_index = local_offset_index + rank_offset + H`

### Step 2: Fix local halo index mapping in `index_tracing_halo_test.jl`
- Status: IN PROGRESS

## Deferred Steps

### Step 3: Investigate fold at row Ny

The serial `fill_halo_regions!` for UPivot CC does:
```julia
c[i, Ny, k] = ifelse(i > Nx ÷ 2, sign * c[i′, Ny, k], c[i, Ny, k])
```
This overwrites the right half of row Ny. The distributed case should do the same.

- Check `src/OrthogonalSphericalShellGrids/distributed_zipper.jl`
- If the distributed fold doesn't apply the same interior overwrite, fix it
- Files: `src/OrthogonalSphericalShellGrids/distributed_zipper.jl`

### Step 4: Fix `reconstruct_global_field` for FPivot

FPivot Partition(1,4) shows zeros from row 11+ in the reconstructed global field.
- `construct_global_array` may have size mismatch for FPivot Center-y fields
- Center-y `size(field)` may return different values on different ranks
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
