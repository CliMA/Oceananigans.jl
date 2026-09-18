---
name: run-tests
description: Run targeted Oceananigans tests, prioritized by what's likely to break
user_invocable: true
---

# Run Tests

Run targeted tests one-by-one, prioritized by what's most likely to fail given recent changes.
Never run the full test suite blindly — it's too large. Fix each failure before moving on.

## Step 1: Identify What Changed

Look at the recent changes (staged, unstaged, or recent commits) and determine which test files
are most likely affected. Use this mapping:

| Changed area | Tests to run first (`group/name` = `test/group/name.jl`) |
|---|---|
| `src/Grids/` | `unit/grids`, `unit/grid_reconstruction` |
| `src/Fields/` | `unit/field`, `abstract_operations/computed_field`, `unit/field_scans` |
| `src/Operators/` | `unit/operators`, `unit/vector_rotation_operators` |
| `src/BoundaryConditions/` | `unit/boundary_conditions`, `time_stepping/boundary_conditions_integration` |
| `src/Models/NonhydrostaticModels/` | `time_stepping/nonhydrostatic_models`, `time_stepping/time_stepping` |
| `src/Models/HydrostaticFreeSurfaceModels/` | `hydrostatic_free_surface/hydrostatic_free_surface_models`, `hydrostatic_free_surface/split_explicit_free_surface_solver` |
| `src/Models/ShallowWaterModels/` | `shallow_water/shallow_water_models` |
| `src/TimeSteppers/` | `time_stepping/time_stepping`, `time_stepping/dynamics` |
| `src/TurbulenceClosures/` | `turbulence_closures/turbulence_closures` |
| `src/Advection/` | `time_stepping/immersed_advection`, `time_stepping/dynamics` |
| `src/BuoyancyFormulations/` | `unit/buoyancy`, `time_stepping/seawater_density` |
| `src/Solvers/` | `poisson_solvers/poisson_solvers`, `general_solvers/batched_tridiagonal_solver` |
| `src/Simulations/` | `simulation/simulations`, `simulation/diagnostics` |
| `src/OutputWriters/` | `simulation/output_writers`, `simulation/jld2_writer`, `simulation/netcdf_writer` |
| `src/OutputReaders/` | `simulation/output_readers` |
| `src/ImmersedBoundaries/` | `unit/immersed_boundary_grid`, `hydrostatic_free_surface/hydrostatic_free_surface_immersed_boundaries` |
| `src/Coriolis/` | `coriolis/coriolis` |
| `src/Forcings/` | `time_stepping/forcings` |
| `src/Oceananigans.jl` (exports) | `unit/quality_assurance` |
| Docstrings | `unit/quality_assurance` (doctests checked via docs build) |
| `ext/` (extensions) | Corresponding `<extension>/<extension>` (e.g. `makie/makie`, `metal/metal`) |

## Step 2: Run the Most Likely Test First

Tests are selected by a prefix of their `group/name` and run in parallel worker processes.
Run a single test file on CPU:

```sh
CUDA_VISIBLE_DEVICES=-1 TEST_ARCHITECTURE=CPU julia --project -e '
using Pkg
Pkg.test("Oceananigans"; test_args=["unit/grids"])
'
```

Or run a whole group if the changes span multiple areas:

```sh
CUDA_VISIBLE_DEVICES=-1 TEST_ARCHITECTURE=CPU julia --project -e '
using Pkg
Pkg.test("Oceananigans"; test_args=["unit", "--jobs=4", "--verbose"])
'
```

`Pkg.test(; test_args=["--list"])` prints every test with its last measured duration. The
groups are the directories under `test/`: `unit`, `abstract_operations`, `coriolis`,
`tripolar_grid`, `poisson_solvers`, `general_solvers`, `turbulence_closures`, `time_stepping`,
`regression`, `hydrostatic_free_surface`, `vertical_coordinate`, `shallow_water`, `simulation`,
`lagrangian_particles`, `multi_region`, `conservative_regridding`, `scripts`, `memory_allocation`,
`init`; and, only when selected explicitly, `distributed`, `enzyme`, `reactant`, `sharding`,
`metal`, `amdgpu`, `oneapi`, `makie`, `convergence`. `TEST_GROUP=unit` is equivalent to
`test_args=["unit"]`.

## Step 3: Fix and Iterate

1. If a test fails, fix the issue
2. Re-run the same test to confirm the fix
3. Move on to the next most likely test
4. After direct tests pass, run `unit/quality_assurance` to catch import/doctest issues

## Notes

- GPU tests may fail with "dynamic invocation error" — always test on CPU first
- `unit/quality_assurance` checks explicit imports and Aqua.jl quality — run this for any change
- `test/mpi/` tests need 4 MPI ranks (`MPI_TEST=true mpiexec -n 4 julia --project -e 'using Pkg; Pkg.test()'` with `TEST_GROUP=distributed`) — skip unless changes touch `src/DistributedComputations/`
- If Julia version issues arise, delete `Manifest.toml` and run `Pkg.instantiate()`
