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

| Changed area | Test files to run first |
|---|---|
| `src/Grids/` | `test_grids.jl`, `test_grid_reconstruction.jl` |
| `src/Fields/` | `test_field.jl`, `test_computed_field.jl`, `test_field_scans.jl` |
| `src/Operators/` | `test_operators.jl`, `test_vector_rotation_operators.jl` |
| `src/BoundaryConditions/` | `test_boundary_conditions.jl`, `test_boundary_conditions_integration.jl` |
| `src/Models/NonhydrostaticModels/` | `test_nonhydrostatic_models.jl`, `test_time_stepping.jl` |
| `src/Models/HydrostaticFreeSurfaceModels/` | `test_hydrostatic_free_surface_models.jl`, `test_split_explicit_free_surface_solver.jl` |
| `src/Models/ShallowWaterModels/` | `test_shallow_water_models.jl` |
| `src/TimeSteppers/` | `test_time_stepping.jl`, `test_dynamics.jl` |
| `src/TurbulenceClosures/` | `test_turbulence_closures.jl` |
| `src/Advection/` | `test_immersed_advection.jl`, `test_dynamics.jl` |
| `src/BuoyancyFormulations/` | `test_buoyancy.jl`, `test_seawater_density.jl` |
| `src/Solvers/` | `test_poisson_solvers.jl`, `test_batched_tridiagonal_solver.jl` |
| `src/Simulations/` | `test_simulations.jl`, `test_diagnostics.jl` |
| `src/OutputWriters/` | `test_output_writers.jl`, `test_jld2_writer.jl`, `test_netcdf_writer.jl` |
| `src/OutputReaders/` | `test_output_readers.jl` |
| `src/ImmersedBoundaries/` | `test_immersed_boundary_grid.jl`, `test_hydrostatic_free_surface_immersed_boundaries.jl` |
| `src/Coriolis/` | `test_coriolis.jl` |
| `src/Forcings/` | `test_forcings.jl` |
| `src/Oceananigans.jl` (exports) | `test_quality_assurance.jl` |
| Docstrings | `test_quality_assurance.jl` (doctests checked via docs build) |
| `ext/` (extensions) | Corresponding `test_<extension>.jl` |

## Step 2: Run the Most Likely Test First

Run a single test file on CPU:

```sh
julia --project -e '
using Pkg
ENV["CUDA_VISIBLE_DEVICES"] = "-1"
ENV["TEST_ARCHITECTURE"] = "CPU"
ENV["TEST_FILE"] = "test_grids.jl"
Pkg.test("Oceananigans")
'
```

Or run a test group if the changes span multiple areas:

```sh
julia --project -e '
using Pkg
ENV["CUDA_VISIBLE_DEVICES"] = "-1"
ENV["TEST_ARCHITECTURE"] = "CPU"
ENV["TEST_GROUP"] = "unit"
Pkg.test("Oceananigans")
'
```

Available test groups: `init`, `unit`, `abstract_operations`, `tripolar_grid`,
`poisson_solvers_1`, `poisson_solvers_2`, `general_solvers`, `simulation`,
`lagrangian_particles`, `time_stepping_1`, `time_stepping_2`, `time_stepping_3`,
`turbulence_closures`, `shallow_water`, `hydrostatic_free_surface`, `multi_region`,
`nonhydrostatic_regression`, `hydrostatic_regression`, `vertical_coordinate`,
`enzyme`, `reactant_1`, `reactant_2`, `makie`, `convergence`, `scripts`

## Step 3: Fix and Iterate

1. If a test fails, fix the issue
2. Re-run the same test to confirm the fix
3. Move on to the next most likely test
4. After direct tests pass, run `test_quality_assurance.jl` to catch import/doctest issues

## Notes

- GPU tests may fail with "dynamic invocation error" — always test on CPU first
- `test_quality_assurance.jl` checks explicit imports and Aqua.jl quality — run this for any change
- Distributed/MPI tests require special setup — skip unless changes touch `src/DistributedComputations/`
- If Julia version issues arise, delete `Manifest.toml` and run `Pkg.instantiate()`
