---
paths:
  - test/**/*.jl
---

# Testing Rules

## Running Tests

```julia
# Default test suite (everything that needs no extra hardware); files run in parallel worker processes
Pkg.test("Oceananigans")

# Select tests by prefix of "group/name", i.e. of the path test/<group>/<name>.jl
Pkg.test("Oceananigans"; test_args=["unit"])                  # all of test/unit/
Pkg.test("Oceananigans"; test_args=["unit/grids", "coriolis"]) # one file plus one group
Pkg.test("Oceananigans"; test_args=["--list"])                # list tests with last durations
Pkg.test("Oceananigans"; test_args=["--jobs=4", "--verbose"])

# TEST_GROUP is an equivalent, comma-separated selection (used by CI)
ENV["TEST_GROUP"] = "unit"
Pkg.test("Oceananigans")

# CPU-only (disable GPU)
ENV["CUDA_VISIBLE_DEVICES"] = "-1"
ENV["TEST_ARCHITECTURE"] = "CPU"
Pkg.test("Oceananigans")
```

## Writing Tests

- Place tests in `test/<group>/` and include `test/setup/dependencies_for_runtests.jl` at the top with `joinpath(@__DIR__, ...)`; every `.jl` file there is a test, helper files go in `test/setup/`
- Test on both CPU and GPU when possible
- Name test files descriptively (snake_case)
- Include both unit tests and integration tests
- Test numerical accuracy where analytical solutions exist

## Debugging

- GPU "dynamic invocation error": run on CPU first to isolate GPU-specific issues
- Julia version issues: delete Manifest.toml, then `Pkg.instantiate()`
- Ensure doctests pass; use Aqua.jl for package quality checks

## Quality

- Ensure all explicit imports are correct (tests check this automatically)
- Always add tests for new functionality
- **Avoid `@allowscalar` in new tests** — transfer data to CPU with `Array(interior(field))` first
- Use minimal grid sizes to reduce CI time
- Avoid hardcoded grid indices — use `size(grid, d)` instead of literal numbers
- Each test file must be self-contained: it runs in its own module and worker process, so it cannot rely on names from other test files
