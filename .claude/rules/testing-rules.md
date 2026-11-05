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

## Quiet Tests

Tests run under ParallelTestRunner, which captures each file's output and prints it in the CI log.
Anything a test prints is noise there, so new tests should print nothing:

- **No `@info "Testing ..."` banners**: ParallelTestRunner does not show progress. Put any extra
  information in the `@testset` name instead, using `@testset "... [$FT]" for FT in float_types`
  for loops.
- **Never print to check output**: test it. Use `@test sprint(show, x) == "..."` and
  `@test summary(x) == "..."`, not `show(x); println()`. For parts that vary with the architecture,
  such as array types, interpolate (`$(summary(arch))`) or use `startswith`/`endswith`.
- **Silence what isn't under test**: pass `verbose=false` to `Simulation`, and keep output writers
  and checkpointers non-verbose.
- **Check expected messages with `@test_logs`**: this covers warnings and logs printed whatever the
  verbosity, such as checkpoint pickup, appending to existing files or FFT solvers on immersed grids.
  Give the exact sequence of messages, and use regexes only for timings and file sizes.
  `@test_logs` returns the value of its expression, so it can wrap constructors:
  `model = @test_logs (:warn, r"^...") NonhydrostaticModel(grid; ...)`. Splat an empty tuple when
  a message is expected only conditionally, and build long sequences with small helper functions.
- **Write files to a temporary directory**, never the working directory. Use `dir = mktempdir()`,
  pass `dir` to writers and checkpointers, build expected paths with `joinpath(dir, ...)`, and
  finish with `rm(dir; recursive=true)`.
- **Each test file runs in its own module**, so interpolated types print module-qualified, e.g.
  `Oceananigans.Grids.Periodic`. In `summary`/`show` methods, print type names with `nameof`. When
  checking printed output locally, include the file inside a module
  (`module Sandbox; include("test/...jl"); end`) or run it through `Pkg.test`, not `include` in `Main`.
