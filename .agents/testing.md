# Testing Guidelines

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

* GPU tests may fail with "dynamic invocation error". In that case, the tests should be run on CPU.
  If the error goes away, the problem is GPU-specific, and often a type-inference issue.

## Writing Tests

- Place tests in `test/<group>/` and include `test/setup/dependencies_for_runtests.jl` at the top with `joinpath(@__DIR__, ...)`
- Every `.jl` file under `test/<group>/` is a test; helper files go in `test/setup/`
- Test on both CPU and GPU when possible
- Name test files descriptively (snake_case)
- Include both unit tests and integration tests
- Test numerical accuracy where analytical solutions exist

## Quality Assurance

- Ensure doctests pass
- Use Aqua.jl for package quality checks

## Debugging Tips

- Sometimes "Julia version compatibility" issues are resolved by deleting the Manifest.toml,
  and then re-populating it with `using Pkg; Pkg.instantiate()`.
- GPU tests may fail with "dynamic invocation error". Run on CPU first to isolate GPU-specific issues.
