# Testing Guidelines

## Running Tests

```julia
# All tests
Pkg.test("Oceananigans")

# Run specific test groups by setting the TEST_GROUP environment variable
ENV["TEST_GROUP"] = "unit"  # or "time_stepping", "regression", etc.
Pkg.test("Oceananigans")

# CPU-only (disable GPU)
ENV["CUDA_VISIBLE_DEVICES"] = "-1"
ENV["TEST_ARCHITECTURE"] = "CPU"
Pkg.test("Oceananigans")
```

* GPU tests may fail with "dynamic invocation error". In that case, the tests should be run on CPU.
  If the error goes away, the problem is GPU-specific, and often a type-inference issue.

## Writing Tests

- Place tests in `test/` directory
- Follow the existing test group structure
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
