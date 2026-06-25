---
paths:
  - test/**/*.jl
---

# Testing Rules

## Running Tests

```julia
# All tests
Pkg.test("Oceananigans")

# Specific test group
ENV["TEST_GROUP"] = "unit"  # or "time_stepping", "regression", etc.
Pkg.test("Oceananigans")

# CPU-only (disable GPU)
ENV["CUDA_VISIBLE_DEVICES"] = "-1"
ENV["TEST_ARCHITECTURE"] = "CPU"
Pkg.test("Oceananigans")
```

## Writing Tests

- Place tests in `test/` directory following the existing group structure
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
- Make sure test files are actually included in `runtests.jl` (AI tools sometimes create orphaned files)
