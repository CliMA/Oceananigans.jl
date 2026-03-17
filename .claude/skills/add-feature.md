---
name: add-feature
description: Checklist for adding new physics or features to Oceananigans
user_invocable: true
---

# Add Feature

Follow this checklist when adding new physics or features to Oceananigans.

## Checklist

1. **Create module** in the appropriate subdirectory under `src/`
2. **Define types/structs** with docstrings (use `jldoctest` blocks, never plain `julia`)
3. **Implement kernel functions** - must be GPU-compatible:
   - Use `@kernel` and `@index` from KernelAbstractions.jl
   - Keep type-stable and allocation-free
   - Use `ifelse` instead of short-circuiting `if`/`else`
   - Mark inner functions with `@inline`
   - No loops over grid points outside kernels - use `launch!`
4. **Add unit tests** in `test/` following existing group structure
5. **Update exports** in `src/Oceananigans.jl` if the user interface changed
6. **Add validation examples** in `validation/` or `examples/` when appropriate
7. **Verify on CPU**: Run tests with `ENV["TEST_ARCHITECTURE"] = "CPU"`
8. **Check explicit imports**: Tests automatically verify proper imports

## Key Conventions

- File names: snake_case
- Type names: PascalCase
- Function names: snake_case
- Kernel names: may be prefixed with underscore (e.g., `_compute_tendency_kernel`)
- Model constructors: `grid` is positional, keyword args after semicolon
