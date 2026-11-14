---
paths:
  - src/**/*.jl
---

# Kernel Function Rules

GPU-compatible kernel functions are critical for Oceananigans performance.

## Requirements

- Use KernelAbstractions.jl syntax: `@kernel`, `@index`
- Keep kernels **type-stable** and **allocation-free**
- Use `ifelse` instead of short-circuiting `if`/`else` statements
- No error messages inside kernels
- Models **never** go inside kernels
- Mark functions called inside kernels with `@inline`
- **Never use loops outside kernels**: Always replace `for` loops that iterate over grid points
  with kernels launched via `launch!`. This ensures code works on both CPU and GPU.

## Type Stability

- All structs must be concretely typed
- Type instability in kernel functions ruins GPU performance
- Julia compiler can infer types; use type annotations primarily for **multiple dispatch**, not documentation

## Numeric Types

- **Never hardcode Float64**: no literal `0.0` or `1.0` in kernels
- Use `zero(grid)`, `one(grid)`, `convert(FT, 1//2)`, or rational literals
- Use `on_architecture` for data transfers — never manual `Array()` / `CuArray()` calls

## Memory Efficiency

- Favor inline computations over allocating temporary memory
- Minimize memory allocation overall
- Design solutions that work within the existing framework

## Staggered Grid & Indexing

- Velocities live at cell faces, tracers at cell centers (Arakawa C-grid)
- Take care of staggered grid location when writing operators or designing diagnostics
- **Always use 3D indexing** for fields (`field[i, j, k]`); 2D indexing works by coincidence
  but is unsupported and may break
