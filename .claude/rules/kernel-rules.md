---
paths:
  - src/**/*.jl
  - ext/**/*.jl
---

# Kernels, Operators, and Grid Locations

Source code runs on CPUs and GPUs from one implementation. CPU tests pass on code that fails or
crawls on a GPU, so these rules carry the GPU constraints that a CPU run will not reveal.

## Kernels

- Write kernels with `@kernel` / `@index` and launch them with `launch!(arch, grid, :xyz, kernel!, args...)`.
  Never loop over grid points outside a kernel: on a GPU that is scalar indexing, which is
  either an error or thousands of device round trips.
- Functions called inside kernels are `@inline`. Kernels compose deep stacks of small operators,
  and inlining lets the compiler optimize across them.
- Kernels must be type-stable and allocation-free. A type instability that the CPU tolerates
  becomes a "dynamic invocation" error on the GPU.
- Use `ifelse` rather than `if`/`else`, `&&`, `||`, or `? :` on values that vary across the grid.
  Threads in a warp that take different branches run both paths serially. `ifelse` evaluates
  *both* arguments, so each must be valid for every index it can be called with.
- No error messages, `@assert`, or string interpolation inside kernels. Validate in the
  constructor, before launch.
- Only device-compatible objects can be kernel arguments. A model is not one: pass its fields,
  grid, closure, and so on. A new struct that holds arrays needs an `Adapt.adapt_structure` method
  (see `src/Forcings/continuous_forcing.jl`) and usually an `on_architecture` method.
- Never reassign a variable captured by a closure that reaches a kernel (forcing functions,
  boundary condition functions, masks). Reassignment turns the capture into a `Core.Box`, the
  closure is no longer `isbits`, and the launch fails, often only on a GPU.

```julia
# wrong: x₀ becomes a Core.Box
x₀ = Lx / 2
x₀ = x₀ + Δ
mask(x, y, z) = x > x₀

# right: single assignment
x₀ = Lx / 2 + Δ
mask(x, y, z) = x > x₀
```

## Floating point

Fields may be `Float32`. A `Float64` literal promotes the whole expression to `Float64`, which is
slow on GPUs and silently changes the precision of a `Float32` model.

```julia
# wrong
@inline ℑx(i, j, k, grid, c) = 0.5 * (c[i-1, j, k] + c[i, j, k])

# right (src/Operators/interpolation_operators.jl)
@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i-1, j, k] + c[i, j, k])
```

Use `zero(grid)`, `one(grid)`, `convert(FT, x)`, or rational literals (`1//2`) instead.

## Grid locations

- Oceananigans uses a staggered C-grid: tracers at cell centers, velocity components at the faces
  normal to them. Every field and operator has a location `(LX, LY, LZ)` built from `Center`,
  `Face`, and `Nothing` (a reduced dimension).
- Operator superscripts give the location of the *result*, in x, y, z order; `ᵃ` means "any".
  `δxᶠᵃᵃ(i, j, k, grid, c) = c[i, j, k] - c[i-1, j, k]` takes a center field to x-faces.
  Face `i` therefore lies between centers `i-1` and `i`. Check that each operator you compose
  receives a field at the location it expects; a mismatch compiles and runs and gives wrong answers.
- In a `Bounded` direction, face fields have `N + 1` interior points and center fields have `N`.
  Use `size(field)` or `size(grid, (Face(), Center(), Center()))`, not `grid.Nx`, for a face field.
- Index fields with three indices, `field[i, j, k]`, even when a dimension is reduced or flat.
  Two-index access works on some fields by coincidence and is unsupported.
- Operators read neighbors from the halo. After writing to a field's interior in your own kernel,
  call `fill_halo_regions!` before anything reads across the boundary.

## Types

- Structs are concretely typed; use type parameters, not abstract field types.
- Type annotations on methods are for dispatch, not documentation. Over-constraining them breaks
  `Float32`, GPU arrays, and Reactant's traced arrays.
- Prefer computing values inline in the kernel to allocating temporary fields.
- If an implementation is awkward, there may be an existing abstraction for it in `Operators`,
  `AbstractOperations`, or `Utils`; search before adding a new one.
