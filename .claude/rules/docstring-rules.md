---
paths:
  - src/**/*.jl
---

# Docstring Rules

## Use DocStringExtensions.jl

- Include `$(SIGNATURES)` for automatic signature documentation
- Add examples in docstrings when helpful

## CRITICAL: Always use `jldoctest`, NEVER plain `julia` blocks

Plain code blocks (`` ```julia ``) are NOT tested and can become stale or incorrect.
Doctests (`` ```jldoctest ``) are automatically tested and verified to work.

### Correct - use `jldoctest`:

~~~~
"""
    my_function(x)

Example:

```jldoctest
using Oceananigans

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
typeof(grid)

# output
RectilinearGrid{Float64, Periodic, Periodic, Bounded, Nothing, Nothing, Nothing, Nothing}
```
"""
~~~~

### Wrong - never use plain `julia` blocks:

~~~~
"""
    my_function(x)

```julia
# This code is NOT tested and may be wrong!
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
```
"""
~~~~

## Doctest Best Practices

- Always include expected output after `# output`
- Use simple, verifiable output (e.g., `typeof(result)`, accessing a field)
- Doctests should exercise `Base.show` to verify objects display correctly
- Keep doctests minimal but complete enough to verify the feature works
- **Do NOT use boolean comparisons as the final line** (e.g., avoid `x ≈ 1.0` or `obj isa Type`)
- Instead, make the final line invoke a `show` method that prints something useful
