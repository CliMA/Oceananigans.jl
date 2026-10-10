---
paths:
  - src/**/*.jl
  - ext/**/*.jl
---

# Docstring Rules

- Use `$(TYPEDSIGNATURES)` from DocStringExtensions when the signature has no default values for
  positional or keyword arguments. Write the signature by hand when it does, so the defaults show.
- Code examples in docstrings are `jldoctest` blocks, not `julia` blocks. Doctests run in the
  documentation build; plain `julia` blocks are never executed and go stale silently.
- End a doctest with an expression whose `show` output is worth reading, and put that output after
  `# output`. This tests the feature and its `show` method at once. A final line such as
  `x ≈ 1.0` or `obj isa Type` prints `true` and tests almost nothing.
- Write math in Unicode (`Δt`, `η`, `ρ`), not LaTeX; docstrings are read in the REPL, where LaTeX
  does not render.

~~~~
"""
    my_function(grid)

Example:

```jldoctest
using Oceananigans

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
my_function(grid)

# output
<the printed result>
```
"""
~~~~
