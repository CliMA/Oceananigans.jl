# Documentation

## Building Docs Locally

```sh
julia --project=docs/ docs/make.jl
```

## Fast Local Docs Builds for Testing

When testing documentation changes locally (especially without a GPU), make these temporary changes to `docs/make.jl`:

1. **Comment out Literate examples** - These take the longest to run:
   ```julia
   example_scripts = String[
       # "spherical_baroclinic_instability.jl",
       # ... all examples commented out
   ]
   ```
   Also comment out the corresponding `example_pages` entries.

2. **Add error categories to `warnonly`** - Prevents build failures from GPU-dependent examples and network issues:
   ```julia
   warnonly = [:cross_references, :example_block, :linkcheck],
   ```

3. **Comment out GPU-requiring pages** - Pages like `simulation_tips.md` have `@example` blocks requiring GPU:
   ```julia
   # "Simulation tips" => "simulation_tips.md",  # requires GPU
   ```

4. **Optional speedups** in `makedocs`:
   - `doctest = false` - Skip doctests entirely if not testing those
   - `linkcheck = false` - Skip link validation
   - `draft = true` - Skip many checks for fastest builds

**Important**: Remember to revert these changes before committing!

## Viewing Docs

```julia
using LiveServer
serve(dir="docs/build")
```

## Documentation Style

- Use Documenter.jl syntax for cross-references
- Include code examples in documentation pages
- Add references to papers from the literature by adding bibtex to `oceananigans.bib`, and then
  a corresponding citation
- Make use of cross-references with equations
- In example code, use explicit imports as sparingly as possible. NEVER explicitly import a name that
  is already exported by the user interface. Always rely on `using Oceananigans` for imports and keep
  imports clean. Explicit imports should only be used for source code.

## Docstring Examples (CRITICAL)

**NEVER use plain `julia` code blocks in docstrings. ALWAYS use `jldoctest` blocks.**

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

### Wrong - never use plain `julia` blocks in docstrings:

~~~~
"""
    my_function(x)

Example:

```julia
# This code is NOT tested and may be wrong!
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
```
"""
~~~~

### Key doctest requirements

- Always include expected output after `# output`
- Use simple, verifiable output (e.g., `typeof(result)`, accessing a field)
- Exercise `Base.show` to verify objects display correctly
- Keep doctests minimal but complete enough to verify the feature works
- **Do NOT use boolean comparisons as the final line** (avoid `x ≈ 1.0` or `obj isa Type`)
- Instead, make the final line invoke a `show` method — this tests both the feature and `show` quality
- Example of what NOT to do:
  ```julia
  plt = surface!(ax, T)
  plt isa CairoMakie.Surface  # BAD: boolean comparison
  # output
  true
  ```
- Example of what TO do:
  ```julia
  x, y, z = spherical_coordinates(0.0, 0.0)
  (x, y, z)  # GOOD: shows the actual output
  # output
  (1.0, 0.0, 0.0)
  ```

## Writing Examples

- Explain at the top of the file what a simulation is doing
- Let code "speak for itself" as much as possible, to keep an explanation concise.
  In other words, use a Literate style.
- Use visualization interspersed with model setup or simulation running when needed to
  give an understanding of a complex grid, initial condition, or other model property.
- Look at previous examples. New examples should add as much value as possible while
  remaining simple. This requires judiciously introducing new features and doing creative
  and surprising things with simulations that will spark readers' imagination.
- Don't "over import". Use names that are exported by `using Oceananigans`. If there are
  names that are not exported, but are needed in common/basic examples, consider
  exporting those names from `Oceananigans.jl`.
- **Literate.jl comment conventions**: Examples in `examples/` are processed by Literate.jl.
  - Single `#` comments become markdown blocks in the generated documentation
  - Double `##` comments remain as code comments within code blocks
  - Use `##` for inline code comments that should stay with the code (e.g., `## Helper function`)
  - Use single `#` only for narrative text that should render as markdown
