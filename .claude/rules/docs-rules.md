---
paths:
  - docs/**/*
---

# Documentation Rules

## Building Docs

```sh
julia --project=docs/ docs/make.jl
```

## Fast Local Builds

For local testing (especially without GPU), temporarily modify `docs/make.jl`:
1. Comment out Literate examples in `example_scripts` and `example_pages`
2. Add `warnonly = [:cross_references, :example_block, :linkcheck]`
3. Comment out GPU-requiring pages (e.g., `simulation_tips.md`)
4. Optional: `doctest = false`, `linkcheck = false`, `draft = true`

**Remember to revert these changes before committing!**

## Viewing Docs

```julia
using LiveServer
serve(dir="docs/build")
```

## Style

- Use Documenter.jl syntax for cross-references
- Add paper references via bibtex in `oceananigans.bib` with corresponding citations
- Make use of cross-references with equations
- In example code, NEVER explicitly import names already exported by `using Oceananigans`

## Docstrings

- ALWAYS use `jldoctest` blocks, NEVER plain `julia` blocks
- See `.claude/rules/docstring-rules.md` for full details
