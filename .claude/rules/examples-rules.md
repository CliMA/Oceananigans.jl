---
paths:
  - examples/**/*.jl
---

# Examples Rules

## Writing Examples

- Explain at the top of the file what a simulation is doing
- Let code "speak for itself" - keep explanations concise (Literate style)
- Use visualization interspersed with model setup when needed to illustrate
  complex grids, initial conditions, or other model properties
- New examples should add value while remaining simple: judiciously introduce
  new features and do creative, surprising things with simulations
- Don't "over import". Use names exported by `using Oceananigans`. If needed
  names aren't exported, consider exporting them from `Oceananigans.jl`

## Literate.jl Comment Conventions

Examples in `examples/` are processed by Literate.jl:
- Single `#` comments become markdown blocks in generated documentation
- Double `##` comments remain as code comments within code blocks
- Use `##` for inline code comments that should stay with the code
- Use single `#` only for narrative text that should render as markdown
