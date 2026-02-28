# Oceananigans.jl — Agent Rules

## Project Overview

Oceananigans.jl is a Julia package for fast, friendly, flexible, ocean-flavored fluid dynamics on CPUs and GPUs.
It solves the incompressible (Boussinesq) Navier-Stokes equations with models including:
nonhydrostatic (with free surfaces), hydrostatic free-surface, and shallow water —
on RectilinearGrid, LatitudeLongitudeGrid, CubedSphereGrid, and ImmersedBoundaryGrid.

## Language & Environment

- **Julia 1.10+** | CPU and GPU (CUDA, AMD, Metal, OneAPI)
- **Key packages**: KernelAbstractions.jl, CUDA.jl, Enzyme.jl, Reactant.jl
- **Style**: ExplicitImports.jl for source code; `using Oceananigans` for examples/tests

## Critical Rules

### Kernel Functions (GPU compatibility)

- Use `@kernel` / `@index` (KernelAbstractions.jl)
- Kernels must be **type-stable** and **allocation-free**
- Use `ifelse` — never short-circuiting `if`/`else` in kernels
- No error messages, no Models inside kernels
- Mark functions called inside kernels with `@inline`
- **Never loop over grid points outside kernels** — use `launch!`

### Type Stability & Memory

- All structs must be concretely typed
- Type annotations are for **dispatch**, not documentation
- Minimize allocation; favor inline computation
- **Never hardcode Float64**: no literal `0.0` or `1.0` in kernels or constructors.
  Use `zero(grid)`, `one(grid)`, `convert(FT, 1//2)`, or rational literals

### Imports

- Source code: explicit imports (checked by tests)
- Examples/docs: rely on `using Oceananigans`; never explicitly import exported names

### Docstrings

- Use DocStringExtensions.jl with `$(SIGNATURES)`
- **ALWAYS `jldoctest` blocks, NEVER plain `julia` blocks** — doctests are tested; plain blocks rot
- Include `# output` with verifiable output; prefer `show` methods over boolean comparisons
- Use unicode for math (`Δt`, `η`, `ρ`), not LaTeX — LaTeX doesn't render in the REPL

### Model Constructors

- `grid` is positional: `NonhydrostaticModel(grid; closure=nothing)`
- `ShallowWaterModel(grid, gravitational_acceleration; ...)` — both positional
- Omit semicolon when there are no keyword arguments: `NonhydrostaticModel(grid)` not `NonhydrostaticModel(grid;)`

## Naming Conventions

- **Files**: snake_case matching the type they define — `nonhydrostatic_model.jl`
- **Types/Constructors**: PascalCase **only for true constructors** — `NonhydrostaticModel`
- **Functions**: snake_case — `time_step!`; functions that return values are never PascalCase
- **Kernels**: may prefix with underscore — `_compute_tendency_kernel`
- **Variables**: English long name or readable unicode math notation — never mix abbreviated and
  full forms (e.g., `cond` vs `condition`) to imply a difference; be specific

## Module Structure

```
src/
├── Oceananigans.jl            # Main module, exports
├── Architectures.jl           # CPU/GPU architecture abstractions
├── Grids/                     # Grid types and constructors
├── Fields/                    # Field types and operations
├── Operators/                 # Finite difference operators
├── BoundaryConditions/        # Boundary condition types
├── Models/                    # Model implementations
│   ├── NonhydrostaticModels/
│   ├── HydrostaticFreeSurfaceModels/
│   ├── ShallowWaterModels/
│   └── LagrangianParticleTracking/
├── TimeSteppers/              # Time stepping schemes
├── Solvers/                   # Poisson and tridiagonal solvers
├── TurbulenceClosures/        # LES and eddy viscosity models
├── Advection/                 # Advection schemes
├── BuoyancyFormulations/      # Buoyancy models
├── OutputWriters/             # File I/O
├── Simulations/               # High-level simulation interface
└── Utils/                     # Utilities and helpers
```

## Common Pitfalls

1. **Type instability** in kernels ruins GPU performance
2. **Overconstraining types**: use annotations for dispatch, not documentation
3. **Missing imports**: tests will catch this — add to `using` statements
4. **Plain `julia` blocks in docstrings**: always use `jldoctest`
5. **Subtle bugs from missing method imports**, especially in extensions
6. **Expecting unexported names**: consider exporting them rather than changing user scripts
7. **Extending `getproperty` to fix undefined property bugs**: fix on the caller side instead
8. **"Type is not callable" errors**: variable name shadows a function — rename or qualify
9. **Quick fixes that break correctness**: if a test fails after a change, revisit the original edit
10. **Commented-out code**: delete it. Git is the journal — don't leave commented code, debugging
    artifacts, or stale copy-paste remnants
11. **2D indexing on fields**: always use 3D indexing (`field[i, j, k]`). 2D indexing works by
    coincidence on some fields but is unsupported and will break
12. **Hardcoded Float64**: never use `0.0`, `1.0` in kernels or constructors; use `zero(grid)` etc.
13. **Scope creep in PRs**: keep changes focused on a single concern. Unrelated cleanup goes
    in a separate PR

## Git Workflow

Follow [ColPrac](https://github.com/SciML/ColPrac). Feature branches, descriptive commits,
update tests and docs with code changes, check CI before merging.

## Design Principles

- **Dispatch over conditionals**: use Julia's type system and multiple dispatch instead of
  `if`/`else` branching. Backend-specific code goes in `ext/` extensions, not `if` branches in `src/`
- **Use `on_architecture` for data transfers** — never manual `Array()` / `CuArray()` calls
- **Defaults serve the common case**: avoid `nothing` defaults when a concrete default (like `CPU()`)
  covers 80% of usage. Minimize boilerplate for the typical user.
- **Keyword argument names must be consistent** across related types and constructors
- **Always use explicit `return`** in functions longer than one expression
- **One operation per line** as default; break long expressions across lines

## Agent Behavior

- Prioritize type stability and GPU compatibility
- Follow established patterns in existing code
- Add tests for new functionality; update exports when adding public API
- Reference physics equations in comments when implementing dynamics

## Further Reading

Detailed reference docs are in `.agents/` — read on demand:

| Document | Content |
|----------|---------|
| [`.agents/testing.md`](.agents/testing.md) | Full testing guidelines, running tests, debugging |
| [`.agents/documentation.md`](.agents/documentation.md) | Building docs, fast builds, doctest details, writing examples |
| [`.agents/validation.md`](.agents/validation.md) | Reproducing paper results step-by-step |
| [`.agents/physics.md`](.agents/physics.md) | Ocean/fluid dynamics and numerical methods background |

### Auto-loading Rules

Rules in `.claude/rules/` load automatically when you touch matching files:
- `kernel-rules.md` — GPU kernel requirements (src/)
- `docstring-rules.md` — docstring and jldoctest conventions (src/)
- `testing-rules.md` — test writing and running (test/)
- `docs-rules.md` — documentation building and style (docs/)
- `examples-rules.md` — Literate.jl example conventions (examples/)

### Skills (slash commands)

- `/run-tests` — run targeted tests, prioritized by what's likely to break
- `/build-docs` — build documentation locally
- `/add-feature` — checklist for adding new physics/features
- `/new-simulation` — set up, run, and visualize a new simulation (with or without a reference paper)
- `/babysit-ci` — monitor CI, auto-fix small issues, pause on bigger problems, retrigger flaky runs
