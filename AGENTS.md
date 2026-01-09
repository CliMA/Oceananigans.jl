# Oceananigans.jl rules for agent-assisted development

## Project Overview

Oceananigans.jl is a Julia package for fast, friendly, flexible, ocean-flavored fluid dynamics simulations on CPUs and GPUs.
It provides a framework for solving the incompressible (or Boussinesq) Navier-Stokes equations with various model configurations including:
- Nonhydrostatic models with free surfaces
- Hydrostatic models for large-scale ocean simulations  
- Shallow water models
- Support for a variety of grids: RectilinearGrid, LatitudeLongitudeGrid, CubedSphereGrid
- Support for complex domains using ImmersedBoundaryGrid

## Language & Environment
- **Language**: Julia 1.10+
- **Architectures**: CPU and GPU (CUDA, AMD, Metal, OneAPI)
- **Key Packages**: KernelAbstractions.jl, CUDA.jl, Enzyme.jl, Reactant.jl, AMDGPU.jl, Metal.jl, oneAPI.jl
- **Testing**: Comprehensive test suite covering all model types and features

## Code Style & Conventions

### Julia Best Practices
1. **Explicit Imports**: Use `ExplicitImports.jl` style - explicitly import all used functions/types
   - Import from modules explicitly (already done in src/Oceananigans.jl)
   - Tests automatically check for proper imports
   
2. **Type Stability**: Prioritize type-stable code for performance
   - All structs must be concretely typed
   
3. **Kernel Functions**: For GPU compatibility:
   - Use KernelAbstractions.jl syntax for kernels, eg `@kernel`, `@index`
   - Keep kernels type-stable and allocation-free
   - Short-circuiting if-statements should be avoided if possible, ifelse should always be used if possible
   - No error messages inside kernels
   - Models _never_ go inside kernels
   - Mark functions called inside kernels with `@inline`
   - **Never use loops outside kernels**: Always replace `for` loops that iterate over grid points
     with kernels launched via `launch!`. This ensures code works on both CPU and GPU.
   
4. **Documentation**:
   - Use DocStringExtensions.jl for consistent docstrings
   - Include `$(SIGNATURES)` for automatic signature documentation
   - Add examples in docstrings when helpful
   - **CRITICAL: ALWAYS use `jldoctest` blocks, NEVER use plain `julia` blocks in docstrings**
     (see "Docstring Examples" section below for details)

5. **Memory efficiency**
   - Favor doing lots of computations inline versus allocating temporary memory
   - Generally minimize memory allocation
   - Design solutions that work within the existing framework

6. **Model Constructor Formatting**: Model constructors use positional arguments for required parameters
   - **HydrostaticFreeSurfaceModel**: `HydrostaticFreeSurfaceModel(grid; ...)` - `grid` is positional
   - **NonhydrostaticModel**: `NonhydrostaticModel(grid; ...)` - `grid` is positional
   - **ShallowWaterModel**: `ShallowWaterModel(grid; gravitational_acceleration, ...)` - both `grid` and `gravitational_acceleration` are positional
   - **Important**: When there are no keyword arguments, omit the semicolon:
     - ✅ `NonhydrostaticModel(grid)` 
     - ❌ `NonhydrostaticModel(grid;)`
   - When keyword arguments are present, use the semicolon:
     - ✅ `NonhydrostaticModel(grid; closure=nothing)`
     - ✅ `HydrostaticFreeSurfaceModel(grid; tracers=:c)`

### Naming Conventions
- **Files**: snake_case (e.g., `nonhydrostatic_model.jl`, `compute_hydrostatic_free_surface_tendencies.jl`)
- **Types**: PascalCase (e.g., `NonhydrostaticModel`, `HydrostaticFreeSurfaceModel`, `SeawaterBuoyancy`)
- **Functions**: snake_case (e.g., `time_step!`, `compute_tendencies!`)
- **Kernels**: "Kernels" (functions prefixed with `@kernel`) may be prefixed with an underscore (e.g., `_compute_tendency_kernel`)
- **Variables**: Use _either_ an English long name, or mathematical notation with readable unicode

### Module Structure
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

## Testing Guidelines

### Running Tests
```julia
# All tests
Pkg.test("Oceananigans")

# Run specific test groups by setting the TEST_GROUP environment variable
ENV["TEST_GROUP"] = "unit"  # or "time_stepping", "regression", etc.
Pkg.test("Oceananigans")

# CPU-only (disable GPU)
ENV["CUDA_VISIBLE_DEVICES"] = "-1"
ENV["TEST_ARCHITECTURE"] = "CPU"
Pkg.test("Oceananigans")
```

* GPU tests may fail with "dynamic invocation error". In that case, the tests should be run on CPU.
  If the error goes away, the problem is GPU-specific, and often a type-inference issue.

### Writing Tests
- Place tests in `test/` directory
- Follow the existing test group structure
- Test on both CPU and GPU when possible
- Name test files descriptively (snake_case)
- Include both unit tests and integration tests
- Test numerical accuracy where analytical solutions exist

### Quality Assurance
- Ensure doctests pass
- Use Aqua.jl for package quality checks

### Debugging Tips
- Sometimes "Julia version compatibility" issues are resolved by deleting the Manifest.toml,
  and then re-populating it with `using Pkg; Pkg.instantiate()`.
- GPU tests may fail with "dynamic invocation error". Run on CPU first to isolate GPU-specific issues.

### Docstring Examples (CRITICAL)

**NEVER use plain `julia` code blocks in docstrings. ALWAYS use `jldoctest` blocks.**

Plain code blocks (`` ```julia ``) are NOT tested and can become stale or incorrect.
Doctests (`` ```jldoctest ``) are automatically tested and verified to work.

✅ CORRECT - use `jldoctest`:
```markdown
\"\"\"
    my_function(x)

Example:

```jldoctest
using Oceananigans

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
typeof(grid)

# output
RectilinearGrid{Float64, Periodic, Periodic, Bounded, Nothing, Nothing, Nothing, Nothing}
```
\"\"\"
```

❌ WRONG - never use plain `julia` blocks in docstrings:
```markdown
\"\"\"
    my_function(x)

Example:

```julia
# This code is NOT tested and may be wrong!
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
```
\"\"\"
```

Key doctest requirements:
- Always include expected output after `# output`
- Use simple, verifiable output (e.g., `typeof(result)`, accessing a field that returns a simple value)
- Doctests should exercise `Base.show` to verify objects display correctly
- Keep doctests minimal but complete enough to verify the feature works

## Common Development Tasks

### Adding New Physics or Features
1. Create module in appropriate subdirectory
2. Define types/structs with docstrings
3. Implement kernel functions (GPU-compatible)
4. Add unit tests
5. If the user interface is changed, update main module exports in `src/Oceananigans.jl`
6. Add validation examples in `validation/` or `examples/` when appropriate

### Modifying Core Models
- Nonhydrostatic model: `src/Models/NonhydrostaticModels/`
- Hydrostatic model: `src/Models/HydrostaticFreeSurfaceModels/`
- Shallow water model: `src/Models/ShallowWaterModels/`
- Time stepping: `src/TimeSteppers/`
- Always maintain compatibility with existing model types

## Documentation

### Building Docs Locally
```sh
julia --project=docs/ docs/make.jl
```

### Viewing Docs
```julia
using LiveServer
serve(dir="docs/build")
```

### Documentation Style
- Use Documenter.jl syntax for cross-references
- Include code examples in documentation pages
- Add references to papers from the literature by adding bibtex to `oceananigans.bib`, and then
  a corresponding citation
- Make use of cross-references with equations
- In example code, use explicit imports as sparingly as possible. NEVER explicitly import a name that
  is already exported by the user interface. Always rely on `using Oceananigans` for imports and keep
  imports clean. Explicit imports should only be used for source code.

### Writing Doctests
- Use `jldoctest` blocks for testable examples in docstrings
- **Do NOT use boolean comparisons as the final line** (e.g., avoid `x ≈ 1.0` or `obj isa Type`)
- Instead, make the final line invoke a `show` method that prints something useful
- This serves two purposes:
  1. Helps users understand what the code produces
  2. Tests that our `show` methods are high quality and informative
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

### Writing examples
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

## Important Files to Know

### Core Implementation
- `src/Oceananigans.jl` - Main module, all exports
- `src/Models/NonhydrostaticModels/nonhydrostatic_model.jl` - Nonhydrostatic model definition
- `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl` - Hydrostatic model
- `src/Grids/` - Grid implementations
- `src/Fields/` - Field types and operations
- `src/TimeSteppers/` - Time integration schemes

### Configuration
- `Project.toml` - Package dependencies and compat bounds
- `test/runtests.jl` - Test configuration

### Examples
- `examples/langmuir_turbulence.jl` - Ocean mixed layer with Langmuir turbulence
- `examples/internal_wave.jl` - Internal wave propagation
- `examples/shallow_water_Bickley_jet.jl` - Shallow water instability
- `examples/baroclinic_adjustment.jl` - Baroclinic instability
- `examples/two_dimensional_turbulence.jl` - 2D turbulence
- Many more examples in the `examples/` directory

## Physics Domain Knowledge

### Ocean and Fluid Dynamics
- Incompressible/Boussinesq approximation for ocean flows
- Hydrostatic approximation for large-scale flows
- Free surface dynamics with implicit/explicit time stepping
- Coriolis effects and planetary rotation
- Stratification and buoyancy-driven flows
- Turbulence modeling via LES and eddy viscosity closures

### Numerical Methods
- Finite volume on structured grids (Arakawa C-grid)
- Staggered grid locations: velocities at cell faces, tracers at cell centers
- Various advection schemes: centered, upwind, WENO
- Pressure Poisson solver for incompressibility constraint
- Time stepping: RungeKutta, Adams-Bashforth, Quasi-Adams-Bashforth
- Take care of staggered grid location when writing operators or designing diagnostics

## Common Pitfalls

1. **Type Instability**: Especially in kernel functions - ruins GPU performance
2. **Overconstraining types**: Julia compiler can infer types. Type annotations should be used primarily for _multiple dispatch_, not for documentation.
3. **Forgetting Explicit Imports**: Tests will fail - add to using statements
4. **Using plain `julia` blocks in docstrings**: NEVER do this. ALWAYS use `jldoctest` blocks so examples are tested and verified to work. Plain `julia` blocks are not tested and will become stale.

### Fixing Bugs
- Subtle bugs often occur when a method is not imported, especially in extensions
- Sometimes user scripts expect names to be exported when they are not. Consider exporting the name
  (implementing the user interface that the user expects) rather than changing the user script.
- **Extending getproperty:** Never do this to fix a bug from accessing an undefined property.
  Fix the bug on the _caller_ side. A common source is when a property name is changed; update the calling function.
- **"Type is not callable" errors**: Variable names can conflict with function names in the same scope.
  This leads to errors like "Fields cannot be called". Solution: rename the variable or use a qualified
  function name (e.g., `Module.function_name`).
- **Connecting dots:** If a test fails immediately after a change, re-examine whether that change
  made sense. A quick fix that gets code to _run_ (fixing a test _error_) may make it _incorrect_
  (causing a test _failure_). Revisit the original edit for a more nuanced solution.

## Implementing Validation Cases / Reproducing Paper Results

When implementing a simulation from a published paper:

### 1. Parameter Extraction
- **Read the paper carefully** and extract ALL parameters: domain size, resolution, physical constants, 
  boundary conditions, initial conditions, forcing, closure parameters
- Look for parameter tables (often "Table 1" or similar)
- Check figure captions for additional details
- Note the coordinate system and conventions used

### 2. Geometry Verification (BEFORE running long simulations)
- **Always visualize the grid/domain geometry first**
- Check that:
  - Domain extents match the paper
  - Topography/immersed boundaries are correct
  - Coordinate orientations match (which direction is "downslope"?)
- Compare your geometry plot to figures in the paper

### 3. Initial Condition Verification
- After setting initial conditions, check:
  - `minimum(field)` and `maximum(field)` make physical sense
  - Spatial distribution looks correct (visualize if needed)
  - Dense water is where it should be, stratification is correct, etc.

### 4. Short Test Runs
Before running a long simulation:
- Run for a few timesteps on CPU at low resolution
- Verify:
  - No NaNs appear (check `maximum(abs, u)` etc.)
  - Flow is developing as expected (velocities increasing from zero)
  - Output files contain meaningful data
- Then test on GPU to catch GPU-specific issues

### 5. Progressive Validation
- Run a short simulation (e.g., 1 hour sim time) and visualize
- Check that the physics looks right:
  - Dense water flowing in the correct direction?
  - Velocities reasonable magnitude?
  - Mixing/entrainment happening where expected?
- Compare to early-time figures in the paper if available

### 6. Comparison to Paper Figures
- Create visualizations that match the paper's figure format
- Use the same colormaps, axis ranges, and time snapshots if possible
- Quantitative comparison: compute the same diagnostics as the paper

### 7. Common Issues
- **NaN blowups**: Usually from timestep too large, unstable initial conditions, 
  or if-else statements on GPU (use `ifelse` instead)
- **Nothing happening**: Check that buoyancy anomaly has the right sign, 
  that initial conditions are actually applied, that forcing is active
- **Wrong direction of flow**: Check coordinate conventions (is y increasing 
  upslope or downslope?)
- **GPU issues**: Avoid branching, ensure type stability, use `randn()` carefully


## Git Workflow
- Follow ColPrac (Collaborative Practices for Community Packages)
- Create feature branches for new work
- Write descriptive commit messages
- Update tests and documentation with code changes
- Check CI passes before merging

## Helpful Resources
- Oceananigans docs: https://clima.github.io/OceananigansDocumentation/stable/
- Discussions: https://github.com/CliMA/Oceananigans.jl/discussions
- KernelAbstractions.jl: https://github.com/JuliaGPU/KernelAbstractions.jl
- Reactant.jl: https://github.com/EnzymeAD/Reactant.jl
- Reactant.jl docs: https://enzymead.github.io/Reactant.jl/stable/
- Enzyme.jl: https://github.com/EnzymeAD/Enzyme.jl
- Enzyme.jl docs: https://enzyme.mit.edu/julia/dev
- YASGuide: https://github.com/jrevels/YASGuide
- ColPrac: https://github.com/SciML/ColPrac
- MCPRepl.jl: https://github.com/kahliburke/MCPRepl.jl

## When Unsure
1. Check existing examples in `examples/` directory
2. Look at similar implementations in the codebase
3. Review tests for usage patterns
4. Ask in GitHub discussions
5. Check documentation in `docs/src/`
6. Check validation cases in `validation/`

## AI Assistant Behavior
- Prioritize type stability and GPU compatibility
- Follow established patterns in existing code
- Add tests for new functionality
- Update exports in main module when adding public API
- Consider both CPU and GPU architectures
- Reference physics equations in comments when implementing dynamics
- Maintain consistency with the existing codebase style

## Interactive Julia REPL for AI Agents (MCPRepl.jl)

[MCPRepl.jl](https://github.com/kahliburke/MCPRepl.jl) exposes a Julia REPL via the Model Context Protocol (MCP),
allowing AI agents to execute Julia code, run tests, and iterate quickly during development.

### Installation

If MCPRepl.jl is not already installed, add it to your global Julia environment:

```julia
using Pkg
Pkg.activate()  # Activate global environment
Pkg.add(url="https://github.com/kahliburke/MCPRepl.jl")
```

Then run the security setup (one-time):

```julia
using MCPRepl
MCPRepl.quick_setup(:lax)  # For local development (localhost only, no API key)
```

### Starting the MCP Server

Before the AI agent can use the REPL, start the server in Julia:

```julia
using MCPRepl
MCPRepl.start_proxy(port=3000)  # Recommended: persistent proxy with dashboard
# OR
MCPRepl.start!(port=3000)       # Direct REPL backend
```

The dashboard is available at `http://localhost:3000/dashboard` when using the proxy.

### Cursor Configuration

Create `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "julia-repl": {
      "url": "http://localhost:3000",
      "transport": "http",
      "headers": {
        "X-MCPRepl-Target": "Oceananigans.jl"
      }
    }
  }
}
```

After creating this file, reload Cursor (Cmd+Shift+P → "Reload Window").

### Speeding Up Development with Revise.jl

For rapid iteration, use Revise.jl alongside MCPRepl. This allows code changes to be
reflected immediately without restarting Julia:

```julia
using Revise
using MCPRepl
using Oceananigans

MCPRepl.start_proxy(port=3000)
```

With this setup:
1. The AI agent can execute code via the REPL
2. Source code edits are automatically picked up by Revise
3. No need to restart Julia or re-import packages after editing source files
4. Tests can be run interactively with immediate feedback

### Available MCP Tools

Once connected, the AI agent has access to:
- **`julia_eval`** — Execute Julia code in the REPL
- **`lsp_goto_definition`** — Navigate to symbol definitions
- **`lsp_find_references`** — Find all usages of a symbol
- **`lsp_rename`** — Rename symbols across the codebase
- **`lsp_document_symbols`** — Get file structure/outline
- **`lsp_code_actions`** — Get available quick fixes

### Workflow Example

A typical development workflow:

1. Start Julia with Revise and MCPRepl
2. AI agent makes code changes via file editing
3. Revise automatically loads the changes
4. AI agent tests changes via MCPRepl without restarting
5. Iterate rapidly until the feature/fix is complete

This eliminates the slow compile-restart cycle and enables interactive debugging.

## Current Development Focus

Active areas of development to be aware of:

- **Performance optimization**: Reactant.jl integration for XLA compilation
- **Enzyme.jl integration**: Automatic differentiation support
- **Multi-architecture support**: AMD, Metal, OneAPI in addition to CUDA
- **MPI/distributed computing**: Improvements to distributed grid capabilities
- **Output improvements**: NetCDF and other output format enhancements
- **Grid flexibility**: Enhanced support for lat-lon, cubed sphere, immersed boundaries
- **Lagrangian particle tracking**: Enhanced particle capabilities
- **Validation**: Expanding the validation test suite
