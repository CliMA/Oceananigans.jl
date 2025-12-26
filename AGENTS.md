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
   - Short-circuiting if-statements should be avoided if possible. This includes
     `if`... `else`, as well as the ternary operator `?` ... `:`. The function `ifelse` should be used for logic instead.
   - Do not put error messages inside kernels.
   - Models _never_ go inside kernels
   - Mark functions inside kernels with `@inline`.
   
4. **Documentation**:
   - Use DocStringExtensions.jl for consistent docstrings
   - Include `$(SIGNATURES)` for automatic signature documentation
   - Add examples in docstrings when helpful

5. **Memory leanness**
   - Favor doing computations inline versus allocating temporary memory
   - Generally minimize memory allocation
   - If an implementation is awkward, don't hesitate to suggest an upstream feature
     that will make something easier, rather than forcing in low quality code.

### Oceananigans ecosystem best practices

1. **General coding style**
  - Consult the Notation section in the docs (`docs/src/appendix/notation.md`) for variable names
  - Variables may take a "symbolic form" (often unicode symbols, useful when used in equations) or "English form" (more descriptive and self-explanatory). Use math and English consistently and try not to mix the two in expressions for clarity.
  - For keyword arguments, we like
    * "No space form" for inline functions: `f(x=1, y=2)`,
    * "Single space form for multiline representations:
    ```
    long_function(a = 1,
                  b = 2)
    ```
    * Variables should be declared `const` _only when necessary_, and not otherwise. This helps interpret the meaning and usage of variables. Do not overuse `const`.
  - `TitleCase` style is reserved for types, type aliases, and constructors.
  - `snake_case` style should be used for functions and variables (instances of types)
  - "Number variables" (`Nx`, `Ny`) should start with capital `N`. For number of time steps use `Nt`.
    Spatial indices are `i, j, k` and time index is `n`. 

2. **Import style**
  - Use different style for source code versus user scripts:
    * in source code, explicitly import all names into files
    * in scripts, follow the user interface by writing "using Oceananigans".
    * only use explicit import in scripts for names that are _not_ exported by the top-level file Oceananigans.jl
    * sometimes we need to write `using Oceananigans.Units`

3. **Examples and integration tests**
  - Explain at the top of the file what a simulation is doing
  - Let code "speak for itself" as much as possible, to keep an explanation concise.
    In other words, use a Literate style.
  - Use a lighthearted, funny, engaging, style for example prose.
  - Use visualization interspersed with model setup or simulation running when needed to
    give an understanding of a complex grid, initial condition, or other model property.
  - Look at previous examples. New examples should add as much value as possible while remaining simple. This requires judiciously introducing new features and doing creative and surprising things with simulations that will spark readers' imagination.
  - Follow the style of existing examples, not the source code
  - Remember that initial condition functions act _pointwise_, there should be no broadcasting inside an initial condition function
  - Do not convert between units. Always keep the units the same for calculations, unless plotting coordinates into the functions.
  - If possible, avoid long underscore names. Use concise evocative names like `z = znodes(grid, Center())`.
  - Use unicode that is consistent with the source code. Do not be afraid of unicode for intermediate variables.
  - Make sure that all notation in examples is consistent with `docs/src/appendix/notation.md`
  - Always add axis labels and colorbars to simulations.
  - Check previous examples and strive to make new examples that add new physics and new value relative to old examples. Don't just copy old examples.
  - `@allowscalar` should very sparingly be used or never in an example. If you need to, make a suggestion to change the source code so that `@allowscalar` is not needed.
  - The examples should use exported names primarily. If an example needs an excessive amount of internal names, those names should be exported or a new abstraction needs to be developed.
  - For `discrete_form=true` forcing and boundary conditions, always use `xnode`, `ynode`, and `znode` from Oceananigans. _Never_ access grid metrics manually.
  - Use `Oceananigans.defaults.FloatType = FT` to change the precision; do not set precision within constructors manually.
  - Use integers when values are integers. Do not "eagerly convert" to Float64 by adding ".0" to integers.
  - Constructors should convert to `FT` under the hood, and it should be not be necessary to "manually convert" numbers to `FT`. In other words, we should not see `FT(1)` appearing very often,
    unless _absolutely_ necessary.
  - Keyword arguments that expect tuples (eg `tracers = (:a, :b)`) often "autotuple" single arguments. Always rely on this: i.e. use `tracers = :c` instead of `tracers = (:c,)` (the latter is more prone to mistakes and harder to read)
  - Instances of `NonhydrostaticModel`, `HydrostaticFreeSurfaceModel`, etc. are almost always called `model`
    and instances of `Simulation` are called `simulation`.

4. **Documentation Style**
  - Mathematical notation in `docs/src/appendix/notation.md`
  - Use Documenter.jl syntax for cross-references
  - Include code examples in documentation pages
  - Add references to papers from the literature by adding bibtex to `oceananigans.bib`, and then
    a corresponding citation
  - Make use of cross-references with equations

5. **Common misconceptions**
  - Fields and AbstractOperations can be used in `set!`.

### Naming Conventions
- **Files**: snake_case (e.g., `nonhydrostatic_model.jl`, `compute_hydrostatic_free_surface_tendencies.jl`)
- **Types**: PascalCase (e.g., `NonhydrostaticModel`, `HydrostaticFreeSurfaceModel`, `SeawaterBuoyancy`)
- **Functions**: snake_case (e.g., `time_step!`, `compute_tendencies!`)
- **Kernels**: "Kernels" (functions prefixed with `@kernel`) may be prefixed with an underscore (e.g., `_compute_tendency_kernel`)
- **Variables**: Use _either_ an English long name, or mathematical notation with readable unicode. Variable names should be taken from `docs/src/appendix/notation.md` in the docs. If a new variable is created (or if one doesn't exist), it should be added to the table in notation.md

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

### Fixing bugs
- Subtle bugs often occur when a method is not imported, especially in an extension
- Sometimes user scripts are written expecting names to be exported, when they are not. In that case
  consider exporting the name automatically (ie implement the user interface that the user expects) rather
  than changing the user script
- **Extending getproperty:** never do this to fix a bug associated with accessing an undefined property.
  This bug should be fixed on the _caller_ side, so that an undefined name is not accessed.
  A common source of this bug is when a property name is changed (for example, to make it clearer).
  In this case the calling function merely needs to be updated.
- **"Type is not callable" errors**: Variable naming is hard. Sometimes, variable names conflict. A common issue is when the name of a _field_ (the result
  of a computation) overlaps with the name of a function in the same scope/context. This can lead to errors like "Fields cannot be called".
  The solution to this problem is to change the name of the field to be more verbose, or use a qualified name for the function
  that references the module it is defined in to disambiguate the names (if possible).
- **Connecting dots:** If a test fails immediately after a change was made, go back and re-examine whether that change
  made sense. Sometimes, a simple fix that gets code to _run_ (ie fixing a test _error_) will end up making it _incorrect_ (which hopefully will be caught as a test _failure_). In this case the original edit should be revisited: a more nuanced solution to the test error may be required.

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

### Testing docs
- Consider manually running `@example` blocks, rather than building the whole
  documentation to find errors.
- Unless explicitly asked, do not write `for` loops in docs blocks. Use built-in functions
  (which will launch kernels under the hood) instead.
- Be conservative about developing examples and tutorials. Do not write extensive example code unless asked.
  Instead, produce skeletons or outlines with minimum viable code.

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
