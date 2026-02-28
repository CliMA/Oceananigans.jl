# Important Files to Know

## Core Implementation

- `src/Oceananigans.jl` - Main module, all exports
- `src/Models/NonhydrostaticModels/nonhydrostatic_model.jl` - Nonhydrostatic model definition
- `src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl` - Hydrostatic model
- `src/Grids/` - Grid implementations
- `src/Fields/` - Field types and operations
- `src/TimeSteppers/` - Time integration schemes

## Configuration

- `Project.toml` - Package dependencies and compat bounds
- `test/runtests.jl` - Test configuration

## Examples

- `examples/langmuir_turbulence.jl` - Ocean mixed layer with Langmuir turbulence
- `examples/internal_wave.jl` - Internal wave propagation
- `examples/shallow_water_Bickley_jet.jl` - Shallow water instability
- `examples/baroclinic_adjustment.jl` - Baroclinic instability
- `examples/two_dimensional_turbulence.jl` - 2D turbulence
- Many more examples in the `examples/` directory
