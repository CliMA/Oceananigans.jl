# Library

Documenting the public user interface.

## Architectures
```@autodocs
Modules = [Oceananigans.Architectures]
Private = false
Pages   = ["Architectures.jl"]
```

## Boundary conditions
```@autodocs
Modules = [Oceananigans.BoundaryConditions]
Private = false
Pages   = [
    "BoundaryConditions/BoundaryConditions.jl",
    "BoundaryConditions/boundary_condition_types.jl",
    "BoundaryConditions/boundary_condition.jl",
    "BoundaryConditions/coordinate_boundary_conditions.jl",
    "BoundaryConditions/field_boundary_conditions.jl",
    "BoundaryConditions/solution_boundary_conditions.jl",
    "BoundaryConditions/model_boundary_conditions.jl",
    "BoundaryConditions/tracer_boundary_conditions.jl",
    "BoundaryConditions/tendency_boundary_conditions.jl",
    "BoundaryConditions/pressure_boundary_conditions.jl",
    "BoundaryConditions/boundary_function.jl",
    "BoundaryConditions/show_boundary_conditions.jl",
    "BoundaryConditions/zero_halo_regions.jl",
    "BoundaryConditions/fill_halo_regions.jl",
    "BoundaryConditions/apply_flux_bcs.jl",
    "BoundaryConditions/apply_value_gradient_bcs.jl",
    "BoundaryConditions/apply_flux_periodic_no_penetration_bcs.jl"
]
```

## Buoyancy
```@autodocs
Modules = [Oceananigans.Buoyancy]
Private = false
Pages   = [
    "Buoyancy/no_buoyancy.jl",
    "Buoyancy/buoyancy_tracer.jl",
    "Buoyancy/seawater_buoyancy.jl",
    "Buoyancy/Buoyancy.jl",
    "Buoyancy/linear_equation_of_state.jl",
    "Buoyancy/nonlinear_equation_of_state.jl",
    "Buoyancy/roquet_idealized_nonlinear_eos.jl",
    "Buoyancy/show_buoyancy.jl",
    "Buoyancy/buoyancy_utils.jl"
]
```

## Coriolis
```@autodocs
Modules = [Oceananigans.Coriolis]
Private = false
Pages   = [
    "Coriolis/Coriolis.jl",
    "Coriolis/no_rotation.jl",
    "Coriolis/f_plane.jl",
    "Coriolis/beta_plane.jl"
]
```

## Diagnostics
```@autodocs
Modules = [Oceananigans.Diagnostics]
Private = false
Pages   = [
    "Diagnostics/Diagnostics.jl",
    "Diagnostics/diagnostics_kernels.jl",
    "Diagnostics/horizontal_average.jl",
    "Diagnostics/timeseries.jl",
    "Diagnostics/cfl.jl",
    "Diagnostics/field_maximum.jl",
    "Diagnostics/nan_checker.jl"
]
```

## Fields
```@autodocs
Modules = [Oceananigans.Fields]
Private = false
Pages   = [
    "Fields/Fields.jl",
    "Fields/field.jl",
    "Fields/set!.jl",
    "Fields/show_fields.jl",
    "Fields/field_utils.jl"
]
```

## Forcing
```@autodocs
Modules = [Oceananigans.Forcing]
Private = false
Pages   = [
    "Forcing/Forcing.jl",
    "Forcing/simple_forcing.jl",
    "Forcing/model_forcing.jl"
]
```

## Grids
```@autodocs
Modules = [Oceananigans.Grids]
Private = false
Pages   = [
    "Grids/Grids.jl",
    "Grids/grid_utils.jl",
    "Grids/regular_cartesian_grid.jl"
]
```

## Models
```@autodocs
Modules = [Oceananigans.Models]
Private = false
Pages   = [
    "Models/Models.jl",
    "Models/clock.jl",
    "Models/model.jl",
    "Models/channel_model.jl",
    "Models/non_dimensional_model.jl",
    "Models/show_models.jl",
    "Models/model_utils.jl"
]
```

## Output writers
```@autodocs
Modules = [Oceananigans.OutputWriters]
Private = false
Pages   = [
    "OutputWriters/OutputWriters.jl",
    "OutputWriters/output_writer_utils.jl",
    "OutputWriters/jld2_output_writer.jl",
    "OutputWriters/netcdf_output_writer.jl",
    "OutputWriters/checkpointer.jl"
]
```

## Time steppers
```@autodocs
Modules = [Oceananigans.TimeSteppers]
Private = false
Pages   = [
    "TimeSteppers/TimeSteppers.jl",
    "TimeSteppers/kernels.jl",
    "TimeSteppers/adams_bashforth.jl"
]
```

## Simulations
```@autodocs
Modules = [Oceananigans.Simulations]
Private = false
Pages   = ["Simulations.jl"]
```

## Tubrulence closures
```@autodocs
Modules = [Oceananigans.TurbulenceClosures]
Private = false
Pages   = [
    "TurbulenceClosures/TurbulenceClosures.jl",
    "TurbulenceClosures/turbulence_closure_utils.jl",
    "TurbulenceClosures/closure_operators.jl",
    "TurbulenceClosures/viscous_dissipation_operators.jl",
    "TurbulenceClosures/diffusion_operators.jl",
    "TurbulenceClosures/velocity_tracer_gradients.jl",
    "TurbulenceClosures/closure_tuples.jl",
    "TurbulenceClosures/turbulence_closure_diagnostics.jl",
    "TurbulenceClosures/turbulence_closure_implementations/anisotropic_biharmonic_diffusivity.jl",
    "TurbulenceClosures/turbulence_closure_implementations/smagorinsky_lilly.jl",
    "TurbulenceClosures/turbulence_closure_implementations/constant_isotropic_diffusivity.jl",
    "TurbulenceClosures/turbulence_closure_implementations/verstappen_anisotropic_minimum_dissipation.jl",
    "TurbulenceClosures/turbulence_closure_implementations/blasius_smagorinsky.jl",
    "TurbulenceClosures/turbulence_closure_implementations/constant_anisotropic_diffusivity.jl",
    "TurbulenceClosures/turbulence_closure_implementations/rozema_anisotropic_minimum_dissipation.jl",
    "TurbulenceClosures/turbulence_closure_implementations/leith_enstrophy_diffusivity.jl"
]
```

## Utilities
```@autodocs
Modules = [Oceananigans.Utils]
Private = false
Pages   = [
    "Utils/Utils.jl",
    "Utils/adapt_structure.jl",
    "Utils/units.jl",
    "Utils/loop_macros.jl",
    "Utils/launch_config.jl",
    "Utils/pretty_time.jl",
    "Utils/pretty_filesize.jl",
    "Utils/time_step_wizard.jl",
    "Utils/tuple_utils.jl",
    "Utils/ordered_dict_show.jl",
    "Utils/cell_advection_timescale.jl",
    "Utils/output_writer_diagnostic_utils.jl",
    "Utils/with_tracers.jl"
]
```

## Abstract operations
```@autodocs
Modules = [Oceananigans, Oceananigans.AbstractOperations]
Private = false
Pages   = [
    "AbstractOperations/AbstractOperations.jl",
    "AbstractOperations/unary_operations.jl",
    "AbstractOperations/binary_operations.jl",
    "AbstractOperations/multiary_operations.jl",
    "AbstractOperations/derivatives.jl",
    "AbstractOperations/function_fields.jl",
    "AbstractOperations/computations.jl",
    "AbstractOperations/interpolation_utils.jl",
    "AbstractOperations/show_abstract_operations.jl",
    "AbstractOperations/grid_validation.jl"
]
```
