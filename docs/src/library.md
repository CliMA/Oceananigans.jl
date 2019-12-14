# Library

Documenting the public user interface.

## Boundary conditions
```@autodocs
Modules = [Oceananigans]
Private = false
Pages   = ["boundary_conditions.jl"]
```

## Buoyancy
```@autodocs
Modules = [Oceananigans]
Private = false
Pages   = ["buoyancy.jl"]
```

## Clock
```@autodocs
Modules = [Oceananigans]
Private = false
Pages   = ["clock.jl"]
```

## Coriolis
```@autodocs
Modules = [Oceananigans]
Private = false
Pages   = ["coriolis.jl"]
```

## Diagnostics
```@autodocs
Modules = [Oceananigans, Oceananigans.Diagnostics]
Private = false
Pages   = ["Diagnostics/Diagnostics.jl",
           "Diagnostics/diagnostics_kernels.jl",
           "Diagnostics/horizontal_average.jl",
           "Diagnostics/timeseries.jl",
           "Diagnostics/cfl.jl",
           "Diagnostics/field_maximum.jl",
           "Diagnostics/nan_checker.jl"]
```

## Fields
```@autodocs
Modules = [Oceananigans]
Private = false
Pages   = ["fields.jl"]
```

## Forcing
```@autodocs
Modules = [Oceananigans]
Private = false
Pages   = ["forcing.jl"]
```

## Grids
```@autodocs
Modules = [Oceananigans, Oceananigans.Grids]
Private = false
Pages   = ["Grids/Grids.jl",
           "Grids/grid_utils.jl",
           "Grids/regular_cartesian_grid.jl"]
```

## Models
```@autodocs
Modules = [Oceananigans]
Private = false
Pages   = ["models.jl"]
```

## Output writers
```@autodocs
Modules = [Oceananigans, Oceananigans.OutputWriters]
Private = false
Pages   = ["OutputWriters/OutputWriters.jl",
           "OutputWriters/output_writer_utils.jl",
           "OutputWriters/jld2_output_writer.jl",
           "OutputWriters/netcdf_output_writer.jl",
           "OutputWriters/checkpointer.jl"]
```

## Time steppers
```@autodocs
Modules = [Oceananigans, Oceananigans.TimeSteppers]
Private = false
Pages   = ["TimeSteppers/TimeSteppers.jl",
           "TimeSteppers/kernels.jl",
           "TimeSteppers/adams_bashforth.jl"]
```

## Tubrulence closures
```@autodocs
Modules = [Oceananigans, Oceananigans.TurbulenceClosures]
Private = false
Pages   = ["TurbulenceClosures/TurbulenceClosures.jl",
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
          "TurbulenceClosures/turbulence_closure_implementations/leith_enstrophy_diffusivity.jl"]
```

## Utilities
```@autodocs
Modules = [Oceananigans]
Private = false
Pages   = ["utils.jl"]
```
