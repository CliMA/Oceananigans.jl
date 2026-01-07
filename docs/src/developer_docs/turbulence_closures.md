# Implementing Turbulence Closures

This guide walks through how to implement a new turbulence closure in Oceananigans.
As an example, we'll examine the implementation of `PacanowskiPhilanderVerticalDiffusivity`,
a Richardson number-based vertical mixing parameterization from [PacanowskiPhilander81](@citet).

## Overview of the turbulence closure system

Turbulence closures in Oceananigans contribute diffusive flux divergences to the
momentum and tracer tendency equations. The closure system is designed around:

1. **Abstract types** that define the interface and dispatch
2. **Diffusivity/viscosity computation** that occurs before tendency computation
3. **Flux functions** that compute diffusive fluxes at grid faces
4. **Time discretization** that can be explicit or vertically implicit

### The type hierarchy

All turbulence closures inherit from `AbstractTurbulenceClosure{TimeDiscretization, RequiredHalo}`:

```julia
abstract type AbstractTurbulenceClosure{TimeDiscretization, RequiredHalo} end
```

The type parameters are:

- `TimeDiscretization`: Either `ExplicitTimeDiscretization` or `VerticallyImplicitTimeDiscretization`
- `RequiredHalo`: An integer specifying the minimum halo size needed (typically 1 or 2)

For closures with scalar diffusivities, there's a more specialized abstract type:

```julia
abstract type AbstractScalarDiffusivity{TD, F, N} <: AbstractTurbulenceClosure{TD, N} end
```

where `F` is the "formulation" (e.g., `VerticalFormulation`, `HorizontalFormulation`,
`ThreeDimensionalFormulation`).

## Step-by-step implementation

### Step 1: Define the struct

A turbulence closure struct holds its parameters. For `PacanowskiPhilanderVerticalDiffusivity`:

```julia
struct PacanowskiPhilanderVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 1}
    ν₀ :: FT  # Background viscosity
    ν₁ :: FT  # Shear-driven viscosity coefficient
    κ₀ :: FT  # Background diffusivity
    c  :: FT  # Richardson number scaling coefficient
    n  :: FT  # Exponent for viscosity
    maximum_diffusivity :: FT
    maximum_viscosity :: FT
end
```

Key points:

- **Type parameters**: `TD` is the time discretization, `FT` is the float type
- **Supertype**: `AbstractScalarDiffusivity{TD, VerticalFormulation, 1}` indicates:
  - This closure uses scalar (not tensor) diffusivities
  - It only acts in the vertical direction (`VerticalFormulation`)
  - It requires a halo size of 1
- **All fields are concretely typed**: This is essential for type stability and GPU performance

### Step 2: Create the constructor

Provide a user-friendly constructor with default values:

```julia
function PacanowskiPhilanderVerticalDiffusivity(
        time_discretization = VerticallyImplicitTimeDiscretization(),
        FT = Float64;
        ν₀ = 1e-4,
        ν₁ = 1e-2,
        κ₀ = 1e-5,
        c  = 5.0,
        n  = 2.0,
        maximum_diffusivity = Inf,
        maximum_viscosity = Inf)

    TD = typeof(time_discretization)

    return PacanowskiPhilanderVerticalDiffusivity{TD}(
        convert(FT, ν₀),
        convert(FT, ν₁),
        convert(FT, κ₀),
        convert(FT, c),
        convert(FT, n),
        convert(FT, maximum_diffusivity),
        convert(FT, maximum_viscosity))
end
```

Important conventions:

- The first positional argument is `time_discretization`
- The second positional argument is the float type `FT`
- All physics parameters are keyword arguments
- **Always `convert` to `FT`** to ensure type consistency

### Step 3: Define type aliases

Type aliases make dispatch cleaner and support "closure ensembles" (arrays of closures
for sensitivity studies or ensemble simulations):

```julia
const PPVD = PacanowskiPhilanderVerticalDiffusivity
const PPVDArray = AbstractArray{<:PPVD}
const FlavorOfPPVD = Union{PPVD, PPVDArray}
```

### Step 4: Specify diffusivity locations

Diffusivities live at specific grid locations. For vertical diffusivities that multiply
vertical gradients, they should be at `(Center, Center, Face)`:

```julia
@inline viscosity_location(::FlavorOfPPVD)   = (Center(), Center(), Face())
@inline diffusivity_location(::FlavorOfPPVD) = (Center(), Center(), Face())
```

### Step 5: Define diffusivity accessors

The closure system needs to know how to extract viscosity and diffusivity from
the precomputed `diffusivities` NamedTuple:

```julia
@inline viscosity(::FlavorOfPPVD, diffusivities) = diffusivities.νz
@inline diffusivity(::FlavorOfPPVD, diffusivities, id) = diffusivities.κz
```

The `id` argument is the tracer index—for closures with tracer-specific diffusivities,
this can be used to return different fields for different tracers.

### Step 6: Build closure fields

Closures that precompute diffusivities need storage fields. Define `build_closure_fields`
to create them:

```julia
function build_closure_fields(grid, clock, tracer_names, bcs, closure::FlavorOfPPVD)
    κz = Field{Center, Center, Face}(grid)
    νz = Field{Center, Center, Face}(grid)
    return (; κz, νz)
end
```

The returned `NamedTuple` becomes `model.closure_fields` and is passed to
`compute_diffusivities!` and flux functions.

### Step 7: Implement diffusivity computation

The heart of the closure is `compute_diffusivities!`, which updates the precomputed
diffusivity fields based on the current model state:

```julia
function compute_diffusivities!(diffusivities, closure::FlavorOfPPVD, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    tracers = buoyancy_tracers(model)
    buoyancy = buoyancy_force(model)
    velocities = model.velocities

    launch!(arch, grid, parameters,
            compute_pacanowski_philander_diffusivities!,
            diffusivities, grid, closure, velocities, tracers, buoyancy)

    return nothing
end
```

The kernel does the actual computation:

```julia
@kernel function compute_pacanowski_philander_diffusivities!(diffusivities, grid, closure,
                                                             velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Support closure ensembles
    closure_ij = getclosure(i, j, closure)

    # Compute Richardson number
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, buoyancy, tracers)

    # Extract parameters
    ν₀ = closure_ij.ν₀
    ν₁ = closure_ij.ν₁
    κ₀ = closure_ij.κ₀
    cc = closure_ij.c
    n  = closure_ij.n

    # Pacanowski-Philander formulas
    denominator = 1 + cc * Ri
    νz = ν₀ + ν₁ / denominator^n
    κz = κ₀ + ν₁ / denominator^(n + 1)

    # Apply maximum limits
    νz = min(νz, closure_ij.maximum_viscosity)
    κz = min(κz, closure_ij.maximum_diffusivity)

    @inbounds diffusivities.νz[i, j, k] = νz
    @inbounds diffusivities.κz[i, j, k] = κz
end
```

**GPU compatibility rules for kernels:**

- Use `@kernel` from KernelAbstractions.jl
- Use `@index(Global, NTuple)` to get indices
- Use `@inbounds` for array access
- **Never use `if`/`else` with different types**—use `ifelse` instead
- **Never throw errors**—GPU kernels cannot print or throw
- Avoid allocations

### Step 8: Implement `show` methods

Good display methods help users understand their closures:

```julia
Base.summary(closure::PPVD{TD}) where TD = 
    string("PacanowskiPhilanderVerticalDiffusivity{$TD}")

function Base.show(io::IO, closure::PPVD)
    print(io, summary(closure), ":\n")
    print(io, "├── ν₀: ", prettysummary(closure.ν₀), '\n')
    print(io, "├── ν₁: ", prettysummary(closure.ν₁), '\n')
    print(io, "├── κ₀: ", prettysummary(closure.κ₀), '\n')
    print(io, "├── c: ", prettysummary(closure.c), '\n')
    print(io, "└── n: ", prettysummary(closure.n))
end
```

### Step 9: Add exports

Add the new closure to the exports in `TurbulenceClosures.jl`:

```julia
export
    # ... other exports ...
    PacanowskiPhilanderVerticalDiffusivity,
    # ...
```

And include the source file:

```julia
include("turbulence_closure_implementations/pacanowski_philander_vertical_diffusivity.jl")
```

If the closure should be part of the public API, also add it to the exports in
`src/Oceananigans.jl`.

## The flux computation pipeline

Understanding how diffusivities become tendencies helps in debugging:

1. **`compute_diffusivities!`** is called during `update_state!`
2. The precomputed diffusivities are stored in `model.closure_fields`
3. During tendency computation, **flux functions** are called:
   - `diffusive_flux_x`, `diffusive_flux_y`, `diffusive_flux_z` for tracers
   - `viscous_flux_ux`, `viscous_flux_vy`, etc. for momentum
4. For `AbstractScalarDiffusivity` with standard formulations, these flux functions
   are **already implemented**—you typically don't need to write them

The default flux functions use `viscosity()` and `diffusivity()` accessors along with
`viscosity_location()` and `diffusivity_location()` to interpolate diffusivities to
the correct grid locations.

## Time discretization

Oceananigans supports two time discretizations for diffusive terms:

### Explicit time discretization

All diffusive fluxes are computed explicitly:

```julia
closure = PacanowskiPhilanderVerticalDiffusivity(ExplicitTimeDiscretization())
```

This is simple but has a strict CFL constraint based on diffusivity.

### Vertically implicit time discretization

Vertical diffusion is treated implicitly, which is more stable for large diffusivities:

```julia
closure = PacanowskiPhilanderVerticalDiffusivity(VerticallyImplicitTimeDiscretization())
```

This is the default and recommended for most oceanographic applications. The implicit
solver uses a tridiagonal matrix algorithm.

## Advanced features

### Closures that require extra tracers

Some closures (like `CATKEVerticalDiffusivity`) require prognostic equations for
additional quantities (like TKE). Implement:

```julia
closure_required_tracers(::MyClosure) = (:e,)  # requires tracer named :e
```

### Closures that modify boundary conditions

Some closures need special boundary conditions. Implement:

```julia
add_closure_specific_boundary_conditions(closure::MyClosure, bcs, args...) = modified_bcs
```

### Custom flux functions

For non-standard flux formulations (like tensor diffusivities for isopycnal mixing),
you can override the flux functions directly. See `IsopycnalSkewSymmetricDiffusivity`
for an example.

## Testing your closure

Create tests that verify:

1. **Construction**: The closure constructs with default and custom parameters
2. **Type stability**: Use `@code_warntype` on critical functions
3. **GPU compatibility**: Run on GPU to catch dynamic dispatch issues
4. **Physical behavior**: Test that diffusivities respond correctly to flow conditions
5. **Conservation**: Verify that the closure doesn't create or destroy tracer mass

Example test structure:

```julia
@testset "PacanowskiPhilanderVerticalDiffusivity" begin
    closure = PacanowskiPhilanderVerticalDiffusivity()
    
    grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid; closure, buoyancy=BuoyancyTracer(), tracers=:b)
    
    # Test that model constructs
    @test model isa NonhydrostaticModel
    
    # Test time stepping
    time_step!(model, 1)
    @test model.clock.time == 1
end
```

## Summary

To implement a new turbulence closure:

1. **Define a struct** inheriting from an appropriate abstract type
2. **Create a constructor** with sensible defaults
3. **Define type aliases** for dispatch flexibility
4. **Specify locations** with `viscosity_location` and `diffusivity_location`
5. **Define accessors** with `viscosity` and `diffusivity`
6. **Build fields** with `build_closure_fields`
7. **Compute diffusivities** with `compute_diffusivities!`
8. **Add display methods** with `summary` and `show`
9. **Export** from `TurbulenceClosures.jl` and optionally `Oceananigans.jl`
10. **Write tests** to verify correctness

The source code for `PacanowskiPhilanderVerticalDiffusivity` in
`src/TurbulenceClosures/turbulence_closure_implementations/pacanowski_philander_vertical_diffusivity.jl`
serves as a complete, working example of these principles.

