# [Scans: Reductions and Accumulations](@id scans_tutorial)

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

Scans traverse one or more dimensions of a [`Field`](@ref) to either *reduce* its size or *accumulate* information along the traversal direction.
Oceananigans augments Julia's reductions (such as `sum`, `maximum`, and `cumsum`) with scan objects that are aware of grid metrics, staggering, and conditional masks.
This tutorial walks through the most common scans:

- [`Average`](@ref) for metric-aware means,
- [`Integral`](@ref) for discrete spatial integrals,
- `maximum` and `minimum` reductions, and
- [`CumulativeIntegral`](@ref) for directional sums of cell volumes.

We'll focus on how to construct reductions of both fields and operations, how to pick traversal dimensions, and how to use the resulting scans in diagnostics.

## A compact sample field

We start with a small rectilinear grid and define a tracer-like field with gentle three-dimensional structure.
The horizontal variation is periodic while the vertical variation decays exponentially, so the field has non-zero means, integrals, and extrema.

```@setup scans
Nx, Ny, Nz = 4, 4, 4

grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       x = (0, 2π),
                       y = (0, 2π),
                       z = (-1, 0),
                       topology = (Periodic, Periodic, Bounded))

c = CenterField(grid)
set!(c) do x, y, z
    0.6 + 0.3sin(x / 2) + 0.2cos(y) + 0.4exp(z)
end
```

The next sections reuse this grid and field.

## Metric-aware averages

[`Average`](@ref) returns a `Reduction` object that stores how to traverse the requested dimensions, including the correct metric factors for irregular grids.
Nothing is computed until the reduction is wrapped in `Field` and then evaluated with [`compute!`](@ref Oceananigans.Fields.compute!).

```@example scans
horizontal_mean = Average(c; dims=(1, 2))
summary(horizontal_mean)
```

Wrapping the reduction in `Field` produces a concrete object whose size reflects the dimensions that were not reduced.
Because we collapse `x` and `y`, the result contains a vertical profile defined on the original `z`-levels.

```@example scans
horizontal_profile = Field(horizontal_mean)
compute!(horizontal_profile)
size(horizontal_profile), location(horizontal_profile)
```

Values can be queried like any other `Field`.

```@example scans
horizontal_profile[1, 1, 1], horizontal_profile[1, 1, end]
```

As with any operation, we can average derived quantities without allocating intermediates.
Here we compute the domain-mean kinetic energy density associated with two velocity components based on `c`.

```@example scans
u = c
w = 0.8c + 0.2
kinetic_energy = 0.5 * (u^2 + w^2)

mean_ke = Field(Average(kinetic_energy; dims=(1, 2, 3)))
compute!(mean_ke)
mean_ke[1, 1, 1]
```

### Conditional averages

All scans accept a `condition` keyword that masks out grid points where the condition is `false`.
Below we average only the portions of `c` that exceed the horizontal mean at each level.

```@example scans
threshold = Field(horizontal_mean)
above_threshold = Field(Average(c; dims=(1, 2), condition=c .> threshold))
compute!(above_threshold)
above_threshold[1, 1, 1], above_threshold[1, 1, end]
```

The `condition` may be a boolean field, an operation, or a plain array, so it is straightforward to tailor averages to immersed boundary masks or subregions.

## Integrals and total quantities

[`Integral`](@ref) multiplies the operand by the appropriate grid measures before summing.
This makes it ideal for diagnosing total mass, tracer content, or fluxes.

```@example scans
volume_integral = Field(Integral(c; dims=(1, 2, 3)))
compute!(volume_integral)
volume_integral[1, 1, 1]
```

Integrals naturally return scalars, but they can also yield reduced profiles when only some dimensions are traversed.
The following integrates over ``x`` and ``y`` to produce a bar-integral (horizontal integral) of `c` as a function of depth.

```@example scans
column_integral = Field(Integral(c; dims=(1, 2)))
compute!(column_integral)
size(column_integral), column_integral[1, 1, end]
```

Because integrals are operations, they can be embedded inside diagnostics with no additional allocations.

```@example scans
energy_integral = Field(Integral(kinetic_energy; dims=(1, 2, 3)))
compute!(energy_integral)
energy_integral[1, 1, 1]
```

## Maximum and minimum scans

Standard Julia reductions work on fields directly. Supplying a `dims` argument returns a `ReducedField` that can be wrapped in `Field` when you need a persistent object.

```@example scans
max_over_z = Field(Reduction(maximum!, c, dims=3))
min_over_z = Field(Reduction(minimum!, c, dims=3))
compute!((max_over_z, min_over_z))
max_over_z[1, 1, 1], min_over_z[1, 1, 1]
```

The same pattern applies to operations.
Here we compute the extrema of a vertical derivative, demonstrating that scans respect the staggering of their operands.

```@example scans
∂z_c = ∂z(c)
max_vertical_gradient = Field(Reduction(maximum!, abs(∂z_c), dims=(1, 2, 3)))
compute!(max_vertical_gradient)
max_vertical_gradient[1, 1, 1]
```

## Cumulative integrals

[`CumulativeIntegral`](@ref) walks along one dimension and records the integral from the start up to each location.
This is useful for computing barotropic streamfunctions or vertically integrated fluxes.

```@example scans
vertical_cumulative = Field(CumulativeIntegral(c; dims=3))
compute!(vertical_cumulative)
vertical_cumulative[1, 1, 1], vertical_cumulative[1, 1, end]
```

The `reverse=true` keyword flips the direction of accumulation so that each level reports the integral from the opposite boundary.

```@example scans
reverse_vertical = Field(CumulativeIntegral(c; dims=3, reverse=true))
compute!(reverse_vertical)
reverse_vertical[1, 1, 1], reverse_vertical[1, 1, end]
```

## Where scans show up in practice

Scans are building blocks for diagnostics and model output. They can be stored as `ComputedField`s, supplied to `OutputWriter`s, or used inside callbacks.
Because scans only allocate when first wrapped in `Field`, they are light-weight additions to production simulations.

- Use `Field(Average(...))` for running means, flux decompositions, or horizontally averaged profiles.
- Use `Field(Integral(...))` for conservation checks, domain budgets, or net transports across boundaries.
- Combine conditional scans with immersed boundaries or tracer thresholds to zoom in on dynamically active regions.
- `CumulativeIntegral` yields integrated transports, while `maximum` and `minimum` reductions expose the extrema of derived fields.

To see scans embedded in larger workflows, explore the diagnostics chapter and the example scripts generated in `docs/src/literated`.
