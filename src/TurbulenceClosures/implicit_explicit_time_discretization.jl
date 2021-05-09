using KernelAbstractions: NoneEvent

using Oceananigans.Utils: arch_array
using Oceananigans.Grids: AbstractGrid

abstract type AbstractTimeDiscretization end

"""
    struct ExplicitTimeDiscretization <: AbstractTimeDiscretization end

Represents fully-explicit time discretization of a TurbulenceClosure.
"""
struct ExplicitTimeDiscretization <: AbstractTimeDiscretization end

"""
    struct VerticallyImplicitDiscretization <: AbstractTimeDiscretization end

Represents "vertically-implicit" time discretization of a TurbulenceClosure.

Currently this means that a flux divergence such as `∇ ⋅ q` will be time-discretized as

```
[∇ ⋅ q]ⁿ = [explicit_flux_divergence]ⁿ + [∂z (κ ∂z c)]ⁿ⁺¹,
```

at time step `n`.
"""
struct VerticallyImplicitTimeDiscretization <: AbstractTimeDiscretization end

time_discretization(::AbstractTurbulenceClosure{TimeDiscretization}) where TimeDiscretization = TimeDiscretization()

#####
##### ExplicitTimeDiscretization: move along, nothing to worry about here (use fallbacks).
#####

const AG = AbstractGrid
const ATD = AbstractTimeDiscretization

@inline diffusive_flux_x(i, j, k, grid::AG, ::ATD, args...) = diffusive_flux_x(i, j, k, grid, args...)
@inline diffusive_flux_y(i, j, k, grid::AG, ::ATD, args...) = diffusive_flux_y(i, j, k, grid, args...)
@inline diffusive_flux_z(i, j, k, grid::AG, ::ATD, args...) = diffusive_flux_z(i, j, k, grid, args...) 

@inline viscous_flux_ux(i, j, k, grid::AG, ::ATD, args...) = viscous_flux_ux(i, j, k, grid, args...)
@inline viscous_flux_uy(i, j, k, grid::AG, ::ATD, args...) = viscous_flux_uy(i, j, k, grid, args...)
@inline viscous_flux_uz(i, j, k, grid::AG, ::ATD, args...) = viscous_flux_uz(i, j, k, grid, args...)

@inline viscous_flux_vx(i, j, k, grid::AG, ::ATD, args...) = viscous_flux_vx(i, j, k, grid, args...)
@inline viscous_flux_vy(i, j, k, grid::AG, ::ATD, args...) = viscous_flux_vy(i, j, k, grid, args...)
@inline viscous_flux_vz(i, j, k, grid::AG, ::ATD, args...) = viscous_flux_vz(i, j, k, grid, args...)

@inline viscous_flux_wx(i, j, k, grid::AG, ::ATD, args...) = viscous_flux_wx(i, j, k, grid, args...)
@inline viscous_flux_wy(i, j, k, grid::AG, ::ATD, args...) = viscous_flux_wy(i, j, k, grid, args...)
@inline viscous_flux_wz(i, j, k, grid::AG, ::ATD, args...) = viscous_flux_wz(i, j, k, grid, args...)
