using KernelAbstractions: NoneEvent

using Oceananigans.Utils: arch_array
using Oceananigans.Grids: AbstractGrid

abstract type AbstractTimeDiscretization end

"""
    struct Explicit <: AbstractTimeDiscretization

Represents fully-explicit time-discretization of a `TurbulenceClosure`.
"""
struct Explicit <: AbstractTimeDiscretization end

Base.summary(::Explicit) = "Explicit"

"""
    struct VerticallyImplicit <: AbstractTimeDiscretization

Represents "vertically-implicit" time-discretization of a `TurbulenceClosure`.

This imples that a flux divergence such as `∇ ⋅ q` at the n-th timestep is 
time-discretized as

```julia
[∇ ⋅ q]ⁿ = [explicit_flux_divergence]ⁿ + [∂z (κ ∂z c)]ⁿ⁺¹
```
"""
struct VerticallyImplicit <: AbstractTimeDiscretization end

Base.summary(::VerticallyImplicit) = "VerticallyImplicit"

@inline time_discretization(::AbstractTurbulenceClosure{TimeDiscretization}) where TimeDiscretization = TimeDiscretization()
@inline time_discretization(::Nothing) = Explicit() # placeholder for closure::Nothing

#####
##### Explicit: move along, nothing to worry about here (use fallbacks).
#####

const ATD = AbstractTimeDiscretization

@inline diffusive_flux_x(i, j, k, grid, ::ATD, args...) = diffusive_flux_x(i, j, k, grid, args...)
@inline diffusive_flux_y(i, j, k, grid, ::ATD, args...) = diffusive_flux_y(i, j, k, grid, args...)
@inline diffusive_flux_z(i, j, k, grid, ::ATD, args...) = diffusive_flux_z(i, j, k, grid, args...) 

@inline viscous_flux_ux(i, j, k, grid, ::ATD, args...) = viscous_flux_ux(i, j, k, grid, args...)
@inline viscous_flux_uy(i, j, k, grid, ::ATD, args...) = viscous_flux_uy(i, j, k, grid, args...)
@inline viscous_flux_uz(i, j, k, grid, ::ATD, args...) = viscous_flux_uz(i, j, k, grid, args...)

@inline viscous_flux_vx(i, j, k, grid, ::ATD, args...) = viscous_flux_vx(i, j, k, grid, args...)
@inline viscous_flux_vy(i, j, k, grid, ::ATD, args...) = viscous_flux_vy(i, j, k, grid, args...)
@inline viscous_flux_vz(i, j, k, grid, ::ATD, args...) = viscous_flux_vz(i, j, k, grid, args...)

@inline viscous_flux_wx(i, j, k, grid, ::ATD, args...) = viscous_flux_wx(i, j, k, grid, args...)
@inline viscous_flux_wy(i, j, k, grid, ::ATD, args...) = viscous_flux_wy(i, j, k, grid, args...)
@inline viscous_flux_wz(i, j, k, grid, ::ATD, args...) = viscous_flux_wz(i, j, k, grid, args...)
