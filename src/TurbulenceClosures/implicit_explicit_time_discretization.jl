using KernelAbstractions: NoneEvent

using Oceananigans.Utils: arch_array
using Oceananigans.Grids: AbstractGrid

abstract type AbstractTimeDiscretization end

struct ExplicitDiscretization <: AbstractTimeDiscretization end

struct VerticallyImplicitDiscretization <: AbstractTimeDiscretization end

time_discretization(closure) = ExplicitDiscretization() # fallback

#####
##### Time discretization for diffusive fluxes
#####

@inline diffusive_flux_x(i, j, k, grid::AbstractGrid, time_discretization, args...) = diffusive_flux_x(i, j, k, grid, args...)
@inline diffusive_flux_y(i, j, k, grid::AbstractGrid, time_discretization, args...) = diffusive_flux_y(i, j, k, grid, args...)
@inline diffusive_flux_z(i, j, k, grid::AbstractGrid, time_discretization, args...) = diffusive_flux_z(i, j, k, grid, args...) 

# Elide explicit verticaly diffusive flux
@inline diffusive_flux_z(i, j, k, grid::AbstractGrid{FT}, ::VerticallyImplicitDiscretization, args...) where FT = zero(FT)

#####
##### Time discretization for viscous fluxes
#####

@inline viscous_flux_ux(i, j, k, grid::AbstractGrid, time_discretization, args...) = viscous_flux_ux(i, j, k, grid, args...)
@inline viscous_flux_uy(i, j, k, grid::AbstractGrid, time_discretization, args...) = viscous_flux_uy(i, j, k, grid, args...)
@inline viscous_flux_uz(i, j, k, grid::AbstractGrid, time_discretization, args...) = viscous_flux_uz(i, j, k, grid, args...)

@inline viscous_flux_vx(i, j, k, grid::AbstractGrid, time_discretization, args...) = viscous_flux_vx(i, j, k, grid, args...)
@inline viscous_flux_vy(i, j, k, grid::AbstractGrid, time_discretization, args...) = viscous_flux_vy(i, j, k, grid, args...)
@inline viscous_flux_vz(i, j, k, grid::AbstractGrid, time_discretization, args...) = viscous_flux_vz(i, j, k, grid, args...)

@inline viscous_flux_wx(i, j, k, grid::AbstractGrid, time_discretization, args...) = viscous_flux_wx(i, j, k, grid, args...)
@inline viscous_flux_wy(i, j, k, grid::AbstractGrid, time_discretization, args...) = viscous_flux_wy(i, j, k, grid, args...)
@inline viscous_flux_wz(i, j, k, grid::AbstractGrid, time_discretization, args...) = viscous_flux_wz(i, j, k, grid, args...)

# Elide explicit viscous fluxes
@inline viscous_flux_uz(i, j, k, grid::AbstractGrid{FT}, ::VerticallyImplicitDiscretization, args...) where FT = zero(FT)
@inline viscous_flux_vz(i, j, k, grid::AbstractGrid{FT}, ::VerticallyImplicitDiscretization, args...) where FT = zero(FT)
@inline viscous_flux_wz(i, j, k, grid::AbstractGrid{FT}, ::VerticallyImplicitDiscretization, args...) where FT = zero(FT)

#####
##### Vertically implicit solver
#####

implicit_velocity_step!(u, ::Nothing, model; kwargs...) = NoneEvent()
implicit_tracer_step!(c, ::Nothing, model; kwargs...) = NoneEvent()

implicit_diffusion_solver(closure, args...) = implicit_solver(time_discretization(closure), closure, args...)

implicit_diffusion_solver(::ExplicitDiscretization, args...) = nothing

@inline function vertical_diffusion_upper_diagonal(i, j, k, grid, closure, diffusivities, Δt)
    κ = κᶜᶜᶠ(i, j, k, grid, diffusivity(i, j, k, grid, closure, diffusivities))
    return Δt / (Δzᵃᵃᶜ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid))
end

@inline function vertical_diffusion_lower_diagonal(i, j, k, grid, closure, diffusivities, Δt)
    κ = κᶜᶜᶠ(i, j, k, grid, diffusivity(i, j, k, grid, closure, diffusivities))
    return Δt / (Δzᵃᵃᶜ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid))
end

@inline function vertical_diffusion_diagonal(i, j, k, grid, closure, diffusivities, Δt)
    κ = κᶜᶜᶠ(i, j, k, grid, diffusivity(i, j, k, grid, closure, diffusivities))
    return Δt * (  1 / (Δzᵃᵃᶜ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid)
                        + 1 / (Δzᵃᵃᶜ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid)))
end

"""
    implicit_diffusion_solver(::VerticallyImplicitDiscretization, closure, diffusivities, arch, grid)

Build a tridiagonal solver for the elliptic equation

```math
(1 + Δt ∂z κ ∂z) cⁿ⁺¹ = c★
```

where `cⁿ⁺¹` and `c★` live at cell `Center`s in the vertical.
"""
function implicit_diffusion_solver(::VerticallyImplicitDiscretization, closure, diffusivities, arch, grid)

    right_hand_side = arch_array(arch, zeros(grid.Nx, grid.Ny, grid.Nz))

    solver = BatchedTridiagonal(arch, grid;
                                lower_diagonal = vertical_diffusion_lower_diagonal,
                                      diagonal = vertical_diffusion_diagonal,
                                upper_diagonal = vertical_diffusion_upper_diagonal,
                                right_hand_side = right_hand_side) 

    return solver
end
