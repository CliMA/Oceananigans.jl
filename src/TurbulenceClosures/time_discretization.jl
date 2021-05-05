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
