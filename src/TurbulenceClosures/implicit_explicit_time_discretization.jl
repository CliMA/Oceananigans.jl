using Oceananigans.Advection: Advection, ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization, AdaptiveImplicitVerticalAdvection

@inline Advection.time_discretization(::AbstractTurbulenceClosure{TimeDiscretization}) where TimeDiscretization = TimeDiscretization()
@inline Advection.time_discretization(::Nothing) = ExplicitTimeDiscretization() # placeholder for closure::Nothing

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
