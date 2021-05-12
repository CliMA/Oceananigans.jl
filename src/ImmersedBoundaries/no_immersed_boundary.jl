struct NoImmersedBoundary <: AbstractImmersedBoundary end

#####
##### No immersed boundary: proceed.
#####

@inline viscous_flux_ux(i, j, k, grid, ib::NoImmersedBoundary, args...) = viscous_flux_ux(i, j, k, grid, args...)
@inline viscous_flux_uy(i, j, k, grid, ib::NoImmersedBoundary, args...) = viscous_flux_uy(i, j, k, grid, args...)
@inline viscous_flux_uz(i, j, k, grid, ib::NoImmersedBoundary, args...) = viscous_flux_uz(i, j, k, grid, args...)

@inline viscous_flux_vx(i, j, k, grid, ib::NoImmersedBoundary, args...) = viscous_flux_vx(i, j, k, grid, args...)
@inline viscous_flux_vy(i, j, k, grid, ib::NoImmersedBoundary, args...) = viscous_flux_vy(i, j, k, grid, args...)
@inline viscous_flux_vz(i, j, k, grid, ib::NoImmersedBoundary, args...) = viscous_flux_vz(i, j, k, grid, args...)

@inline viscous_flux_wx(i, j, k, grid, ib::NoImmersedBoundary, args...) = viscous_flux_wx(i, j, k, grid, args...)
@inline viscous_flux_wy(i, j, k, grid, ib::NoImmersedBoundary, args...) = viscous_flux_wy(i, j, k, grid, args...)
@inline viscous_flux_wz(i, j, k, grid, ib::NoImmersedBoundary, args...) = viscous_flux_wz(i, j, k, grid, args...)

@inline diffusive_flux_x(i, j, k, grid, ib::NoImmersedBoundary, args...) = diffusive_flux_x(i, j, k, grid, args...)
@inline diffusive_flux_y(i, j, k, grid, ib::NoImmersedBoundary, args...) = diffusive_flux_y(i, j, k, grid, args...)
@inline diffusive_flux_z(i, j, k, grid, ib::NoImmersedBoundary, args...) = diffusive_flux_z(i, j, k, grid, args...)
