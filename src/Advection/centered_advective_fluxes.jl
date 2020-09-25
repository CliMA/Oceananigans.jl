#####
##### Momentum and tracer advection operators for centered advection schemes
#####

const Centered = AbstractCenteredAdvectionScheme

#####
##### Momentum advection operators
#####

@inline momentum_flux_uu(i, j, k, grid, scheme::Centered, u)    = Axᵃᵃᶠ(i, j, k, grid) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)
@inline momentum_flux_uv(i, j, k, grid, scheme::Centered, u, v) = Ayᵃᵃᶠ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
@inline momentum_flux_uw(i, j, k, grid, scheme::Centered, u, w) = Azᵃᵃᵃ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u)

@inline momentum_flux_vu(i, j, k, grid, scheme::Centered, u, v) = Axᵃᵃᶠ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
@inline momentum_flux_vv(i, j, k, grid, scheme::Centered, v)    = Ayᵃᵃᶠ(i, j, k, grid) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
@inline momentum_flux_vw(i, j, k, grid, scheme::Centered, v, w) = Azᵃᵃᵃ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v)

@inline momentum_flux_wu(i, j, k, grid, scheme::Centered, u, w) = Axᵃᵃᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)
@inline momentum_flux_wv(i, j, k, grid, scheme::Centered, v, w) = Ayᵃᵃᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)
@inline momentum_flux_ww(i, j, k, grid, scheme::Centered, w)    = Azᵃᵃᵃ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)

#####
##### Tracer advection operators
#####
    
@inline advective_tracer_flux_x(i, j, k, grid, scheme::Centered, u, c) = Ax_ψᵃᵃᶠ(i, j, k, grid, u) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_y(i, j, k, grid, scheme::Centered, v, c) = Ay_ψᵃᵃᶠ(i, j, k, grid, v) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_z(i, j, k, grid, scheme::Centered, w, c) = Az_ψᵃᵃᵃ(i, j, k, grid, w) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)
