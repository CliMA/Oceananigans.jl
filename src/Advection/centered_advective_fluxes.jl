#####
##### Momentum and advective tracer flux operators for centered advection schemes
#####
##### See topologically_conditional_interpolation.jl for an explanation of the underscore-prepended functions _symmetric_interpolate_*.
#####

const Centered = AbstractCenteredAdvectionScheme

#####
##### Momentum flux operators
#####
##### Note the convention "momentum_flux_AB" corresponds to the advection _of_ A _by_ B.
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
##### Advective tracer flux operators
#####
    
@inline advective_tracer_flux_x(i, j, k, grid, scheme::Centered, u, c) = @inbounds Axᵃᵃᶠ(i, j, k, grid) * u[i, j, k] * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_y(i, j, k, grid, scheme::Centered, v, c) = @inbounds Ayᵃᵃᶠ(i, j, k, grid) * v[i, j, k] * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_z(i, j, k, grid, scheme::Centered, w, c) = @inbounds Azᵃᵃᵃ(i, j, k, grid) * w[i, j, k] * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)
