#####
##### Momentum and advective tracer flux operators for centered advection schemes
#####
##### See topologically_conditional_interpolation.jl for an explanation of the underscore-prepended functions _symmetric_interpolate_*.
#####

const Centered = AbstractCenteredAdvectionScheme

#####
##### Momentum flux operators
#####
##### Note the convention "momentum_flux_Ua" corresponds to the advection _of_ a _by_ U.
#####

@inline momentum_flux_uu(i, j, k, grid, scheme::Centered, U, u) = Axᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)
@inline momentum_flux_uv(i, j, k, grid, scheme::Centered, V, u) = Ayᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
@inline momentum_flux_uw(i, j, k, grid, scheme::Centered, W, u) = Azᶠᶜᵃ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u)

@inline momentum_flux_vu(i, j, k, grid, scheme::Centered, U, v) = Axᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
@inline momentum_flux_vv(i, j, k, grid, scheme::Centered, V, v) = Ayᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
@inline momentum_flux_vw(i, j, k, grid, scheme::Centered, W, v) = Azᶜᶠᵃ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v)

@inline momentum_flux_wu(i, j, k, grid, scheme::Centered, U, w) = Axᶠᶜᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)
@inline momentum_flux_wv(i, j, k, grid, scheme::Centered, V, w) = Ayᶜᶠᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)
@inline momentum_flux_ww(i, j, k, grid, scheme::Centered, W, w) = Azᶜᶜᵃ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)

#####
##### Advective tracer flux operators
#####
    
@inline advective_tracer_flux_x(i, j, k, grid, scheme::Centered, U, c) = @inbounds Ax_uᶠᶜᶜ(i, j, k, grid, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_y(i, j, k, grid, scheme::Centered, V, c) = @inbounds Ay_vᶜᶠᶜ(i, j, k, grid, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_z(i, j, k, grid, scheme::Centered, W, c) = @inbounds Az_wᶜᶜᵃ(i, j, k, grid, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)
