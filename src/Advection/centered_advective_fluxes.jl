#####
##### Momentum and advective tracer flux operators for centered advection schemes
#####
##### See topologically_conditional_interpolation.jl for an explanation of the underscore-prepended functions _symmetric_interpolate_*.
#####

const CenteredScheme = AbstractCenteredAdvectionScheme

#####
##### Advective momentum flux operators
#####
##### Note the convention "advective_momentum_flux_Ua" corresponds to the advection _of_ a _by_ U.
#####

@inline advective_momentum_flux_Uu(i, j, k, grid, scheme::CenteredScheme, U, u, is, js, ks) = @inbounds Axᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, is, js, ks, U) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, is, js, ks, u)
@inline advective_momentum_flux_Vu(i, j, k, grid, scheme::CenteredScheme, V, u, is, js, ks) = @inbounds Ayᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, is, js, ks, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, is, js, ks, u)
@inline advective_momentum_flux_Wu(i, j, k, grid, scheme::CenteredScheme, W, u, is, js, ks) = @inbounds Azᶠᶜᶠ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, is, js, ks, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, is, js, ks, u)

@inline advective_momentum_flux_Uv(i, j, k, grid, scheme::CenteredScheme, U, v, is, js, ks) = @inbounds Axᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, is, js, ks, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, is, js, ks, v)
@inline advective_momentum_flux_Vv(i, j, k, grid, scheme::CenteredScheme, V, v, is, js, ks) = @inbounds Ayᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, is, js, ks, V) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, is, js, ks, v)
@inline advective_momentum_flux_Wv(i, j, k, grid, scheme::CenteredScheme, W, v, is, js, ks) = @inbounds Azᶜᶠᶠ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, is, js, ks, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, is, js, ks, v)

@inline advective_momentum_flux_Uw(i, j, k, grid, scheme::CenteredScheme, U, w, is, js, ks) = @inbounds Axᶠᶜᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, is, js, ks, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, is, js, ks, w)
@inline advective_momentum_flux_Vw(i, j, k, grid, scheme::CenteredScheme, V, w, is, js, ks) = @inbounds Ayᶜᶠᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, is, js, ks, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, is, js, ks, w)
@inline advective_momentum_flux_Ww(i, j, k, grid, scheme::CenteredScheme, W, w, is, js, ks) = @inbounds Azᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, is, js, ks, W) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, is, js, ks, w)

#####
##### Advective tracer flux operators
#####
    
@inline advective_tracer_flux_x(i, j, k, grid, scheme::CenteredScheme, U, c, is, js, ks) = @inbounds Ax_qᶠᶜᶜ(i, j, k, grid, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, is, js, ks, c)
@inline advective_tracer_flux_y(i, j, k, grid, scheme::CenteredScheme, V, c, is, js, ks) = @inbounds Ay_qᶜᶠᶜ(i, j, k, grid, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, is, js, ks, c)
@inline advective_tracer_flux_z(i, j, k, grid, scheme::CenteredScheme, W, c, is, js, ks) = @inbounds Az_qᶜᶜᶠ(i, j, k, grid, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, is, js, ks, c)
