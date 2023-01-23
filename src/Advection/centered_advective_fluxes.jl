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

@inline advective_momentum_flux_Uu(i, j, k, grid, scheme::CenteredScheme, U, u, is, js, ks) = @inbounds Axᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, U, is, js, ks) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u, is, js, ks)
@inline advective_momentum_flux_Vu(i, j, k, grid, scheme::CenteredScheme, V, u, is, js, ks) = @inbounds Ayᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, V, is, js, ks) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u, is, js, ks)
@inline advective_momentum_flux_Wu(i, j, k, grid, scheme::CenteredScheme, W, u, is, js, ks) = @inbounds Azᶠᶜᶠ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W, is, js, ks) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u, is, js, ks)

@inline advective_momentum_flux_Uv(i, j, k, grid, scheme::CenteredScheme, U, v, is, js, ks) = @inbounds Axᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, U, is, js, ks) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v, is, js, ks)
@inline advective_momentum_flux_Vv(i, j, k, grid, scheme::CenteredScheme, V, v, is, js, ks) = @inbounds Ayᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, V, is, js, ks) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v, is, js, ks)
@inline advective_momentum_flux_Wv(i, j, k, grid, scheme::CenteredScheme, W, v, is, js, ks) = @inbounds Azᶜᶠᶠ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W, is, js, ks) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v, is, js, ks)

@inline advective_momentum_flux_Uw(i, j, k, grid, scheme::CenteredScheme, U, w, is, js, ks) = @inbounds Axᶠᶜᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, U, is, js, ks) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w, is, js, ks)
@inline advective_momentum_flux_Vw(i, j, k, grid, scheme::CenteredScheme, V, w, is, js, ks) = @inbounds Ayᶜᶠᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, V, is, js, ks) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w, is, js, ks)
@inline advective_momentum_flux_Ww(i, j, k, grid, scheme::CenteredScheme, W, w, is, js, ks) = @inbounds Azᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, W, is, js, ks) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w, is, js, ks)

#####
##### Advective tracer flux operators
#####
    
@inline advective_tracer_flux_x(i, j, k, grid, scheme::CenteredScheme, U, c, is, js, ks) = @inbounds Axᶠᶜᶜ(i, j, k, grid) * U[is, js, ks] * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c, is, js, ks)
@inline advective_tracer_flux_y(i, j, k, grid, scheme::CenteredScheme, V, c, is, js, ks) = @inbounds Ayᶜᶠᶜ(i, j, k, grid) * V[is, js, ks] * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c, is, js, ks)
@inline advective_tracer_flux_z(i, j, k, grid, scheme::CenteredScheme, W, c, is, js, ks) = @inbounds Azᶜᶜᶠ(i, j, k, grid) * W[is, js, ks] * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c, is, js, ks)
