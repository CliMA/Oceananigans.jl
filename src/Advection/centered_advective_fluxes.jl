#####
##### Momentum and advective tracer flux operators for centered advection schemes
#####

const CenteredScheme = AbstractCenteredAdvectionScheme

#####
##### Advective momentum flux operators
#####
##### Note the convention "advective_momentum_flux_Ua" corresponds to the advection _of_ a _by_ U.
#####

const ETD = ExplicitTimeDiscretization

@inline advective_momentum_flux_Uu(i, j, k, grid, scheme::CenteredScheme, ::ETD, U, u) = symmetric_interpolate_xᶜᶜᶜ(i, j, k, grid, scheme, Ax_qᶠᶜᶜ, U) * symmetric_interpolate_xᶜᶜᶜ(i, j, k, grid, scheme, u)
@inline advective_momentum_flux_Vu(i, j, k, grid, scheme::CenteredScheme, ::ETD, V, u) = symmetric_interpolate_xᶠᶠᶜ(i, j, k, grid, scheme, Ay_qᶜᶠᶜ, V) * symmetric_interpolate_yᶠᶠᶜ(i, j, k, grid, scheme, u)
@inline advective_momentum_flux_Wu(i, j, k, grid, scheme::CenteredScheme, ::ETD, W, u) = symmetric_interpolate_xᶠᶜᶠ(i, j, k, grid, scheme, Az_qᶜᶜᶠ, W) * symmetric_interpolate_zᶠᶜᶠ(i, j, k, grid, scheme, u)

@inline advective_momentum_flux_Uv(i, j, k, grid, scheme::CenteredScheme, ::ETD, U, v) = symmetric_interpolate_yᶠᶠᶜ(i, j, k, grid, scheme, Ax_qᶠᶜᶜ, U) * symmetric_interpolate_xᶠᶠᶜ(i, j, k, grid, scheme, v)
@inline advective_momentum_flux_Vv(i, j, k, grid, scheme::CenteredScheme, ::ETD, V, v) = symmetric_interpolate_yᶜᶜᶜ(i, j, k, grid, scheme, Ay_qᶜᶠᶜ, V) * symmetric_interpolate_yᶜᶜᶜ(i, j, k, grid, scheme, v)
@inline advective_momentum_flux_Wv(i, j, k, grid, scheme::CenteredScheme, ::ETD, W, v) = symmetric_interpolate_yᶜᶠᶠ(i, j, k, grid, scheme, Az_qᶜᶜᶠ, W) * symmetric_interpolate_zᶜᶠᶠ(i, j, k, grid, scheme, v)

@inline advective_momentum_flux_Uw(i, j, k, grid, scheme::CenteredScheme, ::ETD, U, w) = symmetric_interpolate_zᶠᶜᶠ(i, j, k, grid, scheme, Ax_qᶠᶜᶜ, U) * symmetric_interpolate_xᶠᶜᶠ(i, j, k, grid, scheme, w)
@inline advective_momentum_flux_Vw(i, j, k, grid, scheme::CenteredScheme, ::ETD, V, w) = symmetric_interpolate_zᶜᶠᶠ(i, j, k, grid, scheme, Ay_qᶜᶠᶜ, V) * symmetric_interpolate_yᶜᶠᶠ(i, j, k, grid, scheme, w)
@inline advective_momentum_flux_Ww(i, j, k, grid, scheme::CenteredScheme, ::ETD, W, w) = symmetric_interpolate_zᶜᶜᶜ(i, j, k, grid, scheme, Az_qᶜᶜᶠ, W) * symmetric_interpolate_zᶜᶜᶜ(i, j, k, grid, scheme, w)

#####
##### Advective tracer flux operators
#####

@inline advective_tracer_flux_x(i, j, k, grid, scheme::CenteredScheme, ::ETD, U, c) = @inbounds Ax_qᶠᶜᶜ(i, j, k, grid, U) * symmetric_interpolate_xᶠᶜᶜ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_y(i, j, k, grid, scheme::CenteredScheme, ::ETD, V, c) = @inbounds Ay_qᶜᶠᶜ(i, j, k, grid, V) * symmetric_interpolate_yᶜᶠᶜ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_z(i, j, k, grid, scheme::CenteredScheme, ::ETD, W, c) = @inbounds Az_qᶜᶜᶠ(i, j, k, grid, W) * symmetric_interpolate_zᶜᶜᶠ(i, j, k, grid, scheme, c)
