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

const ETD = ExplicitTimeDiscretization

@inline advective_momentum_flux_Uu(i, j, k, grid, scheme::CenteredScheme, ::ETD, U, u) = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, Ax_qᶠᶜᶜ, U) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)
@inline advective_momentum_flux_Vu(i, j, k, grid, scheme::CenteredScheme, ::ETD, V, u) = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, Ay_qᶜᶠᶜ, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
@inline advective_momentum_flux_Wu(i, j, k, grid, scheme::CenteredScheme, ::ETD, W, u) = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, Az_qᶜᶜᶠ, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u)

@inline advective_momentum_flux_Uv(i, j, k, grid, scheme::CenteredScheme, ::ETD, U, v) = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, Ax_qᶠᶜᶜ, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
@inline advective_momentum_flux_Vv(i, j, k, grid, scheme::CenteredScheme, ::ETD, V, v) = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, Ay_qᶜᶠᶜ, V) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
@inline advective_momentum_flux_Wv(i, j, k, grid, scheme::CenteredScheme, ::ETD, W, v) = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, Az_qᶜᶜᶠ, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v)

@inline advective_momentum_flux_Uw(i, j, k, grid, scheme::CenteredScheme, ::ETD, U, w) = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, Ax_qᶠᶜᶜ, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)
@inline advective_momentum_flux_Vw(i, j, k, grid, scheme::CenteredScheme, ::ETD, V, w) = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, Ay_qᶜᶠᶜ, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)
@inline advective_momentum_flux_Ww(i, j, k, grid, scheme::CenteredScheme, ::ETD, W, w) = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, Az_qᶜᶜᶠ, W) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)

#####
##### Advective tracer flux operators
#####

@inline advective_tracer_flux_x(i, j, k, grid, scheme::CenteredScheme, ::ETD, U, c) = @inbounds Ax_qᶠᶜᶜ(i, j, k, grid, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_y(i, j, k, grid, scheme::CenteredScheme, ::ETD, V, c) = @inbounds Ay_qᶜᶠᶜ(i, j, k, grid, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_z(i, j, k, grid, scheme::CenteredScheme, ::ETD, W, c) = @inbounds Az_qᶜᶜᶠ(i, j, k, grid, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)
