#####
##### Momentum and advective tracer flux operators for centered advection schemes
#####
##### See topologically_conditional_interpolation.jl for an explanation of the underscore-prepended functions _symmetric_interpolate_*.
#####

const AbstractCenteredMultiDimensionalScheme = AbstractMultiDimensionalAdvectionScheme{<:Any, <:Any, <:AbstractCenteredAdvectionScheme} 
const CenteredScheme = Union{AbstractCenteredAdvectionScheme, AbstractCenteredMultiDimensionalScheme}

#####
##### Advective momentum flux operators
#####
##### Note the convention "advective_momentum_flux_Ua" corresponds to the advection _of_ a _by_ U.
#####

@inline advective_momentum_flux_Uu(i, j, k, grid, scheme::CenteredScheme, U, u) = Axᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)
@inline advective_momentum_flux_Vu(i, j, k, grid, scheme::CenteredScheme, V, u) = Ayᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
@inline advective_momentum_flux_Wu(i, j, k, grid, scheme::CenteredScheme, W, u) = Azᶠᶜᶠ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u)

@inline advective_momentum_flux_Uv(i, j, k, grid, scheme::CenteredScheme, U, v) = Axᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
@inline advective_momentum_flux_Vv(i, j, k, grid, scheme::CenteredScheme, V, v) = Ayᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
@inline advective_momentum_flux_Wv(i, j, k, grid, scheme::CenteredScheme, W, v) = Azᶜᶠᶠ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v)

@inline advective_momentum_flux_Uw(i, j, k, grid, scheme::CenteredScheme, U, w) = Axᶠᶜᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)
@inline advective_momentum_flux_Vw(i, j, k, grid, scheme::CenteredScheme, V, w) = Ayᶜᶠᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)
@inline advective_momentum_flux_Ww(i, j, k, grid, scheme::CenteredScheme, W, w) = Azᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)

#####
##### Advective tracer flux operators
#####
    
@inline advective_tracer_flux_x(i, j, k, grid, scheme::CenteredScheme, U, c) = @inbounds Ax_qᶠᶜᶜ(i, j, k, grid, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_y(i, j, k, grid, scheme::CenteredScheme, V, c) = @inbounds Ay_qᶜᶠᶜ(i, j, k, grid, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_z(i, j, k, grid, scheme::CenteredScheme, W, c) = @inbounds Az_qᶜᶜᶠ(i, j, k, grid, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)
