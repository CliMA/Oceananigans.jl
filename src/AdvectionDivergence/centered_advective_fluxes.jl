#####
##### Momentum and advective tracer flux operators for centered advection schemes
#####
##### See topologically_conditional_interpolation.jl for an explanation of the underscore-prepended functions _symmetric_interpolate_*.
#####

const Centered = AbstractCenteredAdvectionScheme

#####
##### Advective momentum flux operators
#####
##### Note the convention "advective_momentum_flux_Ua" corresponds to the advection _of_ a _by_ U.
#####

@inline advective_momentum_flux_Uu(i, j, k, grid, scheme::Centered, U, u) = Axᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)
@inline advective_momentum_flux_Vu(i, j, k, grid, scheme::Centered, V, u) = Ayᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
@inline advective_momentum_flux_Wu(i, j, k, grid, scheme::Centered, W, u) = Azᶠᶜᶠ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u)

@inline advective_momentum_flux_Uv(i, j, k, grid, scheme::Centered, U, v) = Axᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
@inline advective_momentum_flux_Vv(i, j, k, grid, scheme::Centered, V, v) = Ayᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
@inline advective_momentum_flux_Wv(i, j, k, grid, scheme::Centered, W, v) = Azᶜᶠᶠ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v)

@inline advective_momentum_flux_Uw(i, j, k, grid, scheme::Centered, U, w) = Axᶠᶜᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)
@inline advective_momentum_flux_Vw(i, j, k, grid, scheme::Centered, V, w) = Ayᶜᶠᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)
@inline advective_momentum_flux_Ww(i, j, k, grid, scheme::Centered, W, w) = Azᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)

#####
##### Advective tracer flux operators
#####
    
@inline function advective_tracer_flux_x(i, j, k, grid, advection_scheme::Centered, divergence_scheme, U, c)
    ũ  = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, U)
    return Ax_qᶠᶜᶜ(i, j, k, grid, ũ) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
end

@inline function advective_tracer_flux_y(i, j, k, grid, advection_scheme::Centered, divergence_scheme, V, c)
    ṽ  = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, V)
    return Ay_qᶜᶠᶜ(i, j, k, grid, ṽ) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
end

@inline function advective_tracer_flux_z(i, j, k, grid, advection_scheme::Centered, divergence_scheme, W, c)
    w̃ = @inbounds W[i, j, k]
    return Az_qᶜᶜᶠ(i, j, k, grid, w̃) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)
end

@inline advective_tracer_flux_x(i, j, k, grid, advection_scheme::Centered, U, c) = advective_tracer_flux_x(i, j, k, grid, advection_scheme, TrivialSecondOrder(), U, c)
@inline advective_tracer_flux_y(i, j, k, grid, advection_scheme::Centered, V, c) = advective_tracer_flux_y(i, j, k, grid, advection_scheme, TrivialSecondOrder(), V, c)
@inline advective_tracer_flux_z(i, j, k, grid, advection_scheme::Centered, W, c) = advective_tracer_flux_z(i, j, k, grid, advection_scheme, TrivialSecondOrder(), W, c)

