using Oceananigans.Grids: OctaHEALPixMapping, SphericalShellGrid
using Oceananigans.Operators: Axᶜᶜᶜ, Axᶠᶜᶠ, Axᶠᶠᶜ, Ayᶜᶜᶜ, Ayᶜᶠᶠ, Ayᶠᶠᶜ, Azᶜᶠᶠ, Azᶠᶜᶠ,
                              covariant_to_volume_flux_uᶠᶜᶜ, covariant_to_volume_flux_vᶜᶠᶜ

const OHPSG = SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:OctaHEALPixMapping}

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

@inline advective_momentum_flux_Uu(i, j, k, grid, scheme::CenteredScheme, U, u) = @inbounds Axᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)
@inline advective_momentum_flux_Vu(i, j, k, grid, scheme::CenteredScheme, V, u) = @inbounds Ayᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
@inline advective_momentum_flux_Wu(i, j, k, grid, scheme::CenteredScheme, W, u) = @inbounds Azᶠᶜᶠ(i, j, k, grid) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u)

@inline advective_momentum_flux_Uv(i, j, k, grid, scheme::CenteredScheme, U, v) = @inbounds Axᶠᶠᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
@inline advective_momentum_flux_Vv(i, j, k, grid, scheme::CenteredScheme, V, v) = @inbounds Ayᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
@inline advective_momentum_flux_Wv(i, j, k, grid, scheme::CenteredScheme, W, v) = @inbounds Azᶜᶠᶠ(i, j, k, grid) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v)

@inline advective_momentum_flux_Uw(i, j, k, grid, scheme::CenteredScheme, U, w) = @inbounds Axᶠᶜᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)
@inline advective_momentum_flux_Vw(i, j, k, grid, scheme::CenteredScheme, V, w) = @inbounds Ayᶜᶠᶠ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)
@inline advective_momentum_flux_Ww(i, j, k, grid, scheme::CenteredScheme, W, w) = @inbounds Azᶜᶜᶜ(i, j, k, grid) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, W) * _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)

@inline function advective_momentum_flux_Uu(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, u)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    return _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, covariant_to_volume_flux_uᶠᶜᶜ, u_component, v_component) *
           _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)
end

@inline function advective_momentum_flux_Vu(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, u)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    return _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, covariant_to_volume_flux_vᶜᶠᶜ, u_component, v_component) *
           _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
end

@inline function advective_momentum_flux_Uv(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, v)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    return _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, covariant_to_volume_flux_uᶠᶜᶜ, u_component, v_component) *
           _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
end

@inline function advective_momentum_flux_Vv(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, v)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    return _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, covariant_to_volume_flux_vᶜᶠᶜ, u_component, v_component) *
           _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
end

@inline function advective_momentum_flux_Uw(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, w)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    return _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, covariant_to_volume_flux_uᶠᶜᶜ, u_component, v_component) *
           _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)
end

@inline function advective_momentum_flux_Vw(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, w)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    return _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, covariant_to_volume_flux_vᶜᶠᶜ, u_component, v_component) *
           _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)
end

#####
##### Advective tracer flux operators
#####

@inline advective_tracer_flux_x(i, j, k, grid, scheme::CenteredScheme, U, c) = @inbounds Ax_qᶠᶜᶜ(i, j, k, grid, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_y(i, j, k, grid, scheme::CenteredScheme, V, c) = @inbounds Ay_qᶜᶠᶜ(i, j, k, grid, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_z(i, j, k, grid, scheme::CenteredScheme, W, c) = @inbounds Az_qᶜᶜᶠ(i, j, k, grid, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)

@inline octahealpix_folded_polar_i(i, grid) =
    mod(i - 1 + grid.Nx ÷ 2, grid.Nx) + 1

@inline function octahealpix_polar_xface_tracer_state(i, j, k, grid, c)
    FT = eltype(grid)
    half = convert(FT, 1//2)
    quarter = convert(FT, 1//4)
    adjacent_j = ifelse(j == 1, 2, grid.Ny - 1)
    folded_i = octahealpix_folded_polar_i(i, grid)
    folded_im = octahealpix_folded_polar_i(i - 1, grid)

    @inbounds pole_state = quarter * (c[i, j, k] +
                                      c[i-1, j, k] +
                                      c[folded_i, j, k] +
                                      c[folded_im, j, k])

    @inbounds outer_edge_state = quarter * (c[i, j, k] +
                                            c[i-1, j, k] +
                                            c[i, adjacent_j, k] +
                                            c[i-1, adjacent_j, k])

    return half * (pole_state + outer_edge_state)
end

@inline function advective_tracer_flux_x(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, c)
    ũ = spherical_shell_horizontal_tracer_flux_u(U, i, j, k)
    cᶠᶜᶜ = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
    return ũ * cᶠᶜᶜ
end

@inline function advective_tracer_flux_x(i, j, k, grid::OHPSG, scheme::CenteredScheme, U, c)
    ũ = spherical_shell_horizontal_tracer_flux_u(U, i, j, k)
    regular_cᶠᶜᶜ = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
    polar_row = (j == 1) | (j == grid.Ny)
    polar_cᶠᶜᶜ = octahealpix_polar_xface_tracer_state(i, j, k, grid, c)
    weight = convert(eltype(grid), 3//4)
    blended_cᶠᶜᶜ = regular_cᶠᶜᶜ + weight * (polar_cᶠᶜᶜ - regular_cᶠᶜᶜ)
    cᶠᶜᶜ = ifelse(polar_row, blended_cᶠᶜᶜ, regular_cᶠᶜᶜ)
    return ũ * cᶠᶜᶜ
end

@inline function advective_tracer_flux_y(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, c)
    ṽ = spherical_shell_horizontal_tracer_flux_v(U, i, j, k)
    cᶜᶠᶜ = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
    return ṽ * cᶜᶠᶜ
end
