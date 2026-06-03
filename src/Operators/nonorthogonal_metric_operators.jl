#####
##### Non-orthogonal horizontal metric and Hodge operators
#####

@inline _node_value(q::Number, i, j, k) = q
@inline _node_value(q, i, j, k) = @inbounds q[i, j, k]

const OHPSG = SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:OctaHEALPixMapping}

@inline octahealpix_polar_fold_flux_factor(grid) =
    convert(eltype(grid), 7//64)

@inline octahealpix_polar_fold_j(j, grid) =
    (j == 1) | (j == grid.Ny + 1)

@inline Jᶜᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.Jᶜᶜᵃ[i, j]
@inline Jᶠᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.Jᶠᶜᵃ[i, j]
@inline Jᶜᶠᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.Jᶜᶠᵃ[i, j]

@inline g¹¹ᶜᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.g¹¹ᶜᶜᵃ[i, j]
@inline g¹²ᶜᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.g¹²ᶜᶜᵃ[i, j]
@inline g²¹ᶜᶜᵃ(i, j, k, grid::SSG) = g¹²ᶜᶜᵃ(i, j, k, grid)
@inline g²²ᶜᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.g²²ᶜᶜᵃ[i, j]
@inline g¹¹ᶠᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.g¹¹ᶠᶜᵃ[i, j]
@inline g¹²ᶠᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.g¹²ᶠᶜᵃ[i, j]
@inline g²¹ᶜᶠᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.g²¹ᶜᶠᵃ[i, j]
@inline g²²ᶜᶠᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.g²²ᶜᶠᵃ[i, j]

@inline G¹¹ᶜᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.G¹¹ᶜᶜᵃ[i, j]
@inline G¹²ᶜᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.G¹²ᶜᶜᵃ[i, j]
@inline G²¹ᶜᶜᵃ(i, j, k, grid::SSG) = G¹²ᶜᶜᵃ(i, j, k, grid)
@inline G²²ᶜᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.G²²ᶜᶜᵃ[i, j]
@inline G¹¹ᶠᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.G¹¹ᶠᶜᵃ[i, j]
@inline G¹²ᶠᶜᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.G¹²ᶠᶜᵃ[i, j]
@inline G²¹ᶜᶠᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.G²¹ᶜᶠᵃ[i, j]
@inline G²²ᶜᶠᵃ(i, j, k, grid::SSG) = @inbounds grid.metrics.G²²ᶜᶠᵃ[i, j]

@inline computational_width_uᶠᶜᶜ(i, j, k, grid::SSG) = one(grid)
@inline computational_width_vᶜᶠᶜ(i, j, k, grid::SSG) = one(grid)
@inline transverse_computational_width_uᶠᶜᶜ(i, j, k, grid::SSG) = one(grid)
@inline transverse_computational_width_vᶜᶠᶜ(i, j, k, grid::SSG) = one(grid)
@inline horizontal_computational_areaᶠᶠᶜ(i, j, k, grid::SSG) = one(grid)

@inline function computational_width_uᶠᶜᶜ(i, j, k,
                                          grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                                   <:EquiangularGnomonicCubedSpherePanel})
    lower, upper = grid.mapping.α
    return (upper - lower) / grid.Nx
end

@inline horizontal_computational_areaᶠᶠᶜ(i, j, k,
                                         grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                                  <:EquiangularGnomonicCubedSpherePanel}) =
    computational_width_uᶠᶜᶜ(i, j, k, grid) * computational_width_vᶜᶠᶜ(i, j, k, grid)

@inline function computational_width_vᶜᶠᶜ(i, j, k,
                                          grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                                   <:EquiangularGnomonicCubedSpherePanel})
    lower, upper = grid.mapping.β
    return (upper - lower) / grid.Ny
end

@inline function transverse_computational_width_uᶠᶜᶜ(i, j, k,
                                                     grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                                              <:EquiangularGnomonicCubedSpherePanel})
    lower, upper = grid.mapping.β
    return (upper - lower) / grid.Ny
end

@inline function transverse_computational_width_vᶜᶠᶜ(i, j, k,
                                                     grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                                              <:EquiangularGnomonicCubedSpherePanel})
    lower, upper = grid.mapping.α
    return (upper - lower) / grid.Nx
end

for LZ in (:ᶜ, :ᶠ)
    @eval begin
        @inline $(Symbol(:Jᶜᶜ, LZ))(i, j, k, grid::SSG) = Jᶜᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:Jᶠᶜ, LZ))(i, j, k, grid::SSG) = Jᶠᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:Jᶜᶠ, LZ))(i, j, k, grid::SSG) = Jᶜᶠᵃ(i, j, k, grid)

        @inline $(Symbol(:g¹¹ᶜᶜ, LZ))(i, j, k, grid::SSG) = g¹¹ᶜᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:g¹²ᶜᶜ, LZ))(i, j, k, grid::SSG) = g¹²ᶜᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:g²¹ᶜᶜ, LZ))(i, j, k, grid::SSG) = g²¹ᶜᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:g²²ᶜᶜ, LZ))(i, j, k, grid::SSG) = g²²ᶜᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:g¹¹ᶠᶜ, LZ))(i, j, k, grid::SSG) = g¹¹ᶠᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:g¹²ᶠᶜ, LZ))(i, j, k, grid::SSG) = g¹²ᶠᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:g²¹ᶜᶠ, LZ))(i, j, k, grid::SSG) = g²¹ᶜᶠᵃ(i, j, k, grid)
        @inline $(Symbol(:g²²ᶜᶠ, LZ))(i, j, k, grid::SSG) = g²²ᶜᶠᵃ(i, j, k, grid)

        @inline $(Symbol(:G¹¹ᶜᶜ, LZ))(i, j, k, grid::SSG) = G¹¹ᶜᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:G¹²ᶜᶜ, LZ))(i, j, k, grid::SSG) = G¹²ᶜᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:G²¹ᶜᶜ, LZ))(i, j, k, grid::SSG) = G²¹ᶜᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:G²²ᶜᶜ, LZ))(i, j, k, grid::SSG) = G²²ᶜᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:G¹¹ᶠᶜ, LZ))(i, j, k, grid::SSG) = G¹¹ᶠᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:G¹²ᶠᶜ, LZ))(i, j, k, grid::SSG) = G¹²ᶠᶜᵃ(i, j, k, grid)
        @inline $(Symbol(:G²¹ᶜᶠ, LZ))(i, j, k, grid::SSG) = G²¹ᶜᶠᵃ(i, j, k, grid)
        @inline $(Symbol(:G²²ᶜᶠ, LZ))(i, j, k, grid::SSG) = G²²ᶜᶠᵃ(i, j, k, grid)
    end
end

@inline regular_covariant_to_contravariant_flux_uᶠᶜᶜ(i, j, k, grid::SSG, u₁, u₂) =
    G¹¹ᶠᶜᶜ(i, j, k, grid) * _node_value(u₁, i, j, k) +
    G¹²ᶠᶜᶜ(i, j, k, grid) * ℑxyᶠᶜᵃ(i, j, k, grid, u₂)

@inline covariant_to_contravariant_flux_uᶠᶜᶜ(i, j, k, grid::SSG, u₁, u₂) =
    regular_covariant_to_contravariant_flux_uᶠᶜᶜ(i, j, k, grid, u₁, u₂)

@inline regular_covariant_to_contravariant_flux_vᶜᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    G²¹ᶜᶠᶜ(i, j, k, grid) * ℑxyᶜᶠᵃ(i, j, k, grid, u₁) +
    G²²ᶜᶠᶜ(i, j, k, grid) * _node_value(u₂, i, j, k)

@inline covariant_to_contravariant_flux_vᶜᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    regular_covariant_to_contravariant_flux_vᶜᶠᶜ(i, j, k, grid, u₁, u₂)

@inline function covariant_to_contravariant_flux_vᶜᶠᶜ(i, j, k, grid::OHPSG, u₁, u₂)
    regular_flux = regular_covariant_to_contravariant_flux_vᶜᶠᶜ(i, j, k, grid, u₁, u₂)

    south_polar_fold = j == 1
    north_polar_fold = j == grid.Ny + 1
    polar_fold = south_polar_fold | north_polar_fold
    adjacent_cell_j = ifelse(north_polar_fold, grid.Ny, 1)
    polar_fold_factor = octahealpix_polar_fold_flux_factor(grid)
    polar_fold_flux = polar_fold_factor *
                      Azᶜᶜᶜ(i, adjacent_cell_j, k, grid) *
                      g²²ᶜᶠᶜ(i, j, k, grid) *
                      _node_value(u₂, i, j, k)

    return ifelse(polar_fold, polar_fold_flux, regular_flux)
end

@inline covariant_to_contravariant_velocity_uᶠᶜᶜ(i, j, k, grid::SSG, u₁, u₂) =
    covariant_to_contravariant_flux_uᶠᶜᶜ(i, j, k, grid, u₁, u₂) / Jᶠᶜᶜ(i, j, k, grid)

@inline covariant_to_contravariant_velocity_vᶜᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    covariant_to_contravariant_flux_vᶜᶠᶜ(i, j, k, grid, u₁, u₂) / Jᶜᶠᶜ(i, j, k, grid)

@inline function covariant_to_contravariant_velocity_vᶜᶠᶜ(i, j, k, grid::OHPSG, u₁, u₂)
    regular_velocity = regular_covariant_to_contravariant_flux_vᶜᶠᶜ(i, j, k, grid, u₁, u₂) / Jᶜᶠᶜ(i, j, k, grid)
    polar_fold = octahealpix_polar_fold_j(j, grid)
    return ifelse(polar_fold, zero(grid), regular_velocity)
end

@inline covariant_to_volume_flux_uᶠᶜᶜ(i, j, k, grid::SSG, u₁, u₂) =
    Δzᶠᶜᶜ(i, j, k, grid) *
    transverse_computational_width_uᶠᶜᶜ(i, j, k, grid) *
    covariant_to_contravariant_flux_uᶠᶜᶜ(i, j, k, grid, u₁, u₂)

@inline covariant_to_volume_flux_vᶜᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    Δzᶜᶠᶜ(i, j, k, grid) *
    transverse_computational_width_vᶜᶠᶜ(i, j, k, grid) *
    covariant_to_contravariant_flux_vᶜᶠᶜ(i, j, k, grid, u₁, u₂)

@inline hodge_diagonal_volume_flux_uᶠᶜᶜ(i, j, k, grid::SSG) =
    Δzᶠᶜᶜ(i, j, k, grid) *
    transverse_computational_width_uᶠᶜᶜ(i, j, k, grid) *
    G¹¹ᶠᶜᶜ(i, j, k, grid)

@inline hodge_diagonal_volume_flux_vᶜᶠᶜ(i, j, k, grid::SSG) =
    Δzᶜᶠᶜ(i, j, k, grid) *
    transverse_computational_width_vᶜᶠᶜ(i, j, k, grid) *
    G²²ᶜᶠᶜ(i, j, k, grid)

@inline function hodge_diagonal_volume_flux_vᶜᶠᶜ(i, j, k, grid::OHPSG)
    regular_diagonal = Δzᶜᶠᶜ(i, j, k, grid) *
                       transverse_computational_width_vᶜᶠᶜ(i, j, k, grid) *
                       G²²ᶜᶠᶜ(i, j, k, grid)

    south_polar_fold = j == 1
    north_polar_fold = j == grid.Ny + 1
    polar_fold = south_polar_fold | north_polar_fold
    adjacent_cell_j = ifelse(north_polar_fold, grid.Ny, 1)
    polar_fold_diagonal = Δzᶜᶠᶜ(i, j, k, grid) *
                          transverse_computational_width_vᶜᶠᶜ(i, j, k, grid) *
                          octahealpix_polar_fold_flux_factor(grid) *
                          Azᶜᶜᶜ(i, adjacent_cell_j, k, grid) *
                          g²²ᶜᶠᶜ(i, j, k, grid)

    return ifelse(polar_fold, polar_fold_diagonal, regular_diagonal)
end

@inline hodge_diagonal_volume_flux(i, j, k, grid::SSG, source_kind) =
    ifelse(source_kind == 1,
           hodge_diagonal_volume_flux_uᶠᶜᶜ(i, j, k, grid),
           hodge_diagonal_volume_flux_vᶜᶠᶜ(i, j, k, grid))

@inline hodge_weight_uᶠᶜᶜ(i, j, k, grid::SSG) =
    convert(eltype(grid), 1//2) * Azᶠᶜᶜ(i, j, k, grid)

@inline hodge_weight_vᶜᶠᶜ(i, j, k, grid::SSG) =
    convert(eltype(grid), 1//2) * Azᶜᶠᶜ(i, j, k, grid)

@inline hodge_compatible_boundary_flux_uᶠᶜᶜ(i, j, k, grid::OHPSG, U, V) =
    _hodge_compatible_boundary_flux_uᶠᶜᶜ(i, j, k, grid, U, V,
        octahealpix_covariant_xface_halo_source(i, j, grid.Nx, grid.Ny, grid.connectivity)...)

@inline function _hodge_compatible_boundary_flux_uᶠᶜᶜ(i, j, k, grid::OHPSG, U, V,
                                                      source_kind, source_i, source_j, sign)
    boundary_diagonal = hodge_diagonal_volume_flux_uᶠᶜᶜ(i, j, k, grid)
    source_diagonal = hodge_diagonal_volume_flux(source_i, source_j, k, grid, source_kind)
    source_flux = ifelse(source_kind == 1,
                         _node_value(U, source_i, source_j, k),
                         _node_value(V, source_i, source_j, k))

    return sign * boundary_diagonal / source_diagonal * source_flux
end

@inline hodge_compatible_boundary_flux_vᶜᶠᶜ(i, j, k, grid::OHPSG, U, V) =
    _hodge_compatible_boundary_flux_vᶜᶠᶜ(i, j, k, grid, U, V,
        octahealpix_covariant_yface_halo_source(i, j, grid.Nx, grid.Ny, grid.connectivity)...)

@inline function _hodge_compatible_boundary_flux_vᶜᶠᶜ(i, j, k, grid::OHPSG, U, V,
                                                      source_kind, source_i, source_j, sign)
    boundary_diagonal = hodge_diagonal_volume_flux_vᶜᶠᶜ(i, j, k, grid)
    source_diagonal = hodge_diagonal_volume_flux(source_i, source_j, k, grid, source_kind)
    source_flux = ifelse(source_kind == 1,
                         _node_value(U, source_i, source_j, k),
                         _node_value(V, source_i, source_j, k))

    return sign * boundary_diagonal / source_diagonal * source_flux
end

@inline function hodge_compatible_volume_flux_div_xyᶜᶜᶜ(i, j, k, grid::OHPSG, U, V)
    right_flux = ifelse(i == grid.Nx,
                        hodge_compatible_boundary_flux_uᶠᶜᶜ(i + 1, j, k, grid, U, V),
                        _node_value(U, i + 1, j, k))

    top_flux = ifelse(j == grid.Ny,
                      hodge_compatible_boundary_flux_vᶜᶠᶜ(i, j + 1, k, grid, U, V),
                      _node_value(V, i, j + 1, k))

    left_flux = _node_value(U, i, j, k)
    bottom_flux = _node_value(V, i, j, k)

    return right_flux - left_flux + top_flux - bottom_flux
end

@inline hodge_compatible_pressure_correction_uᶠᶜᶜ(i, j, k, grid::OHPSG, p) =
    (hodge_compatible_raw_pressure_correction_uᶠᶜᶜ(i, j, k, grid, p) +
     hodge_compatible_boundary_pressure_correction_uᶠᶜᶜ(i, j, k, grid, p)) /
    hodge_weight_uᶠᶜᶜ(i, j, k, grid)

@inline function hodge_compatible_raw_pressure_correction_uᶠᶜᶜ(i, j, k, grid::OHPSG, p)
    left_cell_pressure = ifelse(i == 1,
                                zero(grid),
                                _node_value(p, i - 1, j, k))

    right_cell_pressure = _node_value(p, i, j, k)

    return left_cell_pressure - right_cell_pressure
end

@inline function hodge_compatible_boundary_pressure_correction_uᶠᶜᶜ(i, j, k, grid::OHPSG, p)
    correction = zero(grid)

    for boundary_j in 1:grid.Ny
        source_kind, source_i, source_j, sign =
            octahealpix_covariant_xface_halo_source(grid.Nx + 1, boundary_j, grid.Nx, grid.Ny, grid.connectivity)

        source_matches = (source_kind == 1) & (source_i == i) & (source_j == j)
        boundary_diagonal = hodge_diagonal_volume_flux_uᶠᶜᶜ(grid.Nx + 1, boundary_j, k, grid)
        source_diagonal = hodge_diagonal_volume_flux(source_i, source_j, k, grid, source_kind)
        boundary_pressure = _node_value(p, grid.Nx, boundary_j, k)
        boundary_correction = sign * boundary_diagonal / source_diagonal * boundary_pressure
        correction += ifelse(source_matches, boundary_correction, zero(grid))
    end

    for boundary_i in 1:grid.Nx
        source_kind, source_i, source_j, sign =
            octahealpix_covariant_yface_halo_source(boundary_i, grid.Ny + 1, grid.Nx, grid.Ny, grid.connectivity)

        source_matches = (source_kind == 1) & (source_i == i) & (source_j == j)
        boundary_diagonal = hodge_diagonal_volume_flux_vᶜᶠᶜ(boundary_i, grid.Ny + 1, k, grid)
        source_diagonal = hodge_diagonal_volume_flux(source_i, source_j, k, grid, source_kind)
        boundary_pressure = _node_value(p, boundary_i, grid.Ny, k)
        boundary_correction = sign * boundary_diagonal / source_diagonal * boundary_pressure
        correction += ifelse(source_matches, boundary_correction, zero(grid))
    end

    return correction
end

@inline hodge_compatible_pressure_correction_vᶜᶠᶜ(i, j, k, grid::OHPSG, p) =
    (hodge_compatible_raw_pressure_correction_vᶜᶠᶜ(i, j, k, grid, p) +
     hodge_compatible_boundary_pressure_correction_vᶜᶠᶜ(i, j, k, grid, p)) /
    hodge_weight_vᶜᶠᶜ(i, j, k, grid)

@inline function hodge_compatible_raw_pressure_correction_vᶜᶠᶜ(i, j, k, grid::OHPSG, p)
    bottom_cell_pressure = ifelse(j == 1,
                                  zero(grid),
                                  _node_value(p, i, j - 1, k))

    top_cell_pressure = _node_value(p, i, j, k)

    return bottom_cell_pressure - top_cell_pressure
end

@inline function hodge_compatible_boundary_pressure_correction_vᶜᶠᶜ(i, j, k, grid::OHPSG, p)
    correction = zero(grid)

    for boundary_j in 1:grid.Ny
        source_kind, source_i, source_j, sign =
            octahealpix_covariant_xface_halo_source(grid.Nx + 1, boundary_j, grid.Nx, grid.Ny, grid.connectivity)

        source_matches = (source_kind == 2) & (source_i == i) & (source_j == j)
        boundary_diagonal = hodge_diagonal_volume_flux_uᶠᶜᶜ(grid.Nx + 1, boundary_j, k, grid)
        source_diagonal = hodge_diagonal_volume_flux(source_i, source_j, k, grid, source_kind)
        boundary_pressure = _node_value(p, grid.Nx, boundary_j, k)
        boundary_correction = sign * boundary_diagonal / source_diagonal * boundary_pressure
        correction += ifelse(source_matches, boundary_correction, zero(grid))
    end

    for boundary_i in 1:grid.Nx
        source_kind, source_i, source_j, sign =
            octahealpix_covariant_yface_halo_source(boundary_i, grid.Ny + 1, grid.Nx, grid.Ny, grid.connectivity)

        source_matches = (source_kind == 2) & (source_i == i) & (source_j == j)
        boundary_diagonal = hodge_diagonal_volume_flux_vᶜᶠᶜ(boundary_i, grid.Ny + 1, k, grid)
        source_diagonal = hodge_diagonal_volume_flux(source_i, source_j, k, grid, source_kind)
        boundary_pressure = _node_value(p, boundary_i, grid.Ny, k)
        boundary_correction = sign * boundary_diagonal / source_diagonal * boundary_pressure
        correction += ifelse(source_matches, boundary_correction, zero(grid))
    end

    return correction
end

@inline covariant_gradient_xᶠᶜᶜ(i, j, k, grid::SSG, ϕ) =
    δxᶠᶜᶜ(i, j, k, grid, ϕ) / computational_width_uᶠᶜᶜ(i, j, k, grid)

@inline covariant_gradient_yᶜᶠᶜ(i, j, k, grid::SSG, ϕ) =
    δyᶜᶠᶜ(i, j, k, grid, ϕ) / computational_width_vᶜᶠᶜ(i, j, k, grid)

@inline covariant_gradient_xᶠᶜᶜ(i, j, k, grid::SSG, ϕ::Number) = zero(grid)
@inline covariant_gradient_yᶜᶠᶜ(i, j, k, grid::SSG, ϕ::Number) = zero(grid)

@inline covariant_gradient_xᶠᶜᶜ(i, j, k, grid::SSG, ϕ::Function, args...) =
    δxᶠᶜᶜ(i, j, k, grid, ϕ, args...) / computational_width_uᶠᶜᶜ(i, j, k, grid)

@inline covariant_gradient_yᶜᶠᶜ(i, j, k, grid::SSG, ϕ::Function, args...) =
    δyᶜᶠᶜ(i, j, k, grid, ϕ, args...) / computational_width_vᶜᶠᶜ(i, j, k, grid)

@inline ∂xᶠᶜᶜ(i, j, k, grid::SSG, ϕ) =
    covariant_gradient_xᶠᶜᶜ(i, j, k, grid, ϕ)

@inline ∂yᶜᶠᶜ(i, j, k, grid::SSG, ϕ) =
    covariant_gradient_yᶜᶠᶜ(i, j, k, grid, ϕ)

@inline ∂xᶠᶜᶜ(i, j, k, grid::SSG, ϕ::Function, args...) =
    covariant_gradient_xᶠᶜᶜ(i, j, k, grid, ϕ, args...)

@inline ∂yᶜᶠᶜ(i, j, k, grid::SSG, ϕ::Function, args...) =
    covariant_gradient_yᶜᶠᶜ(i, j, k, grid, ϕ, args...)

@inline ∂xᶠᶜᶠ(i, j, k, grid::SSG, ϕ) =
    covariant_gradient_xᶠᶜᶜ(i, j, k, grid, ϕ)

@inline ∂yᶜᶠᶠ(i, j, k, grid::SSG, ϕ) =
    covariant_gradient_yᶜᶠᶜ(i, j, k, grid, ϕ)

@inline ∂xᶠᶜᶠ(i, j, k, grid::SSG, ϕ::Function, args...) =
    covariant_gradient_xᶠᶜᶜ(i, j, k, grid, ϕ, args...)

@inline ∂yᶜᶠᶠ(i, j, k, grid::SSG, ϕ::Function, args...) =
    covariant_gradient_yᶜᶠᶜ(i, j, k, grid, ϕ, args...)

@inline ∂xᵣᶠᶜᶜ(i, j, k, grid::SSG, ϕ) =
    covariant_gradient_xᶠᶜᶜ(i, j, k, grid, ϕ)

@inline ∂yᵣᶜᶠᶜ(i, j, k, grid::SSG, ϕ) =
    covariant_gradient_yᶜᶠᶜ(i, j, k, grid, ϕ)

@inline ∂xᵣᶠᶜᶜ(i, j, k, grid::SSG, ϕ::Function, args...) =
    covariant_gradient_xᶠᶜᶜ(i, j, k, grid, ϕ, args...)

@inline ∂yᵣᶜᶠᶜ(i, j, k, grid::SSG, ϕ::Function, args...) =
    covariant_gradient_yᶜᶠᶜ(i, j, k, grid, ϕ, args...)

@inline ∂xᵣᶠᶜᶠ(i, j, k, grid::SSG, ϕ) =
    covariant_gradient_xᶠᶜᶜ(i, j, k, grid, ϕ)

@inline ∂yᵣᶜᶠᶠ(i, j, k, grid::SSG, ϕ) =
    covariant_gradient_yᶜᶠᶜ(i, j, k, grid, ϕ)

@inline ∂xᵣᶠᶜᶠ(i, j, k, grid::SSG, ϕ::Function, args...) =
    covariant_gradient_xᶠᶜᶜ(i, j, k, grid, ϕ, args...)

@inline ∂yᵣᶜᶠᶠ(i, j, k, grid::SSG, ϕ::Function, args...) =
    covariant_gradient_yᶜᶠᶜ(i, j, k, grid, ϕ, args...)

@inline surface_height_covariant_pressure_component_uᶠᶜᶜ(i, j, k, grid::SSG, η) =
    Oceananigans.ImmersedBoundaries.column_depthᶠᶜᵃ(i, j, grid) * ∂xᶠᶜᶜ(i, j, k, grid, η)

@inline surface_height_covariant_pressure_component_vᶜᶠᶜ(i, j, k, grid::SSG, η) =
    Oceananigans.ImmersedBoundaries.column_depthᶜᶠᵃ(i, j, grid) * ∂yᶜᶠᶜ(i, j, k, grid, η)

@inline surface_height_pressure_flux_uᶠᶜᶜ(i, j, k, grid::SSG, η) =
    G¹¹ᶠᶜᶜ(i, j, k, grid) * surface_height_covariant_pressure_component_uᶠᶜᶜ(i, j, k, grid, η) +
    G¹²ᶠᶜᶜ(i, j, k, grid) * ℑxyᶠᶜᵃ(i, j, k, grid, surface_height_covariant_pressure_component_vᶜᶠᶜ, η)

@inline surface_height_pressure_flux_vᶜᶠᶜ(i, j, k, grid::SSG, η) =
    G²¹ᶜᶠᶜ(i, j, k, grid) * ℑxyᶜᶠᵃ(i, j, k, grid, surface_height_covariant_pressure_component_uᶠᶜᶜ, η) +
    G²²ᶜᶠᶜ(i, j, k, grid) * surface_height_covariant_pressure_component_vᶜᶠᶜ(i, j, k, grid, η)

@inline Az_∇h²ᶜᶜᶜ(i, j, k, grid::SSG, η) =
    δxᶜᵃᵃ(i, j, k, grid, surface_height_pressure_flux_uᶠᶜᶜ, η) +
    δyᵃᶜᵃ(i, j, k, grid, surface_height_pressure_flux_vᶜᶠᶜ, η)

@inline covariant_component_uᶜᶜᶜ(i, j, k, grid::SSG, u) = ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline covariant_component_vᶜᶜᶜ(i, j, k, grid::SSG, v) = ℑyᵃᶜᵃ(i, j, k, grid, v)

@inline covariant_kinetic_energy_uᶠᶜᶜ(i, j, k, grid::SSG, u₁, u₂) =
    _node_value(u₁, i, j, k) *
    covariant_to_contravariant_velocity_uᶠᶜᶜ(i, j, k, grid, u₁, u₂)

@inline covariant_kinetic_energy_vᶜᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    _node_value(u₂, i, j, k) *
    covariant_to_contravariant_velocity_vᶜᶠᶜ(i, j, k, grid, u₁, u₂)

@inline function covariant_kinetic_energyᶜᶜᶜ(i, j, k, grid::SSG, u₁, u₂)
    half = convert(eltype(grid), 1//2)
    u¹u₁ = ℑxᶜᵃᵃ(i, j, k, grid, covariant_kinetic_energy_uᶠᶜᶜ, u₁, u₂)
    u²u₂ = ℑyᵃᶜᵃ(i, j, k, grid, covariant_kinetic_energy_vᶜᶠᶜ, u₁, u₂)
    return half * (u¹u₁ + u²u₂)
end

@inline covariant_velocity_line_integral_uᶠᶜᶜ(i, j, k, grid::SSG, u₁) =
    computational_width_uᶠᶜᶜ(i, j, k, grid) * _node_value(u₁, i, j, k)

@inline covariant_velocity_line_integral_vᶜᶠᶜ(i, j, k, grid::SSG, u₂) =
    computational_width_vᶜᶠᶜ(i, j, k, grid) * _node_value(u₂, i, j, k)

@inline covariant_vertical_circulationᶠᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    δxᶠᶠᶜ(i, j, k, grid, covariant_velocity_line_integral_vᶜᶠᶜ, u₂) -
    δyᶠᶠᶜ(i, j, k, grid, covariant_velocity_line_integral_uᶠᶜᶜ, u₁)

@inline covariant_vertical_vorticityᶠᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    covariant_vertical_circulationᶠᶠᶜ(i, j, k, grid, u₁, u₂) * Az⁻¹ᶠᶠᶜ(i, j, k, grid)

@inline covariant_vertical_vorticity_componentᶠᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    covariant_vertical_circulationᶠᶠᶜ(i, j, k, grid, u₁, u₂) / horizontal_computational_areaᶠᶠᶜ(i, j, k, grid)

@inline contravariant_velocity_uᶠᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    ℑyᵃᶠᵃ(i, j, k, grid, covariant_to_contravariant_velocity_uᶠᶜᶜ, u₁, u₂)

@inline contravariant_velocity_vᶠᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    ℑxᶠᵃᵃ(i, j, k, grid, covariant_to_contravariant_velocity_vᶜᶠᶜ, u₁, u₂)

@inline function contravariant_velocity_uᶠᶠᶜ(i, j, k, grid::OHPSG, u₁, u₂)
    regular_velocity = ℑyᵃᶠᵃ(i, j, k, grid, covariant_to_contravariant_velocity_uᶠᶜᶜ, u₁, u₂)
    polar_fold = octahealpix_polar_fold_j(j, grid)
    return ifelse(polar_fold, zero(grid), regular_velocity)
end

@inline function contravariant_velocity_vᶠᶠᶜ(i, j, k, grid::OHPSG, u₁, u₂)
    regular_velocity = ℑxᶠᵃᵃ(i, j, k, grid, covariant_to_contravariant_velocity_vᶜᶠᶜ, u₁, u₂)
    polar_fold = octahealpix_polar_fold_j(j, grid)
    return ifelse(polar_fold, zero(grid), regular_velocity)
end

@inline covariant_vorticity_flux_uᶠᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    covariant_vertical_vorticity_componentᶠᶠᶜ(i, j, k, grid, u₁, u₂) *
    contravariant_velocity_vᶠᶠᶜ(i, j, k, grid, u₁, u₂)

@inline covariant_vorticity_flux_vᶠᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    covariant_vertical_vorticity_componentᶠᶠᶜ(i, j, k, grid, u₁, u₂) *
    contravariant_velocity_uᶠᶠᶜ(i, j, k, grid, u₁, u₂)

@inline covariant_rotational_advection_uᶠᶜᶜ(i, j, k, grid::SSG, u₁, u₂) =
    - ℑyᵃᶜᵃ(i, j, k, grid, covariant_vorticity_flux_uᶠᶠᶜ, u₁, u₂)

@inline covariant_rotational_advection_vᶜᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    + ℑxᶜᵃᵃ(i, j, k, grid, covariant_vorticity_flux_vᶠᶠᶜ, u₁, u₂)

@inline covariant_bernoulli_head_uᶠᶜᶜ(i, j, k, grid::SSG, u₁, u₂) =
    covariant_gradient_xᶠᶜᶜ(i, j, k, grid, covariant_kinetic_energyᶜᶜᶜ, u₁, u₂)

@inline covariant_bernoulli_head_vᶜᶠᶜ(i, j, k, grid::SSG, u₁, u₂) =
    covariant_gradient_yᶜᶠᶜ(i, j, k, grid, covariant_kinetic_energyᶜᶜᶜ, u₁, u₂)

@inline function covariant_bernoulli_head_vᶜᶠᶜ(i, j, k, grid::OHPSG, u₁, u₂)
    bernoulli_head = covariant_gradient_yᶜᶠᶜ(i, j, k, grid, covariant_kinetic_energyᶜᶜᶜ, u₁, u₂)
    return ifelse(octahealpix_polar_fold_j(j, grid), zero(grid), bernoulli_head)
end
