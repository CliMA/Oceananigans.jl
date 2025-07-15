using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans: ∂x_b, ∂y_b, ∂z_b
using Oceananigans.Operators

struct RotatedAdvection{N, FT, U} <: AbstractUpwindBiasedAdvectionScheme{N, FT}
    upwind_scheme :: U
    rotation_percentage :: FT
    maximum_slope :: FT
    percentage_of_diapycnal_flux :: FT
end

function RotatedAdvection(upwind_scheme::U;
                          maximum_slope = 1e5, 
                          rotation_percentage = 1.0,
                          percentage_of_diapycnal_flux = 1e-7) where U 

    FT = eltype(upwind_scheme)
    
    N  = max(required_halo_size_x(upwind_scheme),
             required_halo_size_y(upwind_scheme),
             required_halo_size_z(upwind_scheme))
    
    rotation_percentage = convert(FT, rotation_percentage)
    maximum_slope = convert(FT, maximum_slope)
    percentage_of_diapycnal_flux = convert(FT, percentage_of_diapycnal_flux)

    return RotatedAdvection{N, FT, U}(upwind_scheme, rotation_percentage, maximum_slope, percentage_of_diapycnal_flux)
end

@inline function triad_Sx(ix, jx, kx, iz, jz, kz, grid, buoyancy, tracers)
    bx = ∂x_b(ix, jx, kx, grid, buoyancy, tracers)
    bz = ∂z_b(iz, jz, kz, grid, buoyancy, tracers)
    bz = max(bz, zero(grid))
    return ifelse(bz == 0, zero(grid), - bx / bz)
end

@inline function triad_Sy(iy, jy, ky, iz, jz, kz, grid, buoyancy, tracers)
    by = ∂y_b(iy, jy, ky, grid, buoyancy, tracers)
    bz = ∂z_b(iz, jz, kz, grid, buoyancy, tracers)
    bz = max(bz, zero(grid))
    return ifelse(bz == 0, zero(grid), - by / bz)
end

@inline Sx⁺⁺(i, j, k, grid, buoyancy, tracers) = triad_Sx(i+1, j, k, i, j, k+1, grid, buoyancy, tracers)
@inline Sx⁺⁻(i, j, k, grid, buoyancy, tracers) = triad_Sx(i+1, j, k, i, j, k,   grid, buoyancy, tracers)
@inline Sx⁻⁺(i, j, k, grid, buoyancy, tracers) = triad_Sx(i,   j, k, i, j, k+1, grid, buoyancy, tracers)
@inline Sx⁻⁻(i, j, k, grid, buoyancy, tracers) = triad_Sx(i,   j, k, i, j, k,   grid, buoyancy, tracers)

@inline Sy⁺⁺(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j+1, k, i, j, k+1, grid, buoyancy, tracers)
@inline Sy⁺⁻(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j+1, k, i, j, k,   grid, buoyancy, tracers)
@inline Sy⁻⁺(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j,   k, i, j, k+1, grid, buoyancy, tracers)
@inline Sy⁻⁻(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j,   k, i, j, k,   grid, buoyancy, tracers)

# Fallback, we cannot rotate the fluxes if we do not at least have two active tracers!
@inline function rotated_div_Uc(i, j, k, grid, scheme, U, c, buoyancy, tracers)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _rotated_advective_tracer_flux_x, scheme, buoyancy, tracers, U, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, _rotated_advective_tracer_flux_y, scheme, buoyancy, tracers, U, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, _rotated_advective_tracer_flux_z, scheme, buoyancy, tracers, U, c))
end

@inline _rotated_advective_tracer_flux_x(i, j, k, grid, scheme, b, C, U, args...) = _advective_tracer_flux_x(i, j, k, grid, scheme, U.u, args...)
@inline _rotated_advective_tracer_flux_y(i, j, k, grid, scheme, b, C, U, args...) = _advective_tracer_flux_y(i, j, k, grid, scheme, U.v, args...)
@inline _rotated_advective_tracer_flux_z(i, j, k, grid, scheme, b, C, U, args...) = _advective_tracer_flux_z(i, j, k, grid, scheme, U.w, args...)

@inline function _rotated_advective_tracer_flux_x(i, j, k, grid, scheme::RotatedAdvection, buoyancy, tracers, U, c)

    upwind_scheme   = scheme.upwind_scheme
    centered_scheme = z_advection(upwind_scheme).advecting_velocity_scheme

    𝒜x = _advective_tracer_flux_x(i, j, k, grid, upwind_scheme, U.u, c)

    𝒜z⁺⁺ = _advective_tracer_flux_z(i-1, j, k+1, grid, upwind_scheme, U.w, c)
    𝒜z⁺⁻ = _advective_tracer_flux_z(i-1, j, k,   grid, upwind_scheme, U.w, c)
    𝒜z⁻⁺ = _advective_tracer_flux_z(i,   j, k+1, grid, upwind_scheme, U.w, c)
    𝒜z⁻⁻ = _advective_tracer_flux_z(i,   j, k,   grid, upwind_scheme, U.w, c)

    𝒞z⁺⁺ = _advective_tracer_flux_z(i-1, j, k+1, grid, centered_scheme, U.w, c)
    𝒞z⁺⁻ = _advective_tracer_flux_z(i-1, j, k,   grid, centered_scheme, U.w, c)
    𝒞z⁻⁺ = _advective_tracer_flux_z(i,   j, k+1, grid, centered_scheme, U.w, c)
    𝒞z⁻⁻ = _advective_tracer_flux_z(i,   j, k,   grid, centered_scheme, U.w, c)

    𝒟z⁺⁺ = 𝒜z⁺⁺ - 𝒞z⁺⁺
    𝒟z⁺⁻ = 𝒜z⁺⁻ - 𝒞z⁺⁻
    𝒟z⁻⁺ = 𝒜z⁻⁺ - 𝒞z⁻⁺
    𝒟z⁻⁻ = 𝒜z⁻⁻ - 𝒞z⁻⁻

    R₁₃_∂z_c⁻ = (Sx⁺⁺(i-1, j, k, grid, buoyancy, tracers) * 𝒟z⁺⁺ +
                 Sx⁺⁻(i-1, j, k, grid, buoyancy, tracers) * 𝒟z⁺⁻ +
                 Sx⁻⁺(i,   j, k, grid, buoyancy, tracers) * 𝒟z⁻⁺ +
                 Sx⁻⁻(i,   j, k, grid, buoyancy, tracers) * 𝒟z⁻⁻) / 4

    return 𝒜x + R₁₃_∂z_c⁻
end

@inline function _rotated_advective_tracer_flux_y(i, j, k, grid, scheme::RotatedAdvection, buoyancy, tracers, U, c)

    upwind_scheme   = scheme.upwind_scheme
    centered_scheme = z_advection(upwind_scheme).advecting_velocity_scheme

    𝒜y = _advective_tracer_flux_y(i, j, k, grid, upwind_scheme, U.v, c)

    𝒜z⁺⁺ = _advective_tracer_flux_z(i, j-1, k+1, grid, upwind_scheme, U.w, c)
    𝒜z⁺⁻ = _advective_tracer_flux_z(i, j-1, k,   grid, upwind_scheme, U.w, c)
    𝒜z⁻⁺ = _advective_tracer_flux_z(i, j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜z⁻⁻ = _advective_tracer_flux_z(i, j,   k,   grid, upwind_scheme, U.w, c)

    𝒞z⁺⁺ = _advective_tracer_flux_z(i, j-1, k+1, grid, centered_scheme, U.w, c)
    𝒞z⁺⁻ = _advective_tracer_flux_z(i, j-1, k,   grid, centered_scheme, U.w, c)
    𝒞z⁻⁺ = _advective_tracer_flux_z(i, j,   k+1, grid, centered_scheme, U.w, c)
    𝒞z⁻⁻ = _advective_tracer_flux_z(i, j,   k,   grid, centered_scheme, U.w, c)

    𝒟z⁺⁺ = 𝒜z⁺⁺ - 𝒞z⁺⁺
    𝒟z⁺⁻ = 𝒜z⁺⁻ - 𝒞z⁺⁻
    𝒟z⁻⁺ = 𝒜z⁻⁺ - 𝒞z⁻⁺
    𝒟z⁻⁻ = 𝒜z⁻⁻ - 𝒞z⁻⁻

    R₂₃_∂z_c⁻ = (Sy⁺⁺(i-1, j, k, grid, buoyancy, tracers) * 𝒟z⁺⁺ +
                 Sy⁺⁻(i-1, j, k, grid, buoyancy, tracers) * 𝒟z⁺⁻ +
                 Sy⁻⁺(i,   j, k, grid, buoyancy, tracers) * 𝒟z⁻⁺ +
                 Sy⁻⁻(i,   j, k, grid, buoyancy, tracers) * 𝒟z⁻⁻) / 4

    return 𝒜y + R₂₃_∂z_c⁻
end

@inline function _rotated_advective_tracer_flux_z(i, j, k, grid, scheme::RotatedAdvection, buoyancy, tracers, U, c)

    upwind_scheme   = scheme.upwind_scheme
    centered_scheme = z_advection(upwind_scheme).advecting_velocity_scheme
   
    𝒜z = _advective_tracer_flux_z(i, j, k, grid, upwind_scheme, U.w, c)
    𝒞z = _advective_tracer_flux_z(i, j, k, grid, centered_scheme, U.w, c)

    𝒜x⁺⁺ = _advective_tracer_flux_x(i+1, j, k-1, grid, upwind_scheme, U.u, c)
    𝒜x⁺⁻ = _advective_tracer_flux_x(i+1, j, k,   grid, upwind_scheme, U.u, c)
    𝒜x⁻⁺ = _advective_tracer_flux_x(i,   j, k-1, grid, upwind_scheme, U.u, c)
    𝒜x⁻⁻ = _advective_tracer_flux_x(i,   j, k,   grid, upwind_scheme, U.u, c)

    𝒞x⁺⁺ = _advective_tracer_flux_x(i+1, j, k-1, grid, centered_scheme, U.u, c)
    𝒞x⁺⁻ = _advective_tracer_flux_x(i+1, j, k,   grid, centered_scheme, U.u, c)
    𝒞x⁻⁺ = _advective_tracer_flux_x(i,   j, k-1, grid, centered_scheme, U.u, c)
    𝒞x⁻⁻ = _advective_tracer_flux_x(i,   j, k,   grid, centered_scheme, U.u, c)

    𝒜y⁺⁺ = _advective_tracer_flux_y(i, j+1, k-1, grid, upwind_scheme, U.v, c)
    𝒜y⁺⁻ = _advective_tracer_flux_y(i, j+1, k,   grid, upwind_scheme, U.v, c)
    𝒜y⁻⁺ = _advective_tracer_flux_y(i, j,   k-1, grid, upwind_scheme, U.v, c)
    𝒜y⁻⁻ = _advective_tracer_flux_y(i, j,   k,   grid, upwind_scheme, U.v, c)

    𝒞y⁺⁺ = _advective_tracer_flux_y(i, j+1, k-1, grid, centered_scheme, U.v, c)
    𝒞y⁺⁻ = _advective_tracer_flux_y(i, j+1, k,   grid, centered_scheme, U.v, c)
    𝒞y⁻⁺ = _advective_tracer_flux_y(i, j,   k-1, grid, centered_scheme, U.v, c)
    𝒞y⁻⁻ = _advective_tracer_flux_y(i, j,   k,   grid, centered_scheme, U.v, c)

    𝒟z = 𝒜z - 𝒞z

    𝒟x⁺⁺ = 𝒜x⁺⁺ - 𝒞x⁺⁺
    𝒟x⁺⁻ = 𝒜x⁺⁻ - 𝒞x⁺⁻
    𝒟x⁻⁺ = 𝒜x⁻⁺ - 𝒞x⁻⁺
    𝒟x⁻⁻ = 𝒜x⁻⁻ - 𝒞x⁻⁻

    𝒟y⁺⁺ = 𝒜y⁺⁺ - 𝒞y⁺⁺
    𝒟y⁺⁻ = 𝒜y⁺⁻ - 𝒞y⁺⁻
    𝒟y⁻⁺ = 𝒜y⁻⁺ - 𝒞y⁻⁺
    𝒟y⁻⁻ = 𝒜y⁻⁻ - 𝒞y⁻⁻

    R₃₁_∂z_c = (Sx⁻⁻(i, j, k,   grid, buoyancy, tracers) * 𝒟x⁻⁻ +
                Sx⁺⁻(i, j, k,   grid, buoyancy, tracers) * 𝒟x⁺⁻ +
                Sx⁻⁺(i, j, k-1, grid, buoyancy, tracers) * 𝒟x⁻⁺ +
                Sx⁺⁺(i, j, k-1, grid, buoyancy, tracers) * 𝒟x⁺⁺) / 4

    R₃₂_∂z_c = (Sy⁻⁻(i, j, k,   grid, buoyancy, tracers) * 𝒟y⁻⁻ +
                Sy⁺⁻(i, j, k,   grid, buoyancy, tracers) * 𝒟y⁺⁻ +
                Sy⁻⁺(i, j, k-1, grid, buoyancy, tracers) * 𝒟y⁻⁺ +
                Sy⁺⁺(i, j, k-1, grid, buoyancy, tracers) * 𝒟y⁺⁺) / 4

    Sx² = (Sx⁻⁻(i, j, k,   grid, buoyancy, tracers))^2 +
          (Sx⁺⁻(i, j, k,   grid, buoyancy, tracers))^2 +
          (Sx⁻⁺(i, j, k-1, grid, buoyancy, tracers))^2 +
          (Sx⁺⁺(i, j, k-1, grid, buoyancy, tracers))^2 / 4 

    Sy² = (Sy⁻⁻(i, j, k,   grid, buoyancy, tracers))^2 +
          (Sy⁺⁻(i, j, k,   grid, buoyancy, tracers))^2 +
          (Sy⁻⁺(i, j, k-1, grid, buoyancy, tracers))^2 +
          (Sy⁺⁺(i, j, k-1, grid, buoyancy, tracers))^2 / 4 

    return 𝒜z # 𝒞z + R₃₁_∂z_c + R₃₂_∂z_c + (Sx² + Sy²) * 𝒟z # 
end

# @inline function rotated_div_Uc(i, j, k, grid, scheme::RotatedAdvection, U, c, buoyancy, tracers)
    
#     upwind_scheme = scheme.upwind_scheme
#     centered_scheme_x = x_advection(upwind_scheme).advecting_velocity_scheme
#     centered_scheme_y = y_advection(upwind_scheme).advecting_velocity_scheme
#     centered_scheme_z = z_advection(upwind_scheme).advecting_velocity_scheme

#     # Total advective fluxes
#     𝒜xᶠᶜᶜ⁺ = _advective_tracer_flux_x(i+1, j, k, grid, upwind_scheme, U.u, c)
#     𝒜xᶠᶜᶜ⁻ = _advective_tracer_flux_x(i,   j, k, grid, upwind_scheme, U.u, c)
#     𝒞xᶠᶜᶜ⁺ = _advective_tracer_flux_x(i+1, j, k, grid, centered_scheme_x, U.u, c)
#     𝒞xᶠᶜᶜ⁻ = _advective_tracer_flux_x(i,   j, k, grid, centered_scheme_x, U.u, c)

#     𝒜xᶜᶠᶜ⁺ = ℑxyᶜᶠᵃ(i+1, j, k, grid, _advective_tracer_flux_x, upwind_scheme, U.u, c)
#     𝒜xᶜᶠᶜ⁻ = ℑxyᶜᶠᵃ(i,   j, k, grid, _advective_tracer_flux_x, upwind_scheme, U.u, c)
#     𝒜xᶜᶜᶠ⁺ = ℑxzᶜᵃᶠ(i+1, j, k, grid, _advective_tracer_flux_x, upwind_scheme, U.u, c)
#     𝒜xᶜᶜᶠ⁻ = ℑxzᶜᵃᶠ(i,   j, k, grid, _advective_tracer_flux_x, upwind_scheme, U.u, c)

#     𝒞xᶜᶠᶜ⁺ = ℑxyᶜᶠᵃ(i+1, j, k, grid, _advective_tracer_flux_x, centered_scheme_x, U.u, c)
#     𝒞xᶜᶠᶜ⁻ = ℑxyᶜᶠᵃ(i,   j, k, grid, _advective_tracer_flux_x, centered_scheme_x, U.u, c)
#     𝒞xᶜᶜᶠ⁺ = ℑxzᶜᵃᶠ(i+1, j, k, grid, _advective_tracer_flux_x, centered_scheme_x, U.u, c)
#     𝒞xᶜᶜᶠ⁻ = ℑxzᶜᵃᶠ(i,   j, k, grid, _advective_tracer_flux_x, centered_scheme_x, U.u, c)

#     𝒜yᶜᶠᶜ⁺ = _advective_tracer_flux_y(i, j+1, k, grid, upwind_scheme, U.v, c)
#     𝒜yᶜᶠᶜ⁻ = _advective_tracer_flux_y(i, j,   k, grid, upwind_scheme, U.v, c)
#     𝒞yᶜᶠᶜ⁺ = _advective_tracer_flux_y(i, j+1, k, grid, centered_scheme_y, U.v, c)
#     𝒞yᶜᶠᶜ⁻ = _advective_tracer_flux_y(i, j,   k, grid, centered_scheme_y, U.v, c)

#     𝒜yᶠᶜᶜ⁺ = ℑxyᶠᶜᵃ(i, j+1, k, grid, _advective_tracer_flux_y, upwind_scheme, U.v, c)
#     𝒜yᶠᶜᶜ⁻ = ℑxyᶠᶜᵃ(i, j,   k, grid, _advective_tracer_flux_y, upwind_scheme, U.v, c)
#     𝒜yᶜᶜᶠ⁺ = ℑyzᵃᶜᶠ(i, j+1, k, grid, _advective_tracer_flux_y, upwind_scheme, U.v, c)
#     𝒜yᶜᶜᶠ⁻ = ℑyzᵃᶜᶠ(i, j,   k, grid, _advective_tracer_flux_y, upwind_scheme, U.v, c)

#     𝒞yᶠᶜᶜ⁺ = ℑxyᶠᶜᵃ(i, j+1, k, grid, _advective_tracer_flux_y, centered_scheme_y, U.v, c)
#     𝒞yᶠᶜᶜ⁻ = ℑxyᶠᶜᵃ(i, j,   k, grid, _advective_tracer_flux_y, centered_scheme_y, U.v, c)
#     𝒞yᶜᶜᶠ⁺ = ℑyzᵃᶜᶠ(i, j+1, k, grid, _advective_tracer_flux_y, centered_scheme_y, U.v, c)
#     𝒞yᶜᶜᶠ⁻ = ℑyzᵃᶜᶠ(i, j,   k, grid, _advective_tracer_flux_y, centered_scheme_y, U.v, c)

#     𝒜zᶜᶜᶠ⁺ = _advective_tracer_flux_z(i, j, k+1, grid, upwind_scheme, U.w, c)
#     𝒜zᶜᶜᶠ⁻ = _advective_tracer_flux_z(i, j, k,   grid, upwind_scheme, U.w, c)
#     𝒞zᶜᶜᶠ⁺ = _advective_tracer_flux_z(i, j, k+1, grid, centered_scheme_z, U.w, c)
#     𝒞zᶜᶜᶠ⁻ = _advective_tracer_flux_z(i, j, k,   grid, centered_scheme_z, U.w, c)

#     𝒜zᶠᶜᶜ⁺ = ℑxzᶠᵃᶜ(i, j, k+1, grid, _advective_tracer_flux_z, upwind_scheme, U.w, c)
#     𝒜zᶠᶜᶜ⁻ = ℑxzᶠᵃᶜ(i, j, k,   grid, _advective_tracer_flux_z, upwind_scheme, U.w, c)
#     𝒜zᶜᶠᶜ⁺ = ℑyzᵃᶠᶜ(i, j, k+1, grid, _advective_tracer_flux_z, upwind_scheme, U.w, c)
#     𝒜zᶜᶠᶜ⁻ = ℑyzᵃᶠᶜ(i, j, k,   grid, _advective_tracer_flux_z, upwind_scheme, U.w, c)

#     𝒞zᶠᶜᶜ⁺ = ℑxzᶠᵃᶜ(i, j, k+1, grid, _advective_tracer_flux_z, centered_scheme_z, U.w, c)
#     𝒞zᶠᶜᶜ⁻ = ℑxzᶠᵃᶜ(i, j, k,   grid, _advective_tracer_flux_z, centered_scheme_z, U.w, c)
#     𝒞zᶜᶠᶜ⁺ = ℑyzᵃᶠᶜ(i, j, k+1, grid, _advective_tracer_flux_z, centered_scheme_z, U.w, c)
#     𝒞zᶜᶠᶜ⁻ = ℑyzᵃᶠᶜ(i, j, k,   grid, _advective_tracer_flux_z, centered_scheme_z, U.w, c)

#     𝒟xᶠᶜᶜ⁺ = 𝒜xᶠᶜᶜ⁺ - 𝒞xᶠᶜᶜ⁺
#     𝒟xᶠᶜᶜ⁻ = 𝒜xᶠᶜᶜ⁻ - 𝒞xᶠᶜᶜ⁻
#     𝒟xᶜᶠᶜ⁺ = 𝒜xᶜᶠᶜ⁺ - 𝒞xᶜᶠᶜ⁺
#     𝒟xᶜᶠᶜ⁻ = 𝒜xᶜᶠᶜ⁻ - 𝒞xᶜᶠᶜ⁻
#     𝒟xᶜᶜᶠ⁺ = 𝒜xᶜᶜᶠ⁺ - 𝒞xᶜᶜᶠ⁺
#     𝒟xᶜᶜᶠ⁻ = 𝒜xᶜᶜᶠ⁻ - 𝒞xᶜᶜᶠ⁻

#     𝒟yᶠᶜᶜ⁺ = 𝒜yᶠᶜᶜ⁺ - 𝒞yᶠᶜᶜ⁺
#     𝒟yᶠᶜᶜ⁻ = 𝒜yᶠᶜᶜ⁻ - 𝒞yᶠᶜᶜ⁻
#     𝒟yᶜᶠᶜ⁺ = 𝒜yᶜᶠᶜ⁺ - 𝒞yᶜᶠᶜ⁺
#     𝒟yᶜᶠᶜ⁻ = 𝒜yᶜᶠᶜ⁻ - 𝒞yᶜᶠᶜ⁻
#     𝒟yᶜᶜᶠ⁺ = 𝒜yᶜᶜᶠ⁺ - 𝒞yᶜᶜᶠ⁺
#     𝒟yᶜᶜᶠ⁻ = 𝒜yᶜᶜᶠ⁻ - 𝒞yᶜᶜᶠ⁻

#     𝒟zᶠᶜᶜ⁺ = 𝒜zᶠᶜᶜ⁺ - 𝒞zᶠᶜᶜ⁺
#     𝒟zᶠᶜᶜ⁻ = 𝒜zᶠᶜᶜ⁻ - 𝒞zᶠᶜᶜ⁻
#     𝒟zᶜᶠᶜ⁺ = 𝒜zᶜᶠᶜ⁺ - 𝒞zᶜᶠᶜ⁺
#     𝒟zᶜᶠᶜ⁻ = 𝒜zᶜᶠᶜ⁻ - 𝒞zᶜᶠᶜ⁻
#     𝒟zᶜᶜᶠ⁺ = 𝒜zᶜᶜᶠ⁺ - 𝒞zᶜᶜᶠ⁺
#     𝒟zᶜᶜᶠ⁻ = 𝒜zᶜᶜᶠ⁻ - 𝒞zᶜᶜᶠ⁻


#     # X-Fluxes
#     R₁₁_∂x_c⁺ = 𝒜xᶠᶜᶜ⁺
#     R₁₁_∂x_c⁻ = 𝒜xᶠᶜᶜ⁻

#     R₁₃_∂z_c⁺ = (Sx⁺⁺(i,   j, k, grid, buoyancy, tracers) * ∂zᶜᶜᶠ(i,   j, k+1, grid, c) +
#                  Sx⁺⁻(i,   j, k, grid, buoyancy, tracers) * ∂zᶜᶜᶠ(i,   j, k,   grid, c) +
#                  Sx⁻⁺(i+1, j, k, grid, buoyancy, tracers) * ∂zᶜᶜᶠ(i+1, j, k+1, grid, c) +
#                  Sx⁻⁻(i+1, j, k, grid, buoyancy, tracers) * ∂zᶜᶜᶠ(i+1, j, k,   grid, c)) / 4

# end


#     # TODO: make this a parameter?
#     ϵ = scheme.percentage_of_diapycnal_flux
#     Smax = scheme.maximum_slope

#     # Elements of the rotation tensor
#     R₁₁⁺, R₁₂⁺, R₁₃⁺ = rotation_tensorᶠᶜᶜ(i+1, j, k, grid, buoyancy, tracers, Smax, ϵ)
#     R₁₁⁻, R₁₂⁻, R₁₃⁻ = rotation_tensorᶠᶜᶜ(i,   j, k, grid, buoyancy, tracers, Smax, ϵ)

#     R₂₁⁺, R₂₂⁺, R₂₃⁺ = rotation_tensorᶜᶠᶜ(i, j+1, k, grid, buoyancy, tracers, Smax, ϵ)
#     R₂₁⁻, R₂₂⁻, R₂₃⁻ = rotation_tensorᶜᶠᶜ(i, j,   k, grid, buoyancy, tracers, Smax, ϵ)
    
#     R₃₁⁺, R₃₂⁺, R₃₃⁺ = rotation_tensorᶜᶜᶠ(i, j, k+1, grid, buoyancy, tracers, Smax, ϵ)
#     R₃₁⁻, R₃₂⁻, R₃₃⁻ = rotation_tensorᶜᶜᶠ(i, j, k,   grid, buoyancy, tracers, Smax, ϵ)

#     # Renormalize the rotated fluxes based on the α
#     ℛx⁺ = R₁₁⁺ * 𝒟xᶠᶜᶜ⁺ + R₁₂⁺ * 𝒟yᶠᶜᶜ⁺ + R₁₃⁺ * 𝒟zᶠᶜᶜ⁺
#     ℛx⁻ = R₁₁⁻ * 𝒟xᶠᶜᶜ⁻ + R₁₂⁻ * 𝒟yᶠᶜᶜ⁻ + R₁₃⁻ * 𝒟zᶠᶜᶜ⁻
#     ℛy⁺ = R₂₁⁺ * 𝒟xᶜᶠᶜ⁺ + R₂₂⁺ * 𝒟yᶜᶠᶜ⁺ + R₂₃⁺ * 𝒟zᶜᶠᶜ⁺
#     ℛy⁻ = R₂₁⁻ * 𝒟xᶜᶠᶜ⁻ + R₂₂⁻ * 𝒟yᶜᶠᶜ⁻ + R₂₃⁻ * 𝒟zᶜᶠᶜ⁻
#     ℛz⁺ = R₃₁⁺ * 𝒟xᶜᶜᶠ⁺ + R₃₂⁺ * 𝒟yᶜᶜᶠ⁺ + R₃₃⁺ * 𝒟zᶜᶜᶠ⁺
#     ℛz⁻ = R₃₁⁻ * 𝒟xᶜᶜᶠ⁻ + R₃₂⁻ * 𝒟yᶜᶜᶠ⁻ + R₃₃⁻ * 𝒟zᶜᶜᶠ⁻

#     α = scheme.rotation_percentage

#     # Fluxes
#     Fx⁺ = 𝒞xᶠᶜᶜ⁺ + α + ℛx⁺ + (1 - α) * 𝒟xᶠᶜᶜ⁺
#     Fx⁻ = 𝒞xᶠᶜᶜ⁻ + α + ℛx⁻ + (1 - α) * 𝒟xᶠᶜᶜ⁻
#     Fy⁺ = 𝒞yᶜᶠᶜ⁺ + α + ℛz⁺ + (1 - α) * 𝒟xᶜᶠᶜ⁺
#     Fy⁻ = 𝒞yᶜᶠᶜ⁻ + α + ℛz⁻ + (1 - α) * 𝒟xᶜᶠᶜ⁻
#     Fz⁺ = 𝒞zᶜᶜᶠ⁺ + α + ℛz⁺ + (1 - α) * 𝒟xᶜᶜᶠ⁺
#     Fz⁻ = 𝒞zᶜᶜᶠ⁻ + α + ℛz⁻ + (1 - α) * 𝒟xᶜᶜᶠ⁻
        
#     return 1 / Vᶜᶜᶜ(i, j, k, grid) * (Fx⁺ - Fx⁻ + Fy⁺ - Fy⁻ + Fz⁺ - Fz⁻)
# end

# @inline function rotation_tensorᶠᶜᶜ(i, j, k, grid, buoyancy, tracers, Smax, ϵ)
#     bx =   ∂x_b(i, j, k, grid,       buoyancy, tracers) 
#     by = ℑxyᶜᶠᵃ(i, j, k, grid, ∂y_b, buoyancy, tracers) 
#     bz = ℑxzᶜᵃᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers) 
#     S  = bx^2 + by^2 + bz^2
#     Sx = abs(bx / bz)
#     Sy = abs(by / bz)
#     condition = (Sx < Smax) & (Sy < Smax) & (S > 0) # Tapering

#     R₁₁ = ifelse(condition,   (by^2 + bz^2 + ϵ * bx^2) / S, one(grid)) 
#     R₁₂ = ifelse(condition,        ((ϵ - 1) * bx * by) / S, zero(grid)) 
#     R₁₃ = ifelse(condition,        ((ϵ - 1) * bx * bz) / S, zero(grid))

#     return R₁₁, R₁₂, R₁₃
# end

# @inline function rotation_tensorᶜᶠᶜ(i, j, k, grid, buoyancy, tracers, Smax, ϵ)
#     bx = ℑxyᶜᶠᵃ(i, j, k, grid, ∂x_b, buoyancy, tracers) 
#     by =   ∂y_b(i, j, k, grid,       buoyancy, tracers) 
#     bz = ℑyzᵃᶜᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers) 
#     S  = bx^2 + by^2 + bz^2
#     Sx = abs(bx / bz)
#     Sy = abs(by / bz)
#     condition = (Sx < Smax) & (Sy < Smax) & (S > 0) # Tapering

#     R₂₁ = ifelse(condition,      ((ϵ - 1) * by * bx) / S, zero(grid)) 
#     R₂₂ = ifelse(condition, (bx^2 + bz^2 + ϵ * by^2) / S, one(grid)) 
#     R₂₃ = ifelse(condition,      ((ϵ - 1) * by * bz) / S, zero(grid))

#     return R₂₁, R₂₂, R₂₃
# end

# @inline function rotation_tensorᶜᶜᶠ(i, j, k, grid, buoyancy, tracers, Smax, ϵ)
#     bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers) 
#     by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers) 
#     bz =   ∂z_b(i, j, k, grid,       buoyancy, tracers) 
#     S  = bx^2 + by^2 + bz^2
#     Sx = abs(bx / bz)
#     Sy = abs(by / bz)
#     condition = (Sx < Smax) & (Sy < Smax) & (S > 0) # Tapering

#     R₃₁ = ifelse(condition,      ((ϵ - 1) * bz * bx) / S, zero(grid)) 
#     R₃₂ = ifelse(condition,      ((ϵ - 1) * bz * by) / S, zero(grid))
#     R₃₃ = ifelse(condition, (bx^2 + by^2 + ϵ * bz^2) / S, one(grid)) 

#     return R₃₁, R₃₂, R₃₃
# end