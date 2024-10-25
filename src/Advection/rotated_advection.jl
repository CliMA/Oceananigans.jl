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

# Fallback, we cannot rotate the fluxes if we do not at least have two active tracers!
@inline rotated_div_Uc(i, j, k, grid, scheme, U, c, buoyancy, tracers) = div_Uc(i, j, k, grid, scheme, U, c)

@inline function rotated_div_Uc(i, j, k, grid, scheme::RotatedAdvection, U, c, buoyancy, tracers)
    
    upwind_scheme = scheme.upwind_scheme
    centered_scheme_x = x_advection(upwind_scheme).advecting_velocity_scheme
    centered_scheme_y = y_advection(upwind_scheme).advecting_velocity_scheme
    centered_scheme_z = z_advection(upwind_scheme).advecting_velocity_scheme

    # Total advective fluxes
    𝒜xᶠᶜᶜ⁺ = _advective_tracer_flux_x(i+1, j, k, grid, upwind_scheme, U.u, c)
    𝒜xᶠᶜᶜ⁻ = _advective_tracer_flux_x(i,   j, k, grid, upwind_scheme, U.u, c)
    𝒞xᶠᶜᶜ⁺ = _advective_tracer_flux_x(i+1, j, k, grid, centered_scheme_x, U.u, c)
    𝒞xᶠᶜᶜ⁻ = _advective_tracer_flux_x(i,   j, k, grid, centered_scheme_x, U.u, c)

    𝒜xᶜᶠᶜ⁺ = ℑxyᶜᶠᵃ(i+1, j, k, grid, _advective_tracer_flux_x, upwind_scheme, U.u, c)
    𝒜xᶜᶠᶜ⁻ = ℑxyᶜᶠᵃ(i,   j, k, grid, _advective_tracer_flux_x, upwind_scheme, U.u, c)
    𝒜xᶜᶜᶠ⁺ = ℑxzᶜᵃᶠ(i+1, j, k, grid, _advective_tracer_flux_x, upwind_scheme, U.u, c)
    𝒜xᶜᶜᶠ⁻ = ℑxzᶜᵃᶠ(i,   j, k, grid, _advective_tracer_flux_x, upwind_scheme, U.u, c)

    𝒞xᶜᶠᶜ⁺ = ℑxyᶜᶠᵃ(i+1, j, k, grid, _advective_tracer_flux_x, centered_scheme_x, U.u, c)
    𝒞xᶜᶠᶜ⁻ = ℑxyᶜᶠᵃ(i,   j, k, grid, _advective_tracer_flux_x, centered_scheme_x, U.u, c)
    𝒞xᶜᶜᶠ⁺ = ℑxzᶜᵃᶠ(i+1, j, k, grid, _advective_tracer_flux_x, centered_scheme_x, U.u, c)
    𝒞xᶜᶜᶠ⁻ = ℑxzᶜᵃᶠ(i,   j, k, grid, _advective_tracer_flux_x, centered_scheme_x, U.u, c)

    𝒜yᶜᶠᶜ⁺ = _advective_tracer_flux_y(i, j+1, k, grid, upwind_scheme, U.v, c)
    𝒜yᶜᶠᶜ⁻ = _advective_tracer_flux_y(i, j,   k, grid, upwind_scheme, U.v, c)
    𝒞yᶜᶠᶜ⁺ = _advective_tracer_flux_y(i, j+1, k, grid, centered_scheme_y, U.v, c)
    𝒞yᶜᶠᶜ⁻ = _advective_tracer_flux_y(i, j,   k, grid, centered_scheme_y, U.v, c)

    𝒜yᶠᶜᶜ⁺ = ℑxyᶠᶜᵃ(i, j+1, k, grid, _advective_tracer_flux_y, upwind_scheme, U.v, c)
    𝒜yᶠᶜᶜ⁻ = ℑxyᶠᶜᵃ(i, j,   k, grid, _advective_tracer_flux_y, upwind_scheme, U.v, c)
    𝒜yᶜᶜᶠ⁺ = ℑyzᵃᶜᶠ(i, j+1, k, grid, _advective_tracer_flux_y, upwind_scheme, U.v, c)
    𝒜yᶜᶜᶠ⁻ = ℑyzᵃᶜᶠ(i, j,   k, grid, _advective_tracer_flux_y, upwind_scheme, U.v, c)

    𝒞yᶠᶜᶜ⁺ = ℑxyᶠᶜᵃ(i, j+1, k, grid, _advective_tracer_flux_y, centered_scheme_y, U.v, c)
    𝒞yᶠᶜᶜ⁻ = ℑxyᶠᶜᵃ(i, j,   k, grid, _advective_tracer_flux_y, centered_scheme_y, U.v, c)
    𝒞yᶜᶜᶠ⁺ = ℑyzᵃᶜᶠ(i, j+1, k, grid, _advective_tracer_flux_y, centered_scheme_y, U.v, c)
    𝒞yᶜᶜᶠ⁻ = ℑyzᵃᶜᶠ(i, j,   k, grid, _advective_tracer_flux_y, centered_scheme_y, U.v, c)

    𝒜zᶜᶜᶠ⁺ = _advective_tracer_flux_z(i, j, k+1, grid, upwind_scheme, U.w, c)
    𝒜zᶜᶜᶠ⁻ = _advective_tracer_flux_z(i, j, k,   grid, upwind_scheme, U.w, c)
    𝒞zᶜᶜᶠ⁺ = _advective_tracer_flux_z(i, j, k+1, grid, centered_scheme_z, U.w, c)
    𝒞zᶜᶜᶠ⁻ = _advective_tracer_flux_z(i, j, k,   grid, centered_scheme_z, U.w, c)

    𝒜zᶠᶜᶜ⁺ = ℑxzᶠᵃᶜ(i, j, k+1, grid, _advective_tracer_flux_z, upwind_scheme, U.w, c)
    𝒜zᶠᶜᶜ⁻ = ℑxzᶠᵃᶜ(i, j, k,   grid, _advective_tracer_flux_z, upwind_scheme, U.w, c)
    𝒜zᶜᶠᶜ⁺ = ℑyzᵃᶠᶜ(i, j, k+1, grid, _advective_tracer_flux_z, upwind_scheme, U.w, c)
    𝒜zᶜᶠᶜ⁻ = ℑyzᵃᶠᶜ(i, j, k,   grid, _advective_tracer_flux_z, upwind_scheme, U.w, c)

    𝒞zᶠᶜᶜ⁺ = ℑxzᶠᵃᶜ(i, j, k+1, grid, _advective_tracer_flux_z, centered_scheme_z, U.w, c)
    𝒞zᶠᶜᶜ⁻ = ℑxzᶠᵃᶜ(i, j, k,   grid, _advective_tracer_flux_z, centered_scheme_z, U.w, c)
    𝒞zᶜᶠᶜ⁺ = ℑyzᵃᶠᶜ(i, j, k+1, grid, _advective_tracer_flux_z, centered_scheme_z, U.w, c)
    𝒞zᶜᶠᶜ⁻ = ℑyzᵃᶠᶜ(i, j, k,   grid, _advective_tracer_flux_z, centered_scheme_z, U.w, c)

    𝒟xᶠᶜᶜ⁺ = 𝒜xᶠᶜᶜ⁺ - 𝒞xᶠᶜᶜ⁺
    𝒟xᶠᶜᶜ⁻ = 𝒜xᶠᶜᶜ⁻ - 𝒞xᶠᶜᶜ⁻
    𝒟xᶜᶠᶜ⁺ = 𝒜xᶜᶠᶜ⁺ - 𝒞xᶜᶠᶜ⁺
    𝒟xᶜᶠᶜ⁻ = 𝒜xᶜᶠᶜ⁻ - 𝒞xᶜᶠᶜ⁻
    𝒟xᶜᶜᶠ⁺ = 𝒜xᶜᶜᶠ⁺ - 𝒞xᶜᶜᶠ⁺
    𝒟xᶜᶜᶠ⁻ = 𝒜xᶜᶜᶠ⁻ - 𝒞xᶜᶜᶠ⁻

    𝒟yᶠᶜᶜ⁺ = 𝒜yᶠᶜᶜ⁺ - 𝒞yᶠᶜᶜ⁺
    𝒟yᶠᶜᶜ⁻ = 𝒜yᶠᶜᶜ⁻ - 𝒞yᶠᶜᶜ⁻
    𝒟yᶜᶠᶜ⁺ = 𝒜yᶜᶠᶜ⁺ - 𝒞yᶜᶠᶜ⁺
    𝒟yᶜᶠᶜ⁻ = 𝒜yᶜᶠᶜ⁻ - 𝒞yᶜᶠᶜ⁻
    𝒟yᶜᶜᶠ⁺ = 𝒜yᶜᶜᶠ⁺ - 𝒞yᶜᶜᶠ⁺
    𝒟yᶜᶜᶠ⁻ = 𝒜yᶜᶜᶠ⁻ - 𝒞yᶜᶜᶠ⁻

    𝒟zᶠᶜᶜ⁺ = 𝒜zᶠᶜᶜ⁺ - 𝒞zᶠᶜᶜ⁺
    𝒟zᶠᶜᶜ⁻ = 𝒜zᶠᶜᶜ⁻ - 𝒞zᶠᶜᶜ⁻
    𝒟zᶜᶠᶜ⁺ = 𝒜zᶜᶠᶜ⁺ - 𝒞zᶜᶠᶜ⁺
    𝒟zᶜᶠᶜ⁻ = 𝒜zᶜᶠᶜ⁻ - 𝒞zᶜᶠᶜ⁻
    𝒟zᶜᶜᶠ⁺ = 𝒜zᶜᶜᶠ⁺ - 𝒞zᶜᶜᶠ⁺
    𝒟zᶜᶜᶠ⁻ = 𝒜zᶜᶜᶠ⁻ - 𝒞zᶜᶜᶠ⁻

    # TODO: make this a parameter?
    ϵ = scheme.percentage_of_diapycnal_flux
    Smax = scheme.maximum_slope

    # Elements of the rotation tensor
    R₁₁⁺, R₁₂⁺, R₁₃⁺ = rotation_tensorᶠᶜᶜ(i+1, j, k, grid, buoyancy, tracers, Smax, ϵ)
    R₁₁⁻, R₁₂⁻, R₁₃⁻ = rotation_tensorᶠᶜᶜ(i,   j, k, grid, buoyancy, tracers, Smax, ϵ)

    R₂₁⁺, R₂₂⁺, R₂₃⁺ = rotation_tensorᶜᶠᶜ(i, j+1, k, grid, buoyancy, tracers, Smax, ϵ)
    R₂₁⁻, R₂₂⁻, R₂₃⁻ = rotation_tensorᶜᶠᶜ(i, j,   k, grid, buoyancy, tracers, Smax, ϵ)
    
    R₃₁⁺, R₃₂⁺, R₃₃⁺ = rotation_tensorᶜᶜᶠ(i, j, k+1, grid, buoyancy, tracers, Smax, ϵ)
    R₃₁⁻, R₃₂⁻, R₃₃⁻ = rotation_tensorᶜᶜᶠ(i, j, k,   grid, buoyancy, tracers, Smax, ϵ)

    # Renormalize the rotated fluxes based on the α
    ℛx⁺ = R₁₁⁺ * 𝒟xᶠᶜᶜ⁺ + R₁₂⁺ * 𝒟yᶠᶜᶜ⁺ + R₁₃⁺ * 𝒟zᶠᶜᶜ⁺
    ℛx⁻ = R₁₁⁻ * 𝒟xᶠᶜᶜ⁻ + R₁₂⁻ * 𝒟yᶠᶜᶜ⁻ + R₁₃⁻ * 𝒟zᶠᶜᶜ⁻
    ℛy⁺ = R₂₁⁺ * 𝒟xᶜᶠᶜ⁺ + R₂₂⁺ * 𝒟yᶜᶠᶜ⁺ + R₂₃⁺ * 𝒟zᶜᶠᶜ⁺
    ℛy⁻ = R₂₁⁻ * 𝒟xᶜᶠᶜ⁻ + R₂₂⁻ * 𝒟yᶜᶠᶜ⁻ + R₂₃⁻ * 𝒟zᶜᶠᶜ⁻
    ℛz⁺ = R₃₁⁺ * 𝒟xᶜᶜᶠ⁺ + R₃₂⁺ * 𝒟yᶜᶜᶠ⁺ + R₃₃⁺ * 𝒟zᶜᶜᶠ⁺
    ℛz⁻ = R₃₁⁻ * 𝒟xᶜᶜᶠ⁻ + R₃₂⁻ * 𝒟yᶜᶜᶠ⁻ + R₃₃⁻ * 𝒟zᶜᶜᶠ⁻

    α = scheme.rotation_percentage

    # Fluxes
    Fx⁺ = 𝒞xᶠᶜᶜ⁺ + α + ℛx⁺ + (1 - α) * 𝒟xᶠᶜᶜ⁺
    Fx⁻ = 𝒞xᶠᶜᶜ⁻ + α + ℛx⁻ + (1 - α) * 𝒟xᶠᶜᶜ⁻
    Fy⁺ = 𝒞yᶜᶠᶜ⁺ + α + ℛz⁺ + (1 - α) * 𝒟xᶜᶠᶜ⁺
    Fy⁻ = 𝒞yᶜᶠᶜ⁻ + α + ℛz⁻ + (1 - α) * 𝒟xᶜᶠᶜ⁻
    Fz⁺ = 𝒞zᶜᶜᶠ⁺ + α + ℛz⁺ + (1 - α) * 𝒟xᶜᶜᶠ⁺
    Fz⁻ = 𝒞zᶜᶜᶠ⁻ + α + ℛz⁻ + (1 - α) * 𝒟xᶜᶜᶠ⁻
        
    return 1 / Vᶜᶜᶜ(i, j, k, grid) * (Fx⁺ - Fx⁻ + Fy⁺ - Fy⁻ + Fz⁺ - Fz⁻)
end

@inline function rotation_tensorᶠᶜᶜ(i, j, k, grid, buoyancy, tracers, Smax, ϵ)
    bx =   ∂x_b(i, j, k, grid,       buoyancy, tracers) 
    by = ℑxyᶜᶠᵃ(i, j, k, grid, ∂y_b, buoyancy, tracers) 
    bz = ℑxzᶜᵃᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers) 
    S  = bx^2 + by^2 + bz^2
    Sx = abs(bx / bz)
    Sy = abs(by / bz)
    condition = (Sx < Smax) & (Sy < Smax) & (S > 0) # Tapering

    R₁₁ = ifelse(condition,   (by^2 + bz^2 + ϵ * bx^2) / S, one(grid)) 
    R₁₂ = ifelse(condition,        ((ϵ - 1) * bx * by) / S, zero(grid)) 
    R₁₃ = ifelse(condition,        ((ϵ - 1) * bx * bz) / S, zero(grid))

    return R₁₁, R₁₂, R₁₃
end

@inline function rotation_tensorᶜᶠᶜ(i, j, k, grid, buoyancy, tracers, Smax, ϵ)
    bx = ℑxyᶜᶠᵃ(i, j, k, grid, ∂x_b, buoyancy, tracers) 
    by =   ∂y_b(i, j, k, grid,       buoyancy, tracers) 
    bz = ℑyzᵃᶜᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers) 
    S  = bx^2 + by^2 + bz^2
    Sx = abs(bx / bz)
    Sy = abs(by / bz)
    condition = (Sx < Smax) & (Sy < Smax) & (S > 0) # Tapering

    R₂₁ = ifelse(condition,      ((ϵ - 1) * by * bx) / S, zero(grid)) 
    R₂₂ = ifelse(condition, (bx^2 + bz^2 + ϵ * by^2) / S, one(grid)) 
    R₂₃ = ifelse(condition,      ((ϵ - 1) * by * bz) / S, zero(grid))

    return R₂₁, R₂₂, R₂₃
end

@inline function rotation_tensorᶜᶜᶠ(i, j, k, grid, buoyancy, tracers, Smax, ϵ)
    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers) 
    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers) 
    bz =   ∂z_b(i, j, k, grid,       buoyancy, tracers) 
    S  = bx^2 + by^2 + bz^2
    Sx = abs(bx / bz)
    Sy = abs(by / bz)
    condition = (Sx < Smax) & (Sy < Smax) & (S > 0) # Tapering

    R₃₁ = ifelse(condition,      ((ϵ - 1) * bz * bx) / S, zero(grid)) 
    R₃₂ = ifelse(condition,      ((ϵ - 1) * bz * by) / S, zero(grid))
    R₃₃ = ifelse(condition, (bx^2 + by^2 + ϵ * bz^2) / S, one(grid)) 

    return R₃₁, R₃₂, R₃₃
end