using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans: ∂x_b, ∂y_b, ∂z_b
using Oceananigans.Operators

struct RotatedAdvection{N, FT, U} <: AbstractUpwindBiasedAdvectionScheme{N, FT}
    upwind_scheme :: U
    minimum_rotation_percentage :: FT
    maximum_slope :: FT
end

function RotatedAdvection(upwind_scheme::U;
                          maximum_slope = 1e5, 
                          minimum_rotation_percentage = 0.1) where U 

    FT = eltype(upwind_scheme)
    
    N  = max(required_halo_size_x(upwind_scheme),
             required_halo_size_y(upwind_scheme),
             required_halo_size_z(upwind_scheme))
    
    minimum_rotation_percentage = convert(FT, minimum_rotation_percentage)
    maximum_slope = convert(FT, maximum_slope)

    return RotatedAdvection{N, FT, U}(upwind_scheme, minimum_rotation_percentage, maximum_slope)
end

@inline rotated_div_Uc(i, j, k, grid, scheme, U, c, buoyancy, tracers) = div_Uc(i, j, k, grid, scheme, U, c)

@inline function rotated_div_Uc(i, j, k, grid, scheme::RotatedAdvection, U, c, buoyancy, tracers)
    
    upwind_scheme = scheme.upwind_scheme
    centered_scheme_x = x_advection(upwind_scheme).advecting_velocity_scheme
    centered_scheme_y = y_advection(upwind_scheme).advecting_velocity_scheme
    centered_scheme_z = z_advection(upwind_scheme).advecting_velocity_scheme

    # Total advective fluxes
    𝒜x⁺ = _advective_tracer_flux_x(i+1, j,   k,   grid, upwind_scheme, U.u, c)
    𝒜x⁻ = _advective_tracer_flux_x(i,   j,   k,   grid, upwind_scheme, U.u, c)
    𝒜y⁺ = _advective_tracer_flux_y(i,   j+1, k,   grid, upwind_scheme, U.v, c)
    𝒜y⁻ = _advective_tracer_flux_y(i,   j,   k,   grid, upwind_scheme, U.v, c)
    𝒜z⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜z⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, upwind_scheme, U.w, c)

    # Centered advective fluxes
    𝒞x⁺ = _advective_tracer_flux_x(i+1, j,   k,   grid, centered_scheme_x, U.u, c)
    𝒞x⁻ = _advective_tracer_flux_x(i,   j,   k,   grid, centered_scheme_x, U.u, c)
    𝒞y⁺ = _advective_tracer_flux_y(i,   j+1, k,   grid, centered_scheme_y, U.v, c)
    𝒞y⁻ = _advective_tracer_flux_y(i,   j,   k,   grid, centered_scheme_y, U.v, c)
    𝒞z⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞z⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, centered_scheme_z, U.w, c)

    # Diffusive fluxes
    𝒟x⁺ = 𝒜x⁺ - 𝒞x⁺
    𝒟x⁻ = 𝒜x⁻ - 𝒞x⁻
    𝒟y⁺ = 𝒜y⁺ - 𝒞y⁺
    𝒟y⁻ = 𝒜y⁻ - 𝒞y⁻
    𝒟z⁺ = 𝒜z⁺ - 𝒞z⁺
    𝒟z⁻ = 𝒜z⁻ - 𝒞z⁻

    # TODO: make this a parameter?
    ϵ = Δzᶜᶜᶜ(i, j, k, grid)^2 / max(Δxᶜᶜᶜ(i, j, k, grid), Δyᶜᶜᶜ(i, j, k, grid))^2
    Smax = scheme.maximum_slope

    # Elements of the rotation tensor
    R₁₁⁺, R₁₂⁺, R₁₃⁺ = rotation_tensorᶠᶜᶜ(i+1, j, k, grid, buoyancy, tracers, Smax, ϵ)
    R₁₁⁻, R₁₂⁻, R₁₃⁻ = rotation_tensorᶠᶜᶜ(i,   j, k, grid, buoyancy, tracers, Smax, ϵ)

    R₂₁⁺, R₂₂⁺, R₂₃⁺ = rotation_tensorᶜᶠᶜ(i, j+1, k, grid, buoyancy, tracers, Smax, ϵ)
    R₂₁⁻, R₂₂⁻, R₂₃⁻ = rotation_tensorᶜᶠᶜ(i, j,   k, grid, buoyancy, tracers, Smax, ϵ)
    
    R₃₁⁺, R₃₂⁺, R₃₃⁺ = rotation_tensorᶜᶜᶠ(i, j, k+1, grid, buoyancy, tracers, Smax, ϵ)
    R₃₁⁻, R₃₂⁻, R₃₃⁻ = rotation_tensorᶜᶜᶠ(i, j, k,   grid, buoyancy, tracers, Smax, ϵ)

    # Rotated fluxes
    ℛx⁺ = R₁₁⁺ * 𝒟x⁺ + R₁₂⁺ * 𝒟y⁺ + R₁₃⁺ * 𝒟z⁺
    ℛx⁻ = R₁₁⁻ * 𝒟x⁻ + R₁₂⁻ * 𝒟y⁻ + R₁₃⁻ * 𝒟z⁻

    ℛy⁺ = R₂₁⁺ * 𝒟x⁺ + R₂₂⁺ * 𝒟y⁺ + R₂₃⁺ * 𝒟z⁺
    ℛy⁻ = R₂₁⁻ * 𝒟x⁻ + R₂₂⁻ * 𝒟y⁻ + R₂₃⁻ * 𝒟z⁻

    ℛz⁺ = R₃₁⁺ * 𝒟x⁺ + R₃₂⁺ * 𝒟y⁺ + R₃₃⁺ * 𝒟z⁺
    ℛz⁻ = R₃₁⁻ * 𝒟x⁻ + R₃₂⁻ * 𝒟y⁻ + R₃₃⁻ * 𝒟z⁻

    # Limiting the scheme to a minimum rotation
    α = scheme.minimum_rotation_percentage
    αx⁺ = min(α, abs(ℛx⁺) / abs(𝒟x⁺))
    αx⁻ = min(α, abs(ℛx⁻) / abs(𝒟x⁻))
        
    αy⁺ = min(α, abs(ℛy⁺) / abs(𝒟y⁺))
    αy⁻ = min(α, abs(ℛy⁻) / abs(𝒟y⁻))
       
    αz⁺ = min(α, abs(ℛz⁺) / abs(𝒟z⁺))
    αz⁻ = min(α, abs(ℛz⁻) / abs(𝒟z⁻))


    Fx⁺ = 𝒞x⁺ + αx⁺ * ℛx⁺ + (1 - αx⁺) * 𝒟x⁺
    Fx⁻ = 𝒞x⁻ + αx⁻ * ℛx⁻ + (1 - αx⁻) * 𝒟x⁻
                                            
    Fy⁻ = 𝒞y⁻ + αy⁺ * ℛy⁻ + (1 - αy⁺) * 𝒟y⁻
    Fy⁺ = 𝒞y⁺ + αy⁻ * ℛy⁺ + (1 - αy⁻) * 𝒟y⁺
                                             
    Fz⁺ = 𝒞z⁺ + αz⁺ * ℛz⁺ + (1 - αz⁺) * 𝒟z⁺
    Fz⁻ = 𝒞z⁻ + αz⁻ * ℛz⁻ + (1 - αz⁻) * 𝒟z⁻

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
