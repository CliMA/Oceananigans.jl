using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans: ∂x_b, ∂y_b, ∂z_b
using Oceananigans.Operators

struct RotatedAdvection{N, FT, U} <: AbstractUpwindBiasedAdvectionScheme{N, FT}
    upwind_scheme :: U
end

RotatedAdvection(upwind_scheme::U) where U = 
    RotatedAdvection{required_halo_size_x(upwind_scheme), eltype(upwind_scheme), U}(upwind_scheme)

@inline rotated_div_Uc(i, j, k, grid, scheme, U, c, buoyancy, tracers) = div_Uc(i, j, k, grid, scheme, U, c)

@inline function rotated_div_Uc(i, j, k, grid, scheme::RotatedAdvection, U, c, buoyancy, tracers)
    
    upwind_scheme = scheme.upwind_scheme
    centered_scheme_x = x_advection(upwind_scheme).advecting_velocity_scheme
    centered_scheme_y = y_advection(upwind_scheme).advecting_velocity_scheme
    centered_scheme_z = z_advection(upwind_scheme).advecting_velocity_scheme

    𝒜x⁺ = _advective_tracer_flux_x(i+1, j,   k,   grid, upwind_scheme, U.u, c)
    𝒜x⁻ = _advective_tracer_flux_x(i,   j,   k,   grid, upwind_scheme, U.u, c)
    𝒜y⁺ = _advective_tracer_flux_y(i,   j+1, k,   grid, upwind_scheme, U.v, c)
    𝒜y⁻ = _advective_tracer_flux_y(i,   j,   k,   grid, upwind_scheme, U.v, c)
    𝒜z⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜z⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, upwind_scheme, U.w, c)

    𝒞x⁺ = _advective_tracer_flux_x(i+1, j,   k,   grid, centered_scheme_x, U.u, c)
    𝒞x⁻ = _advective_tracer_flux_x(i,   j,   k,   grid, centered_scheme_x, U.u, c)
    𝒞y⁺ = _advective_tracer_flux_y(i,   j+1, k,   grid, centered_scheme_y, U.v, c)
    𝒞y⁻ = _advective_tracer_flux_y(i,   j,   k,   grid, centered_scheme_y, U.v, c)
    𝒞z⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞z⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, centered_scheme_z, U.w, c)

    𝒟x⁺ = 𝒜x⁺ - 𝒞x⁺
    𝒟x⁻ = 𝒜x⁻ - 𝒞x⁻
    𝒟y⁺ = 𝒜y⁺ - 𝒞y⁺
    𝒟y⁻ = 𝒜y⁻ - 𝒞y⁻
    𝒟z⁺ = 𝒜z⁺ - 𝒞z⁺
    𝒟z⁻ = 𝒜z⁻ - 𝒞z⁻

    ϵ = Δzᶜᶜᶜ(i, j, k, grid)^2 / max(Δxᶜᶜᶜ(i, j, k, grid), Δyᶜᶜᶜ(i, j, k, grid))^2

    R₁₁⁺, R₁₂⁺, R₁₃⁺ = rotation_tensorᶠᶜᶜ(i+1, j, k, grid, buoyancy, tracers, ϵ)
    R₁₁⁻, R₁₂⁻, R₁₃⁻ = rotation_tensorᶠᶜᶜ(i,   j, k, grid, buoyancy, tracers, ϵ)

    R₂₁⁺, R₂₂⁺, R₂₃⁺ = rotation_tensorᶜᶠᶜ(i, j+1, k, grid, buoyancy, tracers, ϵ)
    R₂₁⁻, R₂₂⁻, R₂₃⁻ = rotation_tensorᶜᶠᶜ(i, j,   k, grid, buoyancy, tracers, ϵ)
    
    R₃₁⁺, R₃₂⁺, R₃₃⁺ = rotation_tensorᶜᶜᶠ(i, j, k+1, grid, buoyancy, tracers, ϵ)
    R₃₁⁻, R₃₂⁻, R₃₃⁻ = rotation_tensorᶜᶜᶠ(i, j, k,   grid, buoyancy, tracers, ϵ)

    Fx⁺ = 𝒞x⁺ + R₁₁⁺ * 𝒟x⁺ + R₁₂⁺ * 𝒟y⁺ + R₁₃⁺ * 𝒟z⁺
    Fy⁺ = 𝒞y⁺ + R₂₁⁺ * 𝒟x⁺ + R₂₂⁺ * 𝒟y⁺ + R₂₃⁺ * 𝒟z⁺
    Fz⁺ = 𝒞z⁺ + R₃₁⁺ * 𝒟x⁺ + R₃₂⁺ * 𝒟y⁺ + R₃₃⁺ * 𝒟z⁺

    Fx⁻ = 𝒞x⁻ + R₁₁⁻ * 𝒟x⁻ + R₁₂⁻ * 𝒟y⁻ + R₁₃⁻ * 𝒟z⁻
    Fy⁻ = 𝒞y⁻ + R₂₁⁻ * 𝒟x⁻ + R₂₂⁻ * 𝒟y⁻ + R₂₃⁻ * 𝒟z⁻
    Fz⁻ = 𝒞z⁻ + R₃₁⁻ * 𝒟x⁻ + R₃₂⁻ * 𝒟y⁻ + R₃₃⁻ * 𝒟z⁻

    return 1 / Vᶜᶜᶜ(i, j, k, grid) * (Fx⁺ - Fx⁻ + Fy⁺ - Fy⁻ + Fz⁺ - Fz⁻)
end

@inline function rotation_tensorᶠᶜᶜ(i, j, k, grid, buoyancy, tracers, ϵ)
    bx =   ∂x_b(i, j, k, grid,       buoyancy, tracers) 
    by = ℑxyᶜᶠᵃ(i, j, k, grid, ∂y_b, buoyancy, tracers) 
    bz = ℑxzᶜᵃᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers) 
    S  = bx^2 + by^2 + bz^2
    Sx = abs(bx / bz)
    Sy = abs(by / bz)
    cond = (Sx < 10000) & (Sy < 10000) & (S > 0) # Tapering

    R₁₁ = ifelse(cond,   (by^2 + bz^2 + ϵ * bx^2) / S, one(grid)) 
    R₁₂ = ifelse(cond,        ((ϵ - 1) * bx * by) / S, zero(grid)) 
    R₁₃ = ifelse(cond,        ((ϵ - 1) * bx * bz) / S, zero(grid))

    return R₁₁, R₁₂, R₁₃
end

@inline function rotation_tensorᶜᶠᶜ(i, j, k, grid, buoyancy, tracers, ϵ)
    bx = ℑxyᶜᶠᵃ(i, j, k, grid, ∂x_b, buoyancy, tracers) 
    by =   ∂y_b(i, j, k, grid,       buoyancy, tracers) 
    bz = ℑyzᵃᶜᶠ(i, j, k, grid, ∂z_b, buoyancy, tracers) 
    S  = bx^2 + by^2 + bz^2
    Sx = abs(bx / bz)
    Sy = abs(by / bz)
    cond = (Sx < 10000) & (Sy < 10000) & (S > 0) # Tapering

    R₂₁ = ifelse(cond,      ((ϵ - 1) * by * bx) / S, zero(grid)) 
    R₂₂ = ifelse(cond, (bx^2 + bz^2 + ϵ * by^2) / S, one(grid)) 
    R₂₃ = ifelse(cond,      ((ϵ - 1) * by * bz) / S, zero(grid))

    return R₂₁, R₂₂, R₂₃
end

@inline function rotation_tensorᶜᶜᶠ(i, j, k, grid, buoyancy, tracers, ϵ)
    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers) 
    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers) 
    bz =   ∂z_b(i, j, k, grid,       buoyancy, tracers) 
    S  = bx^2 + by^2 + bz^2
    Sx = abs(bx / bz)
    Sy = abs(by / bz)
    cond = (Sx < 10000) & (Sy < 10000) & (S > 0) # Tapering

    R₃₁ = ifelse(cond,      ((ϵ - 1) * bz * bx) / S, zero(grid)) 
    R₃₂ = ifelse(cond,      ((ϵ - 1) * bz * by) / S, zero(grid))
    R₃₃ = ifelse(cond, (bx^2 + by^2 + ϵ * bz^2) / S, one(grid)) 

    return R₃₁, R₃₂, R₃₃
end
