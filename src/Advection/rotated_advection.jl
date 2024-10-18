using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans: ∂x_b, ∂y_b, ∂z_b

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

    bx⁺ = ∂x_b(i+1, j,   k,   grid, buoyancy, tracers)
    bx⁻ = ∂x_b(i,   j,   k,   grid, buoyancy, tracers)
    by⁺ = ∂y_b(i,   j+1, k,   grid, buoyancy, tracers)
    by⁻ = ∂y_b(i,   j,   k,   grid, buoyancy, tracers)
    bz⁺ = ∂z_b(i,   j,   k+1, grid, buoyancy, tracers)
    bz⁻ = ∂z_b(i,   j,   k,   grid, buoyancy, tracers)

    S = bz⁺^2 + by⁺^2 + bx⁺^2

    R₁₁⁺ = bz⁺^2 + by⁺^2 + ϵ * bx⁺^2
    R₂₂⁺ = bz⁺^2 + bx⁺^2 + ϵ * by⁺^2
    R₃₃⁺ = bx⁺^2 + by⁺^2 + ϵ * bz⁺^2

    R₁₂⁺ = (ϵ - 1) * bx⁺ * by⁺
    R₁₃⁺ = (ϵ - 1) * bx⁺ * bz⁺
    R₂₃⁺ = (ϵ - 1) * by⁺ * bz⁺
    
    R₁₁⁻ = bz⁻^2 + by⁻^2 + ϵ * bx⁻^2
    R₂₂⁻ = bz⁻^2 + bx⁻^2 + ϵ * by⁻^2
    R₃₃⁻ = bx⁻^2 + by⁻^2 + ϵ * bz⁻^2

    R₁₂⁻ = (ϵ - 1) * bx⁻ * by⁻
    R₁₃⁻ = (ϵ - 1) * bx⁻ * bz⁻
    R₂₃⁻ = (ϵ - 1) * by⁻ * bz⁻

    Fx⁺ = 𝒞x⁺ + ifelse(S < 10ϵ, 𝒟x⁺, (R₁₁⁺ * 𝒟x⁺ + R₁₂⁺ * 𝒟y⁺ + R₁₃⁺ * 𝒟z⁺) / S)
    Fy⁺ = 𝒞y⁺ + ifelse(S < 10ϵ, 𝒟y⁺, (R₁₂⁺ * 𝒟x⁺ + R₂₂⁺ * 𝒟y⁺ + R₂₃⁺ * 𝒟z⁺) / S)
    Fz⁺ = 𝒞z⁺ + ifelse(S < 10ϵ, 𝒟z⁺, (R₁₃⁺ * 𝒟x⁺ + R₂₃⁺ * 𝒟y⁺ + R₃₃⁺ * 𝒟z⁺) / S)

    Fx⁻ = 𝒞x⁻ + ifelse(S < 10ϵ, 𝒟x⁻, (R₁₁⁻ * 𝒟x⁻ + R₁₂⁻ * 𝒟y⁻ + R₁₃⁻ * 𝒟z⁻) / S)
    Fy⁻ = 𝒞y⁻ + ifelse(S < 10ϵ, 𝒟y⁻, (R₁₂⁻ * 𝒟x⁻ + R₂₂⁻ * 𝒟y⁻ + R₂₃⁻ * 𝒟z⁻) / S)
    Fz⁻ = 𝒞z⁻ + ifelse(S < 10ϵ, 𝒟z⁻, (R₁₃⁻ * 𝒟x⁻ + R₂₃⁻ * 𝒟y⁻ + R₃₃⁻ * 𝒟z⁻) / S)

    return 1 / Vᶜᶜᶜ(i, j, k, grid) * (Fx⁺ - Fx⁻ + Fy⁺ - Fy⁻ + Fz⁺ - Fz⁻)
end

@inline function rotated_div_Uc(i, j, k, grid::XFlatGrid, scheme::RotatedAdvection, U, c)
    
    upwind_scheme = scheme.upwind_scheme
    centered_scheme_y = y_advection(upwind_scheme).advecting_velocity_scheme
    centered_scheme_z = z_advection(upwind_scheme).advecting_velocity_scheme

    𝒜y⁺ = _advective_tracer_flux_y(i,   j+1, k,   grid, upwind_scheme, U.v, c)
    𝒜y⁻ = _advective_tracer_flux_y(i,   j,   k,   grid, upwind_scheme, U.v, c)
    𝒜z⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜z⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, upwind_scheme, U.w, c)

    𝒞y⁺ = _advective_tracer_flux_y(i,   j+1, k,   grid, centered_scheme_y, U.v, c)
    𝒞y⁻ = _advective_tracer_flux_y(i,   j,   k,   grid, centered_scheme_y, U.v, c)
    𝒞z⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞z⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, centered_scheme_z, U.w, c)

    𝒟y⁺ = 𝒜y⁺ - 𝒞y⁺
    𝒟y⁻ = 𝒜y⁻ - 𝒞y⁻
    𝒟z⁺ = 𝒜z⁺ - 𝒞z⁺
    𝒟z⁻ = 𝒜z⁻ - 𝒞z⁻

    ϵ = Δzᶜᶜᶜ(i, j, k, grid)^2 / Δyᶜᶜᶜ(i, j, k, grid)^2

    by⁺ = ∂y_b(i,   j+1, k,   grid, buoyancy, tracers)
    by⁻ = ∂y_b(i,   j,   k,   grid, buoyancy, tracers)
    bz⁺ = ∂z_b(i,   j,   k+1, grid, buoyancy, tracers)
    bz⁻ = ∂z_b(i,   j,   k,   grid, buoyancy, tracers)

    S = bz⁺^2 + by⁺^2

    R₂₂⁺ = bz⁺^2 + ϵ * by⁺^2
    R₃₃⁺ = by⁺^2 + ϵ * bz⁺^2

    R₂₃⁺ = (ϵ - 1) * by⁺ * bz⁺
    
    R₂₂⁻ = bz⁻^2 + ϵ * by⁻^2
    R₃₃⁻ = by⁻^2 + ϵ * bz⁻^2

    R₂₃⁻ = (ϵ - 1) * by⁻ * bz⁻

    Fy⁺ = 𝒞y⁺ + ifelse(S < ϵ, 𝒟y⁺, (R₂₂⁺ * 𝒟y⁺ + R₂₃⁺ * 𝒟z⁺) / S)
    Fz⁺ = 𝒞z⁺ + ifelse(S < ϵ, 𝒟z⁺, (R₂₃⁺ * 𝒟y⁺ + R₃₃⁺ * 𝒟z⁺) / S)

    Fy⁻ = 𝒞y⁻ + ifelse(S < ϵ, 𝒟y⁻, (R₂₂⁻ * 𝒟y⁻ + R₂₃⁻ * 𝒟z⁻) / S)
    Fz⁻ = 𝒞z⁻ + ifelse(S < ϵ, 𝒟z⁻, (R₃₂⁻ * 𝒟y⁻ + R₃₃⁻ * 𝒟z⁻) / S)

    return 1 / Vᶜᶜᶜ(i, j, k, grid) * (Fy⁺ - Fy⁻ + Fz⁺ - Fz⁻)
end

@inline function rotated_div_Uc(i, j, k, grid::YFlatGrid, scheme::RotatedAdvection, U, c)
    
    upwind_scheme = scheme.upwind_scheme
    centered_scheme_x = x_advection(upwind_scheme).advecting_velocity_scheme
    centered_scheme_z = z_advection(upwind_scheme).advecting_velocity_scheme

    𝒜x⁺ = _advective_tracer_flux_x(i+1, j,   k,   grid, upwind_scheme, U.u, c)
    𝒜x⁻ = _advective_tracer_flux_x(i,   j,   k,   grid, upwind_scheme, U.u, c)
    𝒜z⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜z⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, upwind_scheme, U.w, c)

    𝒞x⁺ = _advective_tracer_flux_x(i+1, j,   k,   grid, centered_scheme_x, U.u, c)
    𝒞x⁻ = _advective_tracer_flux_x(i,   j,   k,   grid, centered_scheme_x, U.u, c)
    𝒞z⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞z⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, centered_scheme_z, U.w, c)

    𝒟x⁺ = 𝒜x⁺ - 𝒞x⁺
    𝒟x⁻ = 𝒜x⁻ - 𝒞x⁻
    𝒟z⁺ = 𝒜z⁺ - 𝒞z⁺
    𝒟z⁻ = 𝒜z⁻ - 𝒞z⁻

    ϵ = Δzᶜᶜᶜ(i, j, k, grid)^2 / Δxᶜᶜᶜ(i, j, k, grid)^2

    bx⁺ = ∂x_b(i+1, j,   k,   grid, buoyancy, tracers)
    bx⁻ = ∂x_b(i,   j,   k,   grid, buoyancy, tracers)
    bz⁺ = ∂z_b(i,   j,   k+1, grid, buoyancy, tracers)
    bz⁻ = ∂z_b(i,   j,   k,   grid, buoyancy, tracers)

    S = bz⁺^2 + bx⁺^2

    R₁₁⁺ = bz⁺^2 + ϵ * bx⁺^2
    R₃₃⁺ = bx⁺^2 + ϵ * bz⁺^2

    R₁₃⁺ = (ϵ - 1) * bx⁺ * bz⁺
    
    R₁₁⁻ = bz⁻^2 + ϵ * bx⁻^2
    R₃₃⁻ = bx⁻^2 + ϵ * bz⁻^2

    R₁₃⁻ = (ϵ - 1) * bx⁻ * bz⁻

    Fx⁺ = 𝒞x⁺ + ifelse(S < ϵ, 𝒟x⁺, (R₁₁⁺ * 𝒟x⁺ + R₁₃⁺ * 𝒟z⁺) / S)
    Fz⁺ = 𝒞z⁺ + ifelse(S < ϵ, 𝒟z⁺, (R₁₃⁺ * 𝒟x⁺ + R₃₃⁺ * 𝒟z⁺) / S)

    Fx⁻ = 𝒞x⁻ + ifelse(S < ϵ, 𝒟x⁻, (R₁₁⁻ * 𝒟x⁻ + R₁₃⁻ * 𝒟z⁻) / S)
    Fz⁻ = 𝒞z⁻ + ifelse(S < ϵ, 𝒟z⁻, (R₁₃⁻ * 𝒟x⁻ + R₃₃⁻ * 𝒟z⁻) / S)

    return 1 / Vᶜᶜᶜ(i, j, k, grid) * (Fx⁺ - Fx⁻ + Fz⁺ - Fz⁻)
end
