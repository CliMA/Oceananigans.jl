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
@inline rotated_div_Uc(i, j, k, grid, scheme::RotatedAdvection, U, c, buoyancy, tracers) = div_Uc(i, j, k, grid, scheme.upwind_scheme, U, c)

@inline function rotated_div_Uc(i, j, k, grid, scheme::RotatedAdvection, U, c, buoyancy::SeawaterBuoyancy, tracers)
    
    upwind_scheme = scheme.upwind_scheme
    centered_scheme_x = x_advection(upwind_scheme).advecting_velocity_scheme
    centered_scheme_y = y_advection(upwind_scheme).advecting_velocity_scheme
    centered_scheme_z = z_advection(upwind_scheme).advecting_velocity_scheme

    # Total advective fluxes
    𝒜x₀⁺ = _advective_tracer_flux_x(i+1, j,   k-1, grid, upwind_scheme, U.u, c)
    𝒜x₀⁻ = _advective_tracer_flux_x(i,   j,   k-1, grid, upwind_scheme, U.u, c)
    𝒜x₁⁺ = _advective_tracer_flux_x(i+1, j,   k  , grid, upwind_scheme, U.u, c)
    𝒜x₁⁻ = _advective_tracer_flux_x(i,   j,   k  , grid, upwind_scheme, U.u, c)
    𝒜x₂⁺ = _advective_tracer_flux_x(i+1, j,   k+1, grid, upwind_scheme, U.u, c)
    𝒜x₂⁻ = _advective_tracer_flux_x(i,   j,   k+1, grid, upwind_scheme, U.u, c)

    𝒜y₀⁺ = _advective_tracer_flux_y(i,   j+1, k-1, grid, upwind_scheme, U.v, c)
    𝒜y₀⁻ = _advective_tracer_flux_y(i,   j,   k-1, grid, upwind_scheme, U.v, c)
    𝒜y₁⁺ = _advective_tracer_flux_y(i,   j+1, k,   grid, upwind_scheme, U.v, c)
    𝒜y₁⁻ = _advective_tracer_flux_y(i,   j,   k,   grid, upwind_scheme, U.v, c)
    𝒜y₂⁺ = _advective_tracer_flux_y(i,   j+1, k+1, grid, upwind_scheme, U.v, c)
    𝒜y₂⁻ = _advective_tracer_flux_y(i,   j,   k+1, grid, upwind_scheme, U.v, c)
    
    𝒜zˣ₀⁺ = _advective_tracer_flux_z(i-1, j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜zˣ₀⁻ = _advective_tracer_flux_z(i-1, j,   k,   grid, upwind_scheme, U.w, c)
    𝒜zˣ₁⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜zˣ₁⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, upwind_scheme, U.w, c)
    𝒜zˣ₁⁺ = _advective_tracer_flux_z(i+1, j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜zˣ₁⁻ = _advective_tracer_flux_z(i+1, j,   k,   grid, upwind_scheme, U.w, c)
    𝒜zʸ₀⁺ = _advective_tracer_flux_z(i,   j+1, k+1, grid, upwind_scheme, U.w, c)
    𝒜zʸ₀⁻ = _advective_tracer_flux_z(i,   j+1, k,   grid, upwind_scheme, U.w, c)
    𝒜zʸ₁⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜zʸ₁⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, upwind_scheme, U.w, c)
    𝒜zʸ₁⁺ = _advective_tracer_flux_z(i,   j-1, k+1, grid, upwind_scheme, U.w, c)
    𝒜zʸ₁⁻ = _advective_tracer_flux_z(i,   j-1, k,   grid, upwind_scheme, U.w, c)

    # Centered advective fluxes
    𝒞x₀⁺ = _advective_tracer_flux_x(i+1, j,   k-1, grid, centered_scheme_x, U.u, c)
    𝒞x₀⁻ = _advective_tracer_flux_x(i,   j,   k-1, grid, centered_scheme_x, U.u, c)
    𝒞x₁⁺ = _advective_tracer_flux_x(i+1, j,   k  , grid, centered_scheme_x, U.u, c)
    𝒞x₁⁻ = _advective_tracer_flux_x(i,   j,   k  , grid, centered_scheme_x, U.u, c)
    𝒞x₂⁺ = _advective_tracer_flux_x(i+1, j,   k+1, grid, centered_scheme_x, U.u, c)
    𝒞x₂⁻ = _advective_tracer_flux_x(i,   j,   k+1, grid, centered_scheme_x, U.u, c)

    𝒞y₀⁺ = _advective_tracer_flux_y(i,   j+1, k-1, grid, centered_scheme_y, U.v, c)
    𝒞y₀⁻ = _advective_tracer_flux_y(i,   j,   k-1, grid, centered_scheme_y, U.v, c)
    𝒞y₁⁺ = _advective_tracer_flux_y(i,   j+1, k,   grid, centered_scheme_y, U.v, c)
    𝒞y₁⁻ = _advective_tracer_flux_y(i,   j,   k,   grid, centered_scheme_y, U.v, c)
    𝒞y₂⁺ = _advective_tracer_flux_y(i,   j+1, k+1, grid, centered_scheme_y, U.v, c)
    𝒞y₂⁻ = _advective_tracer_flux_y(i,   j,   k+1, grid, centered_scheme_y, U.v, c) 
     
    𝒞zˣ₀⁺ = _advective_tracer_flux_z(i-1, j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞zˣ₀⁻ = _advective_tracer_flux_z(i-1, j,   k,   grid, centered_scheme_z, U.w, c)
    𝒞zˣ₁⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞zˣ₁⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, centered_scheme_z, U.w, c)
    𝒞zˣ₂⁺ = _advective_tracer_flux_z(i+1, j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞zˣ₂⁻ = _advective_tracer_flux_z(i+1, j,   k,   grid, centered_scheme_z, U.w, c)
    𝒞zʸ₀⁺ = _advective_tracer_flux_z(i,   j-1, k+1, grid, centered_scheme_z, U.w, c)
    𝒞zʸ₀⁻ = _advective_tracer_flux_z(i,   j-1, k,   grid, centered_scheme_z, U.w, c)
    𝒞zʸ₁⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞zʸ₁⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, centered_scheme_z, U.w, c)
    𝒞zʸ₂⁺ = _advective_tracer_flux_z(i,   j+1, k+1, grid, centered_scheme_z, U.w, c)
    𝒞zʸ₂⁻ = _advective_tracer_flux_z(i,   j+1, k,   grid, centered_scheme_z, U.w, c)

    # Diffusive fluxes for the whole triad
    Dx₀⁺ = 𝒜x₀⁺ - 𝒞x₀⁺ # @ fcc ->  i+1, j, k-1
    Dx₀⁻ = 𝒜x₀⁻ - 𝒞x₀⁻ # @ fcc ->  i,   j, k-1
    Dx₁⁺ = 𝒜x₁⁺ - 𝒞x₁⁺ # @ fcc ->  i+1, j, k
    Dx₁⁻ = 𝒜x₁⁻ - 𝒞x₁⁻ # @ fcc ->  i,   j, k
    Dx₂⁺ = 𝒜x₂⁺ - 𝒞x₂⁺ # @ fcc ->  i+1, j, k+1
    Dx₂⁻ = 𝒜x₂⁻ - 𝒞x₂⁻ # @ fcc ->  i,   j, k+1

    Dy₀⁺ = 𝒜y₀⁺ - 𝒞y₀⁺ # @ cfc ->  i, j+1, k-1
    Dy₀⁻ = 𝒜y₀⁻ - 𝒞y₀⁻ # @ cfc ->  i, j,   k-1
    Dy₁⁺ = 𝒜y₁⁺ - 𝒞y₁⁺ # @ cfc ->  i, j+1, k
    Dy₁⁻ = 𝒜y₁⁻ - 𝒞y₁⁻ # @ cfc ->  i, j,   k
    Dy₂⁺ = 𝒜y₂⁺ - 𝒞y₂⁺ # @ cfc ->  i, j+1, k+1
    Dy₂⁻ = 𝒜y₂⁻ - 𝒞y₂⁻ # @ cfc ->  i, j,   k+1
    
    Dzˣ₀⁺ = 𝒜zˣ₀⁺ - 𝒞zˣ₀⁺ # @ ccf ->  i-1, j, k+1
    Dzˣ₀⁻ = 𝒜zˣ₀⁻ - 𝒞zˣ₀⁻ # @ ccf ->  i-1, j, k
    Dzˣ₁⁺ = 𝒜zˣ₁⁺ - 𝒞zˣ₁⁺ # @ ccf ->  i,   j, k+1
    Dzˣ₁⁻ = 𝒜zˣ₁⁻ - 𝒞zˣ₁⁻ # @ ccf ->  i,   j, k
    Dzˣ₂⁺ = 𝒜zˣ₂⁺ - 𝒞zˣ₂⁺ # @ ccf ->  i+1, j, k+1
    Dzˣ₂⁻ = 𝒜zˣ₂⁻ - 𝒞zˣ₂⁻ # @ ccf ->  i+1, j, k
    Dzʸ₀⁺ = 𝒜zʸ₀⁺ - 𝒞zʸ₀⁺ # @ ccf ->  i, j-1, k+1
    Dzʸ₀⁻ = 𝒜zʸ₀⁻ - 𝒞zʸ₀⁻ # @ ccf ->  i, j-1, k
    Dzʸ₁⁺ = 𝒜zʸ₁⁺ - 𝒞zʸ₁⁺ # @ ccf ->  i, j,   k+1
    Dzʸ₁⁻ = 𝒜zʸ₁⁻ - 𝒞zʸ₁⁻ # @ ccf ->  i, j,   k
    Dzʸ₂⁺ = 𝒜zʸ₁⁺ - 𝒞zʸ₁⁺ # @ ccf ->  i, j+1, k+1
    Dzʸ₂⁻ = 𝒜zʸ₁⁻ - 𝒞zʸ₁⁻ # @ ccf ->  i, j+1, k

    # TODO: make this a parameter?
    ϵ = scheme.percentage_of_diapycnal_flux
    Smax = scheme.maximum_slope

    # Start with the triads!!
    bx₀⁺ = ∂x_b(i+1, j, k-1, grid, buoyancy, tracers)
    bx₀⁻ = ∂x_b(i,   j, k-1, grid, buoyancy, tracers)
    bx₁⁺ = ∂x_b(i+1, j, k  , grid, buoyancy, tracers)
    bx₁⁻ = ∂x_b(i,   j, k  , grid, buoyancy, tracers)
    bx₂⁺ = ∂x_b(i+1, j, k+1, grid, buoyancy, tracers)
    bx₂⁻ = ∂x_b(i,   j, k+1, grid, buoyancy, tracers)

    by₀⁺ = ∂y_b(i, j+1, k-1, grid, buoyancy, tracers)
    by₀⁻ = ∂y_b(i, j,   k-1, grid, buoyancy, tracers)
    by₁⁺ = ∂y_b(i, j+1, k  , grid, buoyancy, tracers)
    by₁⁻ = ∂y_b(i, j,   k  , grid, buoyancy, tracers)
    by₂⁺ = ∂y_b(i, j+1, k+1, grid, buoyancy, tracers)
    by₂⁻ = ∂y_b(i, j,   k+1, grid, buoyancy, tracers)
    
    bzˣ₀⁺ = ∂z_b(i-1, j,   k+1, grid, buoyancy, tracers)
    bzˣ₀⁻ = ∂z_b(i-1, j,   k,   grid, buoyancy, tracers)
    bzˣ₁⁺ = ∂z_b(i,   j,   k+1, grid, buoyancy, tracers)
    bzˣ₁⁻ = ∂z_b(i,   j,   k,   grid, buoyancy, tracers)
    bzˣ₂⁺ = ∂z_b(i+1, j,   k+1, grid, buoyancy, tracers)
    bzˣ₂⁻ = ∂z_b(i+1, j,   k,   grid, buoyancy, tracers)
    bzʸ₀⁺ = ∂z_b(i,   j-1, k+1, grid, buoyancy, tracers)
    bzʸ₀⁻ = ∂z_b(i,   j-1, k,   grid, buoyancy, tracers)
    bzʸ₁⁺ = ∂z_b(i,   j,   k+1, grid, buoyancy, tracers)
    bzʸ₁⁻ = ∂z_b(i,   j,   k,   grid, buoyancy, tracers)
    bzʸ₂⁺ = ∂z_b(i,   j+1, k+1, grid, buoyancy, tracers)
    bzʸ₂⁻ = ∂z_b(i,   j+1, k,   grid, buoyancy, tracers)
    
    # Small slope approximation, let's try?
    ℛx⁺ = 1 / 4Δzᶜᶜᶜ(i, j, k, grid) * (
        Δzᶜᶜᶠ(i, j, k,   grid) * (2Dx₁⁺ + bx₁⁺ / bzˣ₂⁻ * Dzˣ₂⁻ + bx₁⁺ / bzˣ₂⁺ * Dzˣ₂⁺) +
        Δzᶜᶜᶠ(i, j, k+1, grid) * (2Dx₁⁺ + bx₁⁺ / bzˣ₁⁻ * Dzˣ₁⁻ + bx₁⁺ / bzˣ₁⁺ * Dzˣ₁⁺)
    )

    ℛx⁻ = 1 / 4Δzᶜᶜᶜ(i-1, j, k, grid) * (
        Δzᶜᶜᶠ(i-1, j, k,   grid) * (2Dx₁⁻ + bx₁⁻ / bzˣ₁⁻ * Dzˣ₁⁻ + bx₁⁻ / bzˣ₁⁺ * Dzˣ₁⁺) +
        Δzᶜᶜᶠ(i-1, j, k+1, grid) * (2Dx₁⁻ + bx₁⁻ / bzˣ₀⁻ * Dzˣ₀⁻ + bx₁⁻ / bzˣ₀⁺ * Dzˣ₀⁺)
    )

    # Small slope approximation, let's try?
    ℛy⁺ = 1 / 4Δzᶜᶜᶜ(i, j, k, grid) * (
        Δzᶜᶜᶠ(i, j, k,   grid) * (2Dy₁⁺ + by₁⁺ / bzʸ₂⁻ * Dzʸ₂⁻ + by₁⁺ / bzʸ₂⁺ * Dzʸ₂⁺) +
        Δzᶜᶜᶠ(i, j, k+1, grid) * (2Dy₁⁺ + by₁⁺ / bzʸ₁⁻ * Dzʸ₁⁻ + by₁⁺ / bzʸ₁⁺ * Dzʸ₁⁺)
    )

    ℛy⁻ = 1 / 4Δzᶜᶜᶜ(i-1, j, k, grid) * (
        Δzᶜᶜᶠ(i-1, j, k,   grid) * (2Dy₁⁻ + by₁⁻ / bzʸ₁⁻ * Dzʸ₁⁻ + by₁⁻ / bzʸ₁⁺ * Dzʸ₁⁺) +
        Δzᶜᶜᶠ(i-1, j, k+1, grid) * (2Dy₁⁻ + by₁⁻ / bzʸ₀⁻ * Dzʸ₀⁻ + by₁⁻ / bzʸ₀⁺ * Dzʸ₀⁺)
    )

    # Small slope approximation, let's try?
    ℛz⁺ = 1 / 4Δxᶜᶜᶜ(i, j, k, grid) * (
        Δxᶠᶜᶜ(i, j, k,   grid) * (bx₁⁻ / bzˣ₁⁺ * (bx₁⁻ / bzˣ₁⁺ * Dzˣ₁⁺ + Dx₁⁻)  +
                                  bx₂⁻ / bzˣ₁⁺ * (bx₂⁻ / bzˣ₁⁺ * Dzˣ₁⁺ + Dx₂⁻)) +
        Δxᶠᶜᶜ(i+1, j, k, grid) * (bx₁⁺ / bzˣ₁⁺ * (bx₁⁺ / bzˣ₁⁺ * Dzˣ₁⁺ + Dx₁⁺)  +
                                  bx₂⁺ / bzˣ₁⁺ * (bx₂⁺ / bzˣ₁⁺ * Dzˣ₁⁺ + Dx₂⁺))
    )

    ℛz⁻ = 1 / 4Δxᶜᶜᶜ(i, j, k-1, grid) * (
        Δxᶠᶜᶜ(i,   j, k-1, grid) * (bx₀⁻ / bzˣ₁⁻ * (bx₀⁻ / bzˣ₁⁻ * Dzˣ₁⁻ + Dx₀⁻)  +
                                    bx₁⁻ / bzˣ₁⁻ * (bx₁⁻ / bzˣ₁⁻ * Dzˣ₁⁻ + Dx₁⁻)) +
        Δxᶠᶜᶜ(i+1, j, k-1, grid) * (bx₀⁺ / bzˣ₁⁻ * (bx₀⁺ / bzˣ₁⁻ * Dzˣ₁⁻ + Dx₀⁺)  +
                                    bx₁⁺ / bzˣ₁⁻ * (bx₁⁺ / bzˣ₁⁻ * Dzˣ₁⁻ + Dx₁⁺))
    )

    α = scheme.rotation_percentage

    # Fluxes
    Fx⁺ = 𝒞x⁺ + α * ℛx⁺ + (1 - α) * 𝒟x⁺
    Fx⁻ = 𝒞x⁻ + α * ℛx⁻ + (1 - α) * 𝒟x⁻                                           
    Fz⁺ = 𝒞z⁺ + α * ℛz⁺ + (1 - α) * 𝒟z⁺
    Fz⁻ = 𝒞z⁻ + α * ℛz⁻ + (1 - α) * 𝒟z⁻
        
    return 1 / Vᶜᶜᶜ(i, j, k, grid) * (Fx⁺ - Fx⁻ + Fy⁺ - Fy⁻ + Fz⁺ - Fz⁻)
end
