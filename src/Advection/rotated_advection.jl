using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans: ∂x_b, ∂y_b, ∂z_b
using Oceananigans.Operators

struct RotatedAdvection{N, FT, U} <: AbstractUpwindBiasedAdvectionScheme{N, FT}
    upwind_scheme :: U
    minimum_rotation_percentage :: FT
    maximum_slope :: FT
    percentage_of_diapycnal_flux :: FT
end

function RotatedAdvection(upwind_scheme::U;
                          maximum_slope = 1e5, 
                          minimum_rotation_percentage = 0.1,
                          percentage_of_diapycnal_flux = 1e-7) where U 

    FT = eltype(upwind_scheme)
    
    N  = max(required_halo_size_x(upwind_scheme),
             required_halo_size_y(upwind_scheme),
             required_halo_size_z(upwind_scheme))
    
    minimum_rotation_percentage = convert(FT, minimum_rotation_percentage)
    maximum_slope = convert(FT, maximum_slope)
    percentage_of_diapycnal_flux = convert(FT, percentage_of_diapycnal_flux)

    return RotatedAdvection{N, FT, U}(upwind_scheme, minimum_rotation_percentage, maximum_slope, percentage_of_diapycnal_flux)
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
    
    𝒜zˣ₀⁺ = _advective_tracer_flux_z(i+1, j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜zˣ₀⁻ = _advective_tracer_flux_z(i+1, j,   k,   grid, upwind_scheme, U.w, c)
    𝒜zˣ₁⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜zˣ₁⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, upwind_scheme, U.w, c)
    𝒜zˣ₁⁺ = _advective_tracer_flux_z(i-1, j,   k+1, grid, upwind_scheme, U.w, c)
    𝒜zˣ₁⁻ = _advective_tracer_flux_z(i-1, j,   k,   grid, upwind_scheme, U.w, c)
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
     
    𝒞zˣ₀⁺ = _advective_tracer_flux_z(i+1, j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞zˣ₀⁻ = _advective_tracer_flux_z(i+1, j,   k,   grid, centered_scheme_z, U.w, c)
    𝒞zˣ₁⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞zˣ₁⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, centered_scheme_z, U.w, c)
    𝒞zˣ₂⁺ = _advective_tracer_flux_z(i-1, j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞zˣ₂⁻ = _advective_tracer_flux_z(i-1, j,   k,   grid, centered_scheme_z, U.w, c)
    𝒞zʸ₀⁺ = _advective_tracer_flux_z(i,   j-1, k+1, grid, centered_scheme_z, U.w, c)
    𝒞zʸ₀⁻ = _advective_tracer_flux_z(i,   j-1, k,   grid, centered_scheme_z, U.w, c)
    𝒞zʸ₁⁺ = _advective_tracer_flux_z(i,   j,   k+1, grid, centered_scheme_z, U.w, c)
    𝒞zʸ₁⁻ = _advective_tracer_flux_z(i,   j,   k,   grid, centered_scheme_z, U.w, c)
    𝒞zʸ₂⁺ = _advective_tracer_flux_z(i,   j+1, k+1, grid, centered_scheme_z, U.w, c)
    𝒞zʸ₂⁻ = _advective_tracer_flux_z(i,   j+1, k,   grid, centered_scheme_z, U.w, c)

    # Diffusive fluxes for the whole triad
    𝒟x₀⁺ = 𝒜x₀⁺ - 𝒞x₀⁺ # @ fcc ->  i+1, j, k-1
    𝒟x₀⁻ = 𝒜x₀⁻ - 𝒞x₀⁻ # @ fcc ->  i,   j, k-1
    𝒟x₁⁺ = 𝒜x₁⁺ - 𝒞x₁⁺ # @ fcc ->  i+1, j, k
    𝒟x₁⁻ = 𝒜x₁⁻ - 𝒞x₁⁻ # @ fcc ->  i,   j, k
    𝒟x₂⁺ = 𝒜x₂⁺ - 𝒞x₂⁺ # @ fcc ->  i+1, j, k+1
    𝒟x₂⁻ = 𝒜x₂⁻ - 𝒞x₂⁻ # @ fcc ->  i,   j, k+1

    𝒟y₀⁺ = 𝒜y₀⁺ - 𝒞y₀⁺ # @ cfc ->  i, j+1, k-1
    𝒟y₀⁻ = 𝒜y₀⁻ - 𝒞y₀⁻ # @ cfc ->  i, j,   k-1
    𝒟y₁⁺ = 𝒜y₁⁺ - 𝒞y₁⁺ # @ cfc ->  i, j+1, k
    𝒟y₁⁻ = 𝒜y₁⁻ - 𝒞y₁⁻ # @ cfc ->  i, j,   k
    𝒟y₂⁺ = 𝒜y₂⁺ - 𝒞y₂⁺ # @ cfc ->  i, j+1, k+1
    𝒟y₂⁻ = 𝒜y₂⁻ - 𝒞y₂⁻ # @ cfc ->  i, j,   k+1
    
    𝒟zˣ₀⁺ = 𝒜zˣ₀⁺ - 𝒞zˣ₀⁺ # @ ccf ->  i-1, j, k+1
    𝒟zˣ₀⁻ = 𝒜zˣ₀⁻ - 𝒞zˣ₀⁻ # @ ccf ->  i-1, j, k
    𝒟zˣ₁⁺ = 𝒜zˣ₁⁺ - 𝒞zˣ₁⁺ # @ ccf ->  i,   j, k+1
    𝒟zˣ₁⁻ = 𝒜zˣ₁⁻ - 𝒞zˣ₁⁻ # @ ccf ->  i,   j, k
    𝒟zˣ₂⁺ = 𝒜zˣ₁⁺ - 𝒞zˣ₁⁺ # @ ccf ->  i+1, j, k+1
    𝒟zˣ₂⁻ = 𝒜zˣ₁⁻ - 𝒞zˣ₁⁻ # @ ccf ->  i+1, j, k
    𝒟zʸ₀⁺ = 𝒜zʸ₀⁺ - 𝒞zʸ₀⁺ # @ ccf ->  i, j-1, k+1
    𝒟zʸ₀⁻ = 𝒜zʸ₀⁻ - 𝒞zʸ₀⁻ # @ ccf ->  i, j-1, k
    𝒟zʸ₁⁺ = 𝒜zʸ₁⁺ - 𝒞zʸ₁⁺ # @ ccf ->  i, j,   k+1
    𝒟zʸ₁⁻ = 𝒜zʸ₁⁻ - 𝒞zʸ₁⁻ # @ ccf ->  i, j,   k
    𝒟zʸ₂⁺ = 𝒜zʸ₁⁺ - 𝒞zʸ₁⁺ # @ ccf ->  i, j+1, k+1
    𝒟zʸ₂⁻ = 𝒜zʸ₁⁻ - 𝒞zʸ₁⁻ # @ ccf ->  i, j+1, k

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
    
    bzˣ₀⁺ = ∂z_b(i+1, j,   k+1, grid, buoyancy, tracers)
    bzˣ₀⁻ = ∂z_b(i+1, j,   k,   grid, buoyancy, tracers)
    bzˣ₁⁺ = ∂z_b(i,   j,   k+1, grid, buoyancy, tracers)
    bzˣ₁⁻ = ∂z_b(i,   j,   k,   grid, buoyancy, tracers)
    bzˣ₂⁺ = ∂z_b(i-1, j,   k+1, grid, buoyancy, tracers)
    bzˣ₂⁻ = ∂z_b(i-1, j,   k,   grid, buoyancy, tracers)
    bzʸ₀⁺ = ∂z_b(i,   j-1, k+1, grid, buoyancy, tracers)
    bzʸ₀⁻ = ∂z_b(i,   j-1, k,   grid, buoyancy, tracers)
    bzʸ₁⁺ = ∂z_b(i,   j,   k+1, grid, buoyancy, tracers)
    bzʸ₁⁻ = ∂z_b(i,   j,   k,   grid, buoyancy, tracers)
    bzʸ₂⁺ = ∂z_b(i,   j+1, k+1, grid, buoyancy, tracers)
    bzʸ₂⁻ = ∂z_b(i,   j+1, k,   grid, buoyancy, tracers)
    
    # Slopes
    

    # Rotated fluxes, Cannot do this!!!
    ℛx⁺ = R₁₁⁺ * 𝒟x⁺ + R₁₂⁺ * 𝒟y⁺ + R₁₃⁺ * 𝒟z⁺
    ℛx⁻ = R₁₁⁻ * 𝒟x⁻ + R₁₂⁻ * 𝒟y⁻ + R₁₃⁻ * 𝒟z⁻
    ℛy⁺ = R₂₁⁺ * 𝒟x⁺ + R₂₂⁺ * 𝒟y⁺ + R₂₃⁺ * 𝒟z⁺
    ℛy⁻ = R₂₁⁻ * 𝒟x⁻ + R₂₂⁻ * 𝒟y⁻ + R₂₃⁻ * 𝒟z⁻
    ℛz⁺ = R₃₁⁺ * 𝒟x⁺ + R₃₂⁺ * 𝒟y⁺ + R₃₃⁺ * 𝒟z⁺
    ℛz⁻ = R₃₁⁻ * 𝒟x⁻ + R₃₂⁻ * 𝒟y⁻ + R₃₃⁻ * 𝒟z⁻

    α = scheme.minimum_rotation_percentage

    # Tapering when the slope of the tracer is
    # the same as the slope of the buoyancy
    αx⁺ = ifelse(Rx⁺ < α, α, one(grid))
    αx⁻ = ifelse(Rx⁻ < α, α, one(grid))
    αy⁺ = ifelse(Ry⁺ < α, α, one(grid))
    αy⁻ = ifelse(Ry⁻ < α, α, one(grid))
    αz⁺ = ifelse(Rz⁺ < α, α, one(grid))
    αz⁻ = ifelse(Rz⁻ < α, α, one(grid))

    # Renormalize the rotated fluxes based on the α
    ℛx⁺ = R₁₁⁺ * αx⁺ * 𝒟x⁺ + R₁₂⁺ * αy⁺ * 𝒟y⁺ + R₁₃⁺ * αz⁺ * 𝒟z⁺
    ℛx⁻ = R₁₁⁻ * αx⁻ * 𝒟x⁻ + R₁₂⁻ * αy⁻ * 𝒟y⁻ + R₁₃⁻ * αz⁻ * 𝒟z⁻
    ℛy⁺ = R₂₁⁺ * αx⁺ * 𝒟x⁺ + R₂₂⁺ * αy⁺ * 𝒟y⁺ + R₂₃⁺ * αz⁺ * 𝒟z⁺
    ℛy⁻ = R₂₁⁻ * αx⁻ * 𝒟x⁻ + R₂₂⁻ * αy⁻ * 𝒟y⁻ + R₂₃⁻ * αz⁻ * 𝒟z⁻
    ℛz⁺ = R₃₁⁺ * αx⁺ * 𝒟x⁺ + R₃₂⁺ * αy⁺ * 𝒟y⁺ + R₃₃⁺ * αz⁺ * 𝒟z⁺
    ℛz⁻ = R₃₁⁻ * αx⁻ * 𝒟x⁻ + R₃₂⁻ * αy⁻ * 𝒟y⁻ + R₃₃⁻ * αz⁻ * 𝒟z⁻

    # Fluxes
    Fx⁺ = 𝒞x⁺ + ℛx⁺ + (1 - αx⁺) * 𝒟x⁺
    Fx⁻ = 𝒞x⁻ + ℛx⁻ + (1 - αx⁻) * 𝒟x⁻                                           
    Fz⁺ = 𝒞z⁺ + ℛz⁺ + (1 - αz⁺) * 𝒟z⁺
    Fz⁻ = 𝒞z⁻ + ℛz⁻ + (1 - αz⁻) * 𝒟z⁻
        
    return 1 / Vᶜᶜᶜ(i, j, k, grid) * (Fx⁺ - Fx⁻ + Fy⁺ - Fy⁻ + Fz⁺ - Fz⁻)
end
