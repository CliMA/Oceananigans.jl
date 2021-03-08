#####
##### Diffusive fluxes
#####

@inline diffusive_flux_x(i, j, k, grid, clock, κᶠᶜᶜ::Number, c) = κᶠᶜᶜ * Axᶠᶜᶜ(i, j, k, grid) * ∂xᶠᶜᵃ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, clock, κᶜᶠᶜ::Number, c) = κᶜᶠᶜ * Ayᶜᶠᶜ(i, j, k, grid) * ∂yᶜᶠᵃ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, clock, κᶜᶜᶠ::Number, c) = κᶜᶜᶠ * Azᶜᶜᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, c)

# Diffusivities-as-functions

@inline diffusive_flux_x(i, j, k, grid, clock, κ::Function, c) =
        diffusive_flux_x(i, j, k, grid, clock, κ(xnode(Face, i, grid), ynode(Center, j, grid), znode(Center, k, grid), clock.time), c)

@inline diffusive_flux_y(i, j, k, grid, clock, κ::Function, c) =
        diffusive_flux_y(i, j, k, grid, clock, κ(xnode(Center, i, grid), ynode(Face, j, grid), znode(Center, k, grid), clock.time), c)

@inline diffusive_flux_z(i, j, k, grid, clock, κ::Function, c) =
        diffusive_flux_z(i, j, k, grid, clock, κ(xnode(Center, i, grid), ynode(Center, j, grid), znode(Face, k, grid), clock.time), c)
                     
#####
##### Laplacian diffusion operator
#####

"""
    ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, κˣ, κʸ, κᶻ, c)

Calculates diffusion for a tracer c via

    1/V * [δxᶜᵃᵃ(κˣ * Ax * ∂xᶠᵃᵃ(c)) + δyᵃᶜᵃ(κʸ * Ay * ∂yᵃᶠᵃ(c)) + δzᵃᵃᶜ(κᶻ * Az * ∂zᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, clock, κˣ, κʸ, κᶻ, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, diffusive_flux_x, clock, κˣ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, diffusive_flux_y, clock, κʸ, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, diffusive_flux_z, clock, κᶻ, c))
end

@inline ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, clock, κ, c) = ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, clock, κ, κ, κ, c)

#####
#####
#####

"""
    κ_∂x_c(i, j, k, grid, c, κ, closure, args...)

Return `κ ∂x c`, where `κ` is an array or function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂x_c(i, j, k, grid, κ, c, closure, args...)
    κ = ℑxᶠᵃᵃ(i, j, k, grid, κ, closure, args...)
    ∂x_c = ∂xᶠᵃᵃ(i, j, k, grid, c)
    return κ * ∂x_c
end

"""
    κ_∂y_c(i, j, k, grid, c, κ, closure, args...)

Return `κ ∂y c`, where `κ` is an array or function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂y_c(i, j, k, grid, κ, c, closure, args...)
    κ = ℑyᵃᶠᵃ(i, j, k, grid, κ, closure, args...)
    ∂y_c = ∂yᵃᶠᵃ(i, j, k, grid, c)
    return κ * ∂y_c
end

"""
    κ_∂z_c(i, j, k, grid, c, κ, closure, buoyancy, u, v, w, T, S)

Return `κ ∂z c`, where `κ` is an array or function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂z_c(i, j, k, grid, κ, c, closure, args...)
    κ = ℑzᵃᵃᶠ(i, j, k, grid, κ, closure, args...)
    ∂z_c = ∂zᵃᵃᶠ(i, j, k, grid, c)
    return κ * ∂z_c
end

