using Oceananigans.Grids: LatitudeLongitudeGrid, OrthogonalSphericalShellGrid, peripheral_node, φnode
using Oceananigans.Operators: Δx_qᶜᶠᶜ, Δy_qᶠᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ, hack_sind, hack_cosd
using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.ImmersedBoundaries

"""
    struct NonhydrostaticSphericalCoriolis{S, FT} <: AbstractRotation

A parameter object for constant rotation around a vertical axis on the sphere.

This is only designed to work for a LatitudeLongitudeGrid.

There is no attempt to conserve energy or enstrophy at this early stage.
"""
struct NonhydrostaticSphericalCoriolis{S, FT} <: AbstractRotation
    rotation_rate :: FT
end

"""
    NonhydrostaticSphericalCoriolis([FT=Float64;]
                                 rotation_rate = Ω_Earth,

Return a parameter object for Coriolis forces on a sphere rotating at `rotation_rate`.

Keyword arguments
=================

- `rotation_rate`: Sphere's rotation rate; default: [`Ω_Earth`](@ref).
"""
function NonhydrostaticSphericalCoriolis(FT::DataType=Oceananigans.defaults.FloatType;
                                      rotation_rate = Ω_Earth,

    return NonhydrostaticSphericalCoriolis{FT}(rotation_rate)
end

Adapt.adapt_structure(to, coriolis::NonhydrostaticSphericalCoriolis) =
    NonhydrostaticSphericalCoriolis(Adapt.adapt(to))

@inline φᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, face)

@inline fᶠᶠᵃ(i, j, k, grid) = 2 * coriolis.rotation_rate * hack_sind(φᶠᶠᵃ(i, j, k, grid))
@inline f̃ᶠᶠᵃ(i, j, k, grid) = 2 * coriolis.rotation_rate * hack_cosd(φᶠᶠᵃ(i, j, k, grid))

@inline f_ℑx_vᶠᶠᵃ(i, j, k, grid, coriolis, v) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)
@inline f̃_ℑx_wᶠᶠᵃ(i, j, k, grid, coriolis, v) = f̃ᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, w)

@inline f_ℑy_uᶠᶠᵃ(i, j, k, grid, coriolis, u) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)

@inline f̃_ℑz_uᶠᶠᵃ(i, j, k, grid, coriolis, u) = f̃ᶠᶠᵃ(i, j, k, grid, coriolis) * ℑzᵃᶠᵃ(i, j, k, grid, Δz_qᶜᶜᶠ, u)

@inline x_f_cross_U(i, j, k, grid, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_vᶠᶠᵃ, U[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)  
              + ℑzᵃᶜᵃ(i, j, k, grid, f̃_ℑx_wᶠᶠᵃ, U[3]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_uᶠᶠᵃ, U[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)

@inline z_f_cross_U(i, j, k, grid, U) =
    @inbounds - ℑxᶜᵃᵃ(i, j, k, grid, f̃_ℑz_uᶠᶠᵃ, U[1]) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid)

function Base.show(io::IO, nonhydrostatic_spherical_coriolis::NonhydrostaticSphericalCoriolis)
    rotation_rate   = hydrostatic_spherical_coriolis.rotation_rate
    rotation_rate_Earth = rotation_rate / Ω_Earth
    rotation_rate_str = @sprintf("%.2e s⁻¹ = %.2e Ω_Earth", rotation_rate, rotation_rate_Earth)

    return print(io, "NonhydrostaticSphericalCoriolis", '\n',
                 "├─ rotation rate: ", rotation_rate_str, '\n')
end
