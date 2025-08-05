using Oceananigans.Grids: LatitudeLongitudeGrid, OrthogonalSphericalShellGrid, peripheral_node, φnode
using Oceananigans.Operators: Δx_qᶜᶠᶜ, Δy_qᶠᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ, hack_sind, hack_cosd
using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.ImmersedBoundaries

"""
    struct SphericalCoriolis{S, FT} <: AbstractRotation

A parameter object for constant rotation around a vertical axis on the sphere.
"""
struct SphericalCoriolis{H, S, FT} <: AbstractRotation
    rotation_rate :: FT
    scheme :: S
    HydrostaticCoriolis :: H
end

"""
    SphericalCoriolis([FT=Float64;]
                                 rotation_rate = Ω_Earth,
                                 scheme = EnstrophyConserving())
                                 HydrostaticCoriolis = 

Return a parameter object for Coriolis forces on a sphere rotating at `rotation_rate`.

Keyword arguments
=================

- `rotation_rate`: Sphere's rotation rate; default: [`Ω_Earth`](@ref).
- `scheme`: Either `EnergyConserving()`, `EnstrophyConserving()`, or `EnstrophyConserving()` (default).
- 'HydrostaticCoriolis`: `HydrostaticCoriolis` or `NonhydrostaticCoriolis`.
"""

function SphericalCoriolis(FT::DataType=Oceananigans.defaults.FloatType;
                                      rotation_rate = Ω_Earth,
                                      scheme :: S = EnstrophyConserving(FT)) where S

    return SphericalCoriolis{S, FT}(rotation_rate, scheme)
end

Adapt.adapt_structure(to, coriolis::SphericalCoriolis) =
    SphericalCoriolis(Adapt.adapt(to, coriolis.rotation_rate),
                                 Adapt.adapt(to, coriolis.scheme))

@inline φᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, face)
@inline φᶠᶠᵃ(i, j, k, grid::OrthogonalSphericalShellGrid) = φnode(i, j, grid, face, face)
@inline φᶠᶠᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶠᶠᵃ(i, j, k, grid.underlying_grid)

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 
    2 * coriolis.rotation_rate * hack_sind(φᶠᶠᵃ(i, j, k, grid))
@inline f̃ᶠᶠᵃ(i, j, k, grid, coriolis::SphericalCoriolis) =
    2 * coriolis.rotation_rate * hack_cosd(φᶠᶠᵃ(i, j, k, grid))

@inline f_ℑy_uᶠᶠᵃ(i, j, k, grid, coriolis, u) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)
@inline f̃_ℑz_uᶠᶠᵃ(i, j, k, grid, coriolis::NonhydrostaticSphericalCoriolis, u) = f̃ᶠᶠᵃ(i, j, k, grid, coriolis) * ℑzᵃᶠᵃ(i, j, k, grid, Δz_qᶜᶜᶠ, u)
@inline f̃_ℑz_uᶠᶠᵃ(i, j, k, grid, coriolis::HydrostaticSphericalCoriolis, u) = zero(grid)

@inline f_ℑx_vᶠᶠᵃ(i, j, k, grid, coriolis, v) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)

@inline f̃_ℑx_wᶠᶠᵃ(i, j, k, grid, coriolis::NonhydrostaticSphericalCoriolis, w) = f̃ᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, w)
@inline f̃_ℑx_wᶠᶠᵃ(i, j, k, grid, coriolis::HydrostaticSphericalCoriolis, w) = zero(grid)

@inline x_f_cross_U(i, j, k, grid, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_vᶠᶠᵃ, U[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)  
                + ℑzᵃᶜᵃ(i, j, k, grid, f̃_ℑx_wᶠᶠᵃ, U[3]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_uᶠᶠᵃ, U[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)


@inline z_f_cross_U(i, j, k, grid, U) =
    @inbounds - ℑxᶜᵃᵃ(i, j, k, grid, f̃_ℑz_uᶠᶠᵃ, U[1]) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid)

#####
##### Show
#####

function Base.show(io::IO, spherical_coriolis::SphericalCoriolis)
    coriolis_scheme = spherical_coriolis.scheme
    rotation_rate   = spherical_coriolis.rotation_rate
    rotation_rate_Earth = rotation_rate / Ω_Earth
    rotation_rate_str = @sprintf("%.2e s⁻¹ = %.2e Ω_Earth", rotation_rate, rotation_rate_Earth)

    return print(io, "SphericalCoriolis", '\n',
                 "├─ rotation rate: ", rotation_rate_str, '\n',
                 "└─ scheme: ", summary(coriolis_scheme))
end
