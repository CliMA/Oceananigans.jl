using Oceananigans.Grids: RegularLatitudeLongitudeGrid
using Oceananigans.Operators: Δx_vᶜᶠᵃ, Δy_uᶠᶜᵃ, Δxᶠᶜᵃ, Δyᶜᶠᵃ

"""
    HydrostaticSphericalCoriolis{FT} <: AbstractRotation

A parameter object for constant rotation around a vertical axis.
"""
struct HydrostaticSphericalCoriolis{FT} <: AbstractRotation
    rotation_rate :: FT
end

"""
    HydrostaticSphericalCoriolis([FT=Float64;] rotation_rate=Ω_Earth)

Returns a parameter object for Coriolis forces on a sphere rotating at `rotation_rate`.
By default, `rotation_rate` is assumed to be Earth's.
"""
HydrostaticSphericalCoriolis(FT::DataType=Float64; rotation_rate=Ω_Earth) =
    HydrostaticSphericalCoriolis{FT}(rotation_rate)

@inline fᵃᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid, coriolis::HydrostaticSphericalCoriolis)
    return @inbounds 2 * coriolis.rotation_rate * sind(grid.ϕᵃᶜᵃ[j])

@inline fᵃᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid, coriolis::HydrostaticSphericalCoriolis)
    return @inbounds 2 * coriolis.rotation_rate * sind(grid.ϕᵃᶠᵃ[j])

@inline x_f_cross_U(i, j, k, grid::RegularLatitudeLongitudeGrid, coriolis::HydrostaticSphericalCoriolis, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, fᵃᶠᵃ, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_vᶜᶠᵃ, U[2]) / Δxᶠᶜᵃ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid::RegularLatitudeLongitudeGrid, coriolis::HydrostaticSphericalCoriolis, U) =
    @inbounds - ℑxᶜᵃᵃ(i, j, k, grid, fᵃᶜᵃ, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_uᶠᶜᵃ, U[1]) / Δyᶜᶠᵃ(i, j, k, grid)

@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, coriolis::HydrostaticSphericalCoriolis, U) where FT = zero(FT)

Base.show(io::IO, hydrostatic_spherical_coriolis::HydrostaticSphericalCoriolis{FT}) where FT =
    print(io, "HydrostaticSphericalCoriolis{$FT}: rotation_rate = ",
          @sprintf("%.2e",  hydrostatic_spherical_coriolis.rotation_rate))
