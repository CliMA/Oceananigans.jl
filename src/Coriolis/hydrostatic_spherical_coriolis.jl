using Oceananigans.Grids: RegularLatitudeLongitudeGrid
using Oceananigans.Operators: Δx_vᶜᶠᵃ, Δy_uᶠᶜᵃ, Δxᶠᶜᵃ, Δyᶜᶠᵃ

# Our two Coriolis schemes are energy-conserving or enstrophy-conserving
# with a "vector invariant" momentum advection scheme, but not with a "flux form"
# or "conservation form" advection scheme (which does not currently exist for
# curvilinear grids).
struct VectorInvariantEnergyConserving end
struct VectorInvariantEnstrophyConserving end

"""
    HydrostaticSphericalCoriolis{FT} <: AbstractRotation

A parameter object for constant rotation around a vertical axis.
"""
struct HydrostaticSphericalCoriolis{S, FT} <: AbstractRotation
    rotation_rate :: FT
    scheme :: S
end

"""
    HydrostaticSphericalCoriolis([FT=Float64;] rotation_rate=Ω_Earth)

Returns a parameter object for Coriolis forces on a sphere rotating at `rotation_rate`.
By default, `rotation_rate` is assumed to be Earth's.
"""
HydrostaticSphericalCoriolis(FT::DataType=Float64; rotation_rate=Ω_Earth, scheme::S=VectorInvariantEnergyConserving()) where S =
    HydrostaticSphericalCoriolis{S, FT}(rotation_rate, scheme)

@inline fᵃᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid, coriolis::HydrostaticSphericalCoriolis) =
    @inbounds 2 * coriolis.rotation_rate * sind(grid.ϕᵃᶠᵃ[j])

@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, coriolis::HydrostaticSphericalCoriolis, U) where FT = zero(FT)

#####
##### Enstrophy-conserving scheme
#####

const VIEnstrophy = HydrostaticSphericalCoriolis{<:VectorInvariantEnstrophyConserving}

@inline x_f_cross_U(i, j, k, grid::RegularLatitudeLongitudeGrid, coriolis::VIEnstrophy, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, fᵃᶠᵃ, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_vᶜᶠᵃ, U[2]) / Δxᶠᶜᵃ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid::RegularLatitudeLongitudeGrid, coriolis::VIEnstrophy, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, fᵃᶠᵃ, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_uᶠᶜᵃ, U[1]) / Δyᶜᶠᵃ(i, j, k, grid)

#####
##### Energy-conserving scheme
#####

const VIEnergy = HydrostaticSphericalCoriolis{<:VectorInvariantEnergyConserving}

@inline f_ℑx_vᶠᶠᵃ(i, j, k, grid, coriolis, v) = fᵃᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_vᶜᶠᵃ, v)
@inline f_ℑy_uᶠᶠᵃ(i, j, k, grid, coriolis, u) = fᵃᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_uᶠᶜᵃ, u)

@inline x_f_cross_U(i, j, k, grid::RegularLatitudeLongitudeGrid, coriolis::VIEnergy, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_vᶠᶠᵃ, coriolis, U[2]) / Δxᶠᶜᵃ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid::RegularLatitudeLongitudeGrid, coriolis::VIEnergy, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_uᶠᶠᵃ, coriolis, U[1]) / Δyᶜᶠᵃ(i, j, k, grid)  

#####
##### Show
#####

Base.show(io::IO, hydrostatic_spherical_coriolis::HydrostaticSphericalCoriolis{FT}) where FT =
    print(io, "HydrostaticSphericalCoriolis{$FT}: rotation_rate = ",
          @sprintf("%.2e",  hydrostatic_spherical_coriolis.rotation_rate))
