using Oceananigans.Grids: LatitudeLongitudeGrid, ConformalCubedSphereFaceGrid
using Oceananigans.Operators: Δx_qᶜᶠᶜ, Δy_qᶠᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ, hack_sind
using Oceananigans.Advection: EnergyConservingScheme, EnstrophyConservingScheme

# Our two Coriolis schemes are energy-conserving or enstrophy-conserving
# with a "vector invariant" momentum advection scheme, but not with a "flux form"
# or "conservation form" advection scheme (which does not currently exist for
# curvilinear grids).

"""
    HydrostaticSphericalCoriolis{FT} <: AbstractRotation

A parameter object for constant rotation around a vertical axis.
"""
struct HydrostaticSphericalCoriolis{S, FT} <: AbstractRotation
    rotation_rate :: FT
    scheme :: S
end

"""
    HydrostaticSphericalCoriolis([FT=Float64;] rotation_rate=Ω_Earth, scheme=EnergyConservingScheme()))

Returns a parameter object for Coriolis forces on a sphere rotating at `rotation_rate`.
By default, `rotation_rate` is assumed to be Earth's.
"""
HydrostaticSphericalCoriolis(FT::DataType=Float64; rotation_rate=Ω_Earth, scheme::S=EnergyConservingScheme()) where S =
    HydrostaticSphericalCoriolis{S, FT}(rotation_rate, scheme)

@inline φᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.φᵃᶠᵃ[j]
@inline φᶠᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.φᶠᶠᵃ[i, j]

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::HydrostaticSphericalCoriolis) =
    2 * coriolis.rotation_rate * hack_sind(φᶠᶠᵃ(i, j, k, grid))

@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, coriolis::HydrostaticSphericalCoriolis, U) where FT = zero(FT)

#####
##### Enstrophy-conserving scheme
#####

const CoriolisEnstrophyConserving = HydrostaticSphericalCoriolis{<:EnstrophyConservingScheme}

@inline x_f_cross_U(i, j, k, grid, coriolis::CoriolisEnstrophyConserving, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, U[2]) / Δxᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, coriolis::CoriolisEnstrophyConserving, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, U[1]) / Δyᶜᶠᶜ(i, j, k, grid)

#####
##### Energy-conserving scheme
#####

const CoriolisEnergyConserving = HydrostaticSphericalCoriolis{<:EnergyConservingScheme}

@inline f_ℑx_vᶠᶠᵃ(i, j, k, grid, coriolis, v) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)
@inline f_ℑy_uᶠᶠᵃ(i, j, k, grid, coriolis, u) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)

@inline x_f_cross_U(i, j, k, grid, coriolis::CoriolisEnergyConserving, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_vᶠᶠᵃ, coriolis, U[2]) / Δxᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, coriolis::CoriolisEnergyConserving, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_uᶠᶠᵃ, coriolis, U[1]) / Δyᶜᶠᶜ(i, j, k, grid)

#####
##### Show
#####

Base.show(io::IO, hydrostatic_spherical_coriolis::HydrostaticSphericalCoriolis{FT}) where FT =
    print(io, "HydrostaticSphericalCoriolis{$FT}: rotation_rate = ",
          @sprintf("%.2e",  hydrostatic_spherical_coriolis.rotation_rate))
