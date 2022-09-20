using Oceananigans.Grids: LatitudeLongitudeGrid, ConformalCubedSphereFaceGrid, peripheral_node
using Oceananigans.Operators: Δx_qᶜᶠᶜ, Δy_qᶠᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ, hack_sind
using Oceananigans.Advection: EnergyConservingScheme, EnstrophyConservingScheme


"""
    struct WetCellEnstrophyConservingScheme

A parameter object for an enstrophy-conserving Coriolis scheme that excludes dry edges (indices for which `peripheral_node == true`)
from the velocity interpolation
"""
struct WetCellEnstrophyConservingScheme end

# Our three Coriolis schemes are energy-conserving, enstrophy-conserving and wet-point enstrophy-conserving
# with a "vector invariant" momentum advection scheme, but not with 3a "flux form"
# or "conservation form" advection scheme (which does not currently exist for
# curvilinear grids).
# The wet point enstrophy-conserving coriolis scheme eliminates the dry edges from 
# the velocity interpolation

"""
    struct HydrostaticSphericalCoriolis{S, FT} <: AbstractRotation

A parameter object for constant rotation around a vertical axis.
"""
struct HydrostaticSphericalCoriolis{S, FT} <: AbstractRotation
    rotation_rate :: FT
    scheme :: S
end

"""
    HydrostaticSphericalCoriolis([FT=Float64;]
                                 rotation_rate = Ω_Earth,
                                 scheme = EnergyConservingScheme())

Returns a parameter object for Coriolis forces on a sphere rotating at `rotation_rate`.
By default, `rotation_rate` is assumed to be Earth's.

Keyword arguments
=================

- `scheme`: Either `EnergyConservingScheme()` (default), `EnstrophyConservingScheme()`, or `WetCellEnstrophyConservingScheme()`.
"""
HydrostaticSphericalCoriolis(FT::DataType=Float64; rotation_rate=Ω_Earth, scheme::S=EnergyConservingScheme()) where S =
    HydrostaticSphericalCoriolis{S, FT}(rotation_rate, scheme)

@inline φᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = @inbounds grid.φᵃᶠᵃ[j]
@inline φᶠᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.φᶠᶠᵃ[i, j]

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::HydrostaticSphericalCoriolis) =
    2 * coriolis.rotation_rate * hack_sind(φᶠᶠᵃ(i, j, k, grid))

@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, coriolis::HydrostaticSphericalCoriolis, U) where FT = zero(FT)

#####
##### Wet Point Enstrophy-conserving scheme
#####

# It might happen that a cell is wet but all the neighbouring staggered nodes are dry,
# (an example is a 1-cell large channel)
# In that case the Coriolis force is equal to zero

const CoriolisWetCellEnstrophyConserving = HydrostaticSphericalCoriolis{<:WetCellEnstrophyConservingScheme}

@inline revert_peripheral_node(i, j, k, grid, f::Function, args...) = @inbounds 1.0 - f(i, j, k, grid, args...)

@inline function mask_dry_points_ℑxyᶠᶜᵃ(i, j, k, grid, f::Function, args...) 
    neighbouring_wet_nodes = @inbounds ℑxyᶠᶜᵃ(i, j, k, grid, revert_peripheral_node, peripheral_node, Center(), Face(), Center())
    return ifelse(neighbouring_wet_nodes == 0, zero(grid),
           @inbounds ℑxyᶠᶜᵃ(i, j, k, grid, f, args...) / neighbouring_wet_nodes)
end

@inline function mask_dry_points_ℑxyᶜᶠᵃ(i, j, k, grid, f::Function, args...) 
    neighbouring_wet_nodes = @inbounds ℑxyᶜᶠᵃ(i, j, k, grid, revert_peripheral_node, peripheral_node, Face(), Center(), Center())
    return ifelse(neighbouring_wet_nodes == 0, zero(grid),
           @inbounds ℑxyᶜᶠᵃ(i, j, k, grid, f, args...) / neighbouring_wet_nodes)
end

@inline x_f_cross_U(i, j, k, grid, coriolis::CoriolisWetCellEnstrophyConserving, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * mask_dry_points_ℑxyᶠᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, U[2]) / Δxᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, coriolis::CoriolisWetCellEnstrophyConserving, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * mask_dry_points_ℑxyᶜᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, U[1]) / Δyᶜᶠᶜ(i, j, k, grid)

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

function Base.show(io::IO, hydrostatic_spherical_coriolis::HydrostaticSphericalCoriolis) 

    rotation_rate = hydrostatic_spherical_coriolis.rotation_rate
    coriolis_scheme = hydrostatic_spherical_coriolis.scheme
    rotation_rate_Earth = rotation_rate / Ω_Earth

    return print(io, "HydrostaticSphericalCoriolis", '\n',
                 "├─ rotation rate: " * @sprintf("%.2e", rotation_rate) * " s⁻¹ = " * @sprintf("%.2e", rotation_rate_Earth) * " Ω_Earth", '\n',
                 "└─ scheme: $(summary(coriolis_scheme))")
end
