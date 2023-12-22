using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BuoyancyModels: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, halo_size
using Oceananigans.ImmersedBoundaries
using Oceananigans.Utils: getnamewrapper
using Adapt 
using Printf

import Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ, Δzᶜᶠᶠ, Δzᶜᶠᶜ, Δzᶠᶜᶠ, Δzᶠᶜᶜ, Δzᶠᶠᶠ, Δzᶠᶠᶜ
import Oceananigans.Advection: ∂t_∂s_grid

"""
    GeneralizedVerticalSpacing{R, S, Z} 

spacings for a generalized vertical coordinate system.
The reference coordinate is stored in `Δr`, while `Δ` contains the z-coordinate.
The `s⁻`, `sⁿ` and `∂t_∂s` fields are the vertical derivative of the vertical coordinate (∂Δ/∂Δr)
at timestep `n-1` and `n` and it's time derivative.
`denomination` contains the "type" of generalized vertical coordinate, for example:
- Zstar: free-surface following
- sigma: terrain following
"""
struct GeneralizedVerticalSpacing{D, R, Z, S}
  denomination :: D # The type of generalized coordinate
            Δr :: R # Reference _non moving_ vertical coordinate (one-dimensional)
             Δ :: Z # moving vertical coordinate (three-dimensional)
            s⁻ :: S # scaling term = ∂Δ/∂Δr at the start of the time step
            sⁿ :: S # scaling term = ∂Δ/∂Δr at the end of the time step
         ∂t_∂s :: S # Time derivative of the vertical coordinate scaling divided by the sⁿ
end

Adapt.adapt_structure(to, coord::GeneralizedVerticalSpacing) = 
    GeneralizedVerticalSpacing(nothing, 
                             Adapt.adapt(to, coord.Δr),
                             Adapt.adapt(to, coord.Δ),
                             Adapt.adapt(to, coord.s⁻),
                             Adapt.adapt(to, coord.sⁿ),
                             Adapt.adapt(to, coord.∂t_∂s))

import Oceananigans.Architectures: arch_array

arch_array(arch, coord::GeneralizedVerticalSpacing) = 
    GeneralizedVerticalSpacing(coord.denomination,
                             arch_array(arch, coord.Δr), 
                             coord.Δ,
                             coord.s⁻,
                             coord.sⁿ,
                             coord.∂t_∂s)

const GeneralizedSpacingRG{D}  = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:GeneralizedVerticalSpacing{D}} where D
const GeneralizedSpacingLLG{D} = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:GeneralizedVerticalSpacing{D}} where D

const GeneralizedSpacingUnderlyingGrid{D} = Union{GeneralizedSpacingRG{D}, GeneralizedSpacingLLG{D}} where D
const GeneralizedSpacingImmersedGrid{D} = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:GeneralizedSpacingUnderlyingGrid{D}} where D

const GeneralizedSpacingGrid{D} = Union{GeneralizedSpacingUnderlyingGrid{D}, GeneralizedSpacingImmersedGrid{D}} where D

""" geopotential following vertical coordinate """
struct Z end

#####
##### General implementation
#####

GeneralizedSpacingGrid(grid, coord) = grid
update_vertical_spacing!(model, grid, Δt; kwargs...) = nothing

##### 
##### Vertical spacings for a generalized vertical coordinate system
#####

# Very bad for GPU performance!!! (z-values are not coalesced in memory for z-derivatives anymore)
# TODO: make z-direction local in memory by not using Fields
@inline Δzᶜᶜᶠ(i, j, k, grid::GeneralizedSpacingGrid) = @inbounds grid.Δzᵃᵃᶠ.Δ[i, j, k]
@inline Δzᶜᶜᶜ(i, j, k, grid::GeneralizedSpacingGrid) = @inbounds grid.Δzᵃᵃᶜ.Δ[i, j, k]

@inline Δzᶜᶠᶠ(i, j, k, grid::GeneralizedSpacingGrid) = ℑyᵃᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.Δ)
@inline Δzᶜᶠᶜ(i, j, k, grid::GeneralizedSpacingGrid) = ℑyᵃᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.Δ)

@inline Δzᶠᶜᶠ(i, j, k, grid::GeneralizedSpacingGrid) = ℑxᶠᵃᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.Δ)
@inline Δzᶠᶜᶜ(i, j, k, grid::GeneralizedSpacingGrid) = ℑxᶠᵃᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.Δ)

@inline Δzᶠᶠᶠ(i, j, k, grid::GeneralizedSpacingGrid) = ℑxyᶠᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.Δ)
@inline Δzᶠᶠᶜ(i, j, k, grid::GeneralizedSpacingGrid) = ℑxyᶠᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.Δ)

##### 
##### Vertical velocity of the Δ-surfaces to be included in the continuity equation
#####

∂t_∂s_grid(i, j, k, grid::GeneralizedSpacingGrid) = grid.Δzᵃᵃᶜ.∂t_∂s[i, j, k] 

#####
##### Additional terms to be included in the momentum equations (fallbacks)
#####

@inline grid_slope_contribution_x(i, j, k, grid, args...) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid, args...) = zero(grid)

@inline grid_slope_contribution_x(i, j, k, grid::GeneralizedSpacingGrid, free_surface, ::Nothing, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid::GeneralizedSpacingGrid, free_surface, ::Nothing, model_fields) = zero(grid)

#####
##### Handling tracer update in generalized vertical coordinates (we update sθ)
#####

@kernel function _ab2_step_tracer_generalized_spacing!(θ, sⁿ, s⁻, Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    FT = eltype(χ)
    one_point_five = convert(FT, 1.5)
    oh_point_five  = convert(FT, 0.5)

    @inbounds begin
        ∂t_∂sθ = (one_point_five + χ) * Gⁿ[i, j, k] - (oh_point_five + χ) * G⁻[i, j, k]
        θ[i, j, k] = s⁻[i, j, k] * θ[i, j, k] / sⁿ[i, j, k] + convert(FT, Δt) * ∂t_∂sθ
    end
end

ab2_step_tracer_field!(tracer_field, grid::GeneralizedSpacingGrid, Δt, χ, Gⁿ, G⁻) =
    launch!(architecture(grid), grid, :xyz, _ab2_step_tracer_generalized_spacing!, 
            tracer_field, 
            grid.Δzᵃᵃᶠ.sⁿ, 
            grid.Δzᵃᵃᶠ.s⁻, 
            Δt, χ, Gⁿ, G⁻)