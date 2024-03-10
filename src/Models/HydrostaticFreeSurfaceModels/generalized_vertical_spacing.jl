using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BuoyancyModels: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, halo_size
using Oceananigans.ImmersedBoundaries
using Oceananigans.Utils: getnamewrapper
using Adapt 
using Printf

import Oceananigans.Grids: with_halo
import Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ, Δzᶜᶠᶠ, Δzᶜᶠᶜ, Δzᶠᶜᶠ, Δzᶠᶜᶜ, Δzᶠᶠᶠ, Δzᶠᶠᶜ
import Oceananigans.Advection: ∂t_∂s_grid
import Oceananigans.Architectures: arch_array

"""
    GeneralizedVerticalSpacing{D, R, S, Z} 

spacings for a generalized vertical coordinate system. The reference (non-moving) spacings are stored in `Δr`. 
`Δ` contains the spacings associated with the moving coordinate system.
`s⁻`, `sⁿ` and `∂t_∂s` fields are the vertical derivative of the vertical coordinate (∂Δ/∂Δr)
at timestep `n-1` and `n` and it's time derivative.
`denomination` contains the "type" of generalized vertical coordinate (the only one implemented is `ZStar`)
"""
struct GeneralizedVerticalSpacing{D, R, Z, S, SN, ST}
  denomination :: D  # The type of generalized coordinate
            Δr :: R  # Reference _non moving_ vertical coordinate (one-dimensional)
             Δ :: Z  # moving vertical coordinate (three-dimensional)
            s⁻ :: S  # scaling term = ∂Δ/∂Δr at the start of the time step
            sⁿ :: SN # scaling term = ∂Δ/∂Δr at the end of the time step
         ∂t_∂s :: ST # Time derivative of the vertical coordinate scaling 
end

Adapt.adapt_structure(to, coord::GeneralizedVerticalSpacing) = 
    GeneralizedVerticalSpacing(coord.denomination, 
                               Adapt.adapt(to, coord.Δr),
                               Adapt.adapt(to, coord.Δ),
                               Adapt.adapt(to, coord.s⁻),
                               Adapt.adapt(to, coord.sⁿ),
                               Adapt.adapt(to, coord.∂t_∂s))

on_architecture(arch, coord::GeneralizedVerticalSpacing) = 
    GeneralizedVerticalSpacing(on_architecture(arch, coord.denomination),
                               on_architecture(arch, coord.Δr), 
                               on_architecture(arch, coord.Δ),
                               on_architecture(arch, coord.s⁻),
                               on_architecture(arch, coord.sⁿ),
                               on_architecture(arch, coord.∂t_∂s))

const GeneralizedSpacingRG{D}  = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:GeneralizedVerticalSpacing{D}} where D
const GeneralizedSpacingLLG{D} = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:GeneralizedVerticalSpacing{D}} where D

const GeneralizedSpacingUnderlyingGrid{D} = Union{GeneralizedSpacingRG{D}, GeneralizedSpacingLLG{D}} where D
const GeneralizedSpacingImmersedGrid{D} = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:GeneralizedSpacingUnderlyingGrid{D}} where D

const GeneralizedSpacingGrid{D} = Union{GeneralizedSpacingUnderlyingGrid{D}, GeneralizedSpacingImmersedGrid{D}} where D

denomination(grid::GeneralizedSpacingGrid) = grid.Δzᵃᵃᶠ.denomination

#####
##### Original grid
#####

retrieve_static_grid(grid) = grid

function retrieve_static_grid(grid::GeneralizedSpacingGrid) 

    Δzᵃᵃᶠ = grid.Δzᵃᵃᶠ.Δr
    Δzᵃᵃᶜ = grid.Δzᵃᵃᶜ.Δr

    TX, TY, TZ = topology(grid)

    args = []
    for prop in propertynames(grid)
        if prop == :Δzᵃᵃᶠ
            push!(args, Δzᵃᵃᶠ)
        elseif prop == :Δzᵃᵃᶜ
            push!(args, Δzᵃᵃᶜ)
        else
            push!(args, getproperty(grid, prop))
        end
    end

    GridType = getnamewrapper(grid)

    return GridType{TX, TY, TZ}(args...)
end

#####
##### Some extensions
#####

function with_halo(new_halo, grid::GeneralizedSpacingGrid)
    old_static_grid = retrieve_static_grid(grid)
    new_static_grid = with_halo(new_halo, old_static_grid)
    vertical_coordinate = denomination(grid)
    new_grid = GeneralizedSpacingGrid(new_static_grid, vertical_coordinate)

    return new_grid
end

function with_halo(halo, ibg::GeneralizedSpacingImmersedGrid) 
    underlying_grid = ibg.underlying_grid
    immersed_boundary = ibg.immersed_boundary
    old_static_grid = retrieve_static_grid(underlying_grid)
    new_static_grid = with_halo(new_halo, underlying_grid)
    vertical_coordinate = denomination(underlying_grid)
    active_cells_map = isnothing(ibg.interior_active_cells)
    new_grid = GeneralizedSpacingGrid(new_static_grid, vertical_coordinate)
    new_ibg = ImmersedBoundaryGrid(new_grid, immersed_boundary; active_cells_map)
    return new_ibg
end

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
# TODO: make it work with partial cells 

@inline Δzᶜᶜᶠ(i, j, k, grid::GeneralizedSpacingGrid) = @inbounds grid.Δzᵃᵃᶠ.Δ[i, j, k]
@inline Δzᶜᶜᶜ(i, j, k, grid::GeneralizedSpacingGrid) = @inbounds grid.Δzᵃᵃᶜ.Δ[i, j, k]

@inline Δzᶜᶠᶠ(i, j, k, grid::GeneralizedSpacingGrid) = ℑyᵃᶠᵃ(i, j, k, grid, Δzᶜᶜᶠ)
@inline Δzᶜᶠᶜ(i, j, k, grid::GeneralizedSpacingGrid) = ℑyᵃᶠᵃ(i, j, k, grid, Δzᶜᶜᶜ)

@inline Δzᶠᶜᶠ(i, j, k, grid::GeneralizedSpacingGrid) = ℑxᶠᵃᵃ(i, j, k, grid, Δzᶜᶜᶠ)
@inline Δzᶠᶜᶜ(i, j, k, grid::GeneralizedSpacingGrid) = ℑxᶠᵃᵃ(i, j, k, grid, Δzᶜᶜᶜ)

@inline Δzᶠᶠᶠ(i, j, k, grid::GeneralizedSpacingGrid) = ℑxyᶠᶠᵃ(i, j, k, grid, Δzᶜᶜᶠ)
@inline Δzᶠᶠᶜ(i, j, k, grid::GeneralizedSpacingGrid) = ℑxyᶠᶠᵃ(i, j, k, grid, Δzᶜᶜᶜ)

@inline Δz_reference(i, j, k, Δz::Number) = Δz
@inline Δz_reference(i, j, k, Δz::AbstractVector) = Δz[k]

@inline Δz_reference(i, j, k, Δz::GeneralizedVerticalSpacing{<:Any, <:Number}) = Δz.Δr
@inline Δz_reference(i, j, k, Δz::GeneralizedVerticalSpacing) = Δz.Δr[k]

@inline Δzᶜᶜᶠ_reference(i, j, k, grid) = Δz_reference(i, j, k, grid.Δzᵃᵃᶠ)
@inline Δzᶜᶜᶜ_reference(i, j, k, grid) = Δz_reference(i, j, k, grid.Δzᵃᵃᶜ)

@inline Δzᶜᶠᶠ_reference(i, j, k, grid) = ℑyᵃᶠᵃ(i, j, k, grid, Δzᶜᶜᶠ_reference)
@inline Δzᶜᶠᶜ_reference(i, j, k, grid) = ℑyᵃᶠᵃ(i, j, k, grid, Δzᶜᶜᶜ_reference)

@inline Δzᶠᶜᶠ_reference(i, j, k, grid) = ℑxᶠᵃᵃ(i, j, k, grid, Δzᶜᶜᶠ_reference)
@inline Δzᶠᶜᶜ_reference(i, j, k, grid) = ℑxᶠᵃᵃ(i, j, k, grid, Δzᶜᶜᶜ_reference)

@inline Δzᶠᶠᶠ_reference(i, j, k, grid) = ℑxyᶠᶠᵃ(i, j, k, grid, Δzᶜᶜᶠ_reference)
@inline Δzᶠᶠᶜ_reference(i, j, k, grid) = ℑxyᶠᶠᵃ(i, j, k, grid, Δzᶜᶜᶜ_reference)

##### 
##### Vertical velocity of the Δ-surfaces to be included in the continuity equation
#####

∂t_∂s_grid(i, j, k, grid::GeneralizedSpacingGrid) = grid.Δzᵃᵃᶜ.∂t_∂s[i, j, k] 

#####
##### Additional terms to be included in the momentum equations (fallbacks)
#####

@inline grid_slope_contribution_x(i, j, k, grid, args...) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid, args...) = zero(grid)

#####
##### Tracer update in generalized vertical coordinates 
##### We advance sθ but store θ once sⁿ⁺¹ is known
#####

@kernel function _ab2_step_tracer_generalized_spacing!(θ, sⁿ, s⁻, Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    FT = eltype(χ)
    one_point_five = convert(FT, 1.5)
    oh_point_five  = convert(FT, 0.5)

    @inbounds begin
        ∂t_∂sθ = (one_point_five + χ) * sⁿ[i, j, k] * Gⁿ[i, j, k] - (oh_point_five + χ) * s⁻[i, j, k] * G⁻[i, j, k]
        
        # We store temporarily sθ in θ. the unscaled θ will be retrived later on with `scale_tracers!`
        θ[i, j, k] = sⁿ[i, j, k] * θ[i, j, k] + convert(FT, Δt) * ∂t_∂sθ
    end
end

ab2_step_tracer_field!(tracer_field, grid::GeneralizedSpacingGrid, Δt, χ, Gⁿ, G⁻) =
    launch!(architecture(grid), grid, :xyz, _ab2_step_tracer_generalized_spacing!, 
            tracer_field, 
            grid.Δzᵃᵃᶠ.sⁿ, 
            grid.Δzᵃᵃᶠ.s⁻, 
            Δt, χ, Gⁿ, G⁻)

const EmptyTuples = Union{NamedTuple{(),Tuple{}}, Tuple{}}

scale_tracers!(::EmptyTuples, ::GeneralizedSpacingGrid; kwargs...) = nothing

tracer_scaling_parameters(param::Symbol, tracers, grid) = KernelParameters((size(grid, 1), size(grid, 2), length(tracers)), (0, 0, 0))
tracer_scaling_parameters(param::KernelParameters{S, O}, tracers, grid) where {S, O} = KernelParameters((S..., length(tracers)), (O..., 0))

function scale_tracers!(tracers, grid::GeneralizedSpacingGrid; parameters = :xy) 
    parameters = tracer_scaling_parameters(parameters, tracers, grid)
    launch!(architecture(grid), grid, parameters, _scale_tracers, tracers, grid.Δzᵃᵃᶠ.sⁿ, 
            Val(grid.Hz), Val(grid.Nz))
    return nothing
end
    
@kernel function _scale_tracers(tracers, sⁿ, ::Val{Hz}, ::Val{Nz}) where {Hz, Nz}
    i, j, n = @index(Global, NTuple)

    @unroll for k in -Hz+1:Nz+Hz
        tracers[n][i, j, k] /= sⁿ[i, j, k]
    end
end