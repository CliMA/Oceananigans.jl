using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BuoyancyModels: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, halo_size
using Oceananigans.ImmersedBoundaries
using Oceananigans.Utils: getnamewrapper
using Adapt 
using Printf

import Oceananigans.Grids: with_halo, znode
import Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ, Δzᶜᶠᶠ, Δzᶜᶠᶜ, Δzᶠᶜᶠ, Δzᶠᶜᶜ, Δzᶠᶠᶠ, Δzᶠᶠᶜ
import Oceananigans.Advection: V_times_∂t_s_grid
import Oceananigans.Architectures: arch_array

"""
    AbstractVerticalSpacing{R} 

spacings for a generalized vertical coordinate system. The reference (non-moving) spacings are stored in `Δr`. 
`Δ` contains the spacings associated with the moving coordinate system.
`s⁻`, `sⁿ` and `∂t_s` fields are the vertical derivative of the vertical coordinate (∂Δr/∂Δz)
at timestep `n-1` and `n` and it's time derivative.
`denomination` contains the "type" of generalized vertical coordinate (the only one implemented is `ZStar`)
"""
abstract type AbstractVerticalSpacing{R} end

const AbstractVerticalSpacingRG  = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractVerticalSpacing} 
const AbstractVerticalSpacingLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractVerticalSpacing}

const AbstractVerticalSpacingUnderlyingGrid = Union{AbstractVerticalSpacingRG, AbstractVerticalSpacingLLG}
const AbstractVerticalSpacingImmersedGrid   = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractVerticalSpacingUnderlyingGrid}

const AbstractVerticalSpacingGrid = Union{AbstractVerticalSpacingUnderlyingGrid, AbstractVerticalSpacingImmersedGrid} 

#####
##### Original grid
#####

retrieve_static_grid(grid) = grid

function retrieve_static_grid(grid::AbstractVerticalSpacingImmersedGrid) 
    underlying_grid  = retrieve_static_grid(grid.underlying_grid)
    active_cells_map = !isnothing(grid.interior_active_cells)
    return ImmersedBoundaryGrid(underlying_grid, grid.immersed_boundary; active_cells_map)
end

reference_zspacings(grid, ::Face)   = grid.Δzᵃᵃᶠ
reference_zspacings(grid, ::Center) = grid.Δzᵃᵃᶜ

function retrieve_static_grid(grid::AbstractVerticalSpacingUnderlyingGrid) 

    Δzᵃᵃᶠ = reference_zspacings(grid, Face())
    Δzᵃᵃᶜ = reference_zspacings(grid, Center())

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

function with_halo(new_halo, grid::AbstractVerticalSpacingUnderlyingGrid)
    old_static_grid = retrieve_static_grid(grid)
    new_static_grid = with_halo(new_halo, old_static_grid)
    vertical_coordinate = denomination(grid)
    new_grid = generalized_spacing_grid(new_static_grid, vertical_coordinate)

    return new_grid
end

function with_halo(new_halo, ibg::AbstractVerticalSpacingImmersedGrid) 
    underlying_grid = ibg.underlying_grid
    immersed_boundary = ibg.immersed_boundary
    new_underlying_grid = with_halo(new_halo, underlying_grid)
    active_cells_map = isnothing(ibg.interior_active_cells)
    new_ibg = ImmersedBoundaryGrid(new_underlying_grid, immersed_boundary; active_cells_map)
    return new_ibg
end

#####
##### General implementation
#####

generalized_spacing_grid(grid, coord) = grid
update_vertical_spacing!(model, grid; kwargs...) = nothing

#####
##### Reference (local) vertical spacings in `r-coordinates`
#####

@inline Δr(i, j, k, Δz::Number) = Δz
@inline Δr(i, j, k, Δz::AbstractVector) = @inbounds Δz[k]

@inline Δr(i, j, k, Δz::AbstractVerticalSpacing{<:Number}) = Δz.Δr
@inline Δr(i, j, k, Δz::AbstractVerticalSpacing) = @inbounds Δz.Δr[k]

@inline Δrᶜᶜᶠ(i, j, k, grid) = Δr(i, j, k, grid.Δzᵃᵃᶠ)
@inline Δrᶜᶜᶜ(i, j, k, grid) = Δr(i, j, k, grid.Δzᵃᵃᶜ)

@inline Δrᶜᶠᶠ(i, j, k, grid) = ℑyᵃᶠᵃ(i, j, k, grid, Δrᶜᶜᶠ)
@inline Δrᶜᶠᶜ(i, j, k, grid) = ℑyᵃᶠᵃ(i, j, k, grid, Δrᶜᶜᶜ)

@inline Δrᶠᶜᶠ(i, j, k, grid) = ℑxᶠᵃᵃ(i, j, k, grid, Δrᶜᶜᶠ)
@inline Δrᶠᶜᶜ(i, j, k, grid) = ℑxᶠᵃᵃ(i, j, k, grid, Δrᶜᶜᶜ)

@inline Δrᶠᶠᶠ(i, j, k, grid) = ℑxyᶠᶠᵃ(i, j, k, grid, Δrᶜᶜᶠ)
@inline Δrᶠᶠᶜ(i, j, k, grid) = ℑxyᶠᶠᵃ(i, j, k, grid, Δrᶜᶜᶜ)

##### 
##### Vertical spacings for a generalized vertical coordinate system
#####

# Fallbacks
@inline previous_vertical_scaling(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline vertical_scaling(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)

@inline Δzᶜᶜᶠ(i, j, k, grid::AbstractVerticalSpacingGrid) = @inbounds Δrᶜᶜᶠ(i, j, k, grid) * vertical_scaling(i, j, k, grid, c, c, f)
@inline Δzᶜᶜᶜ(i, j, k, grid::AbstractVerticalSpacingGrid) = @inbounds Δrᶜᶜᶜ(i, j, k, grid) * vertical_scaling(i, j, k, grid, c, c, c)
@inline Δzᶠᶜᶠ(i, j, k, grid::AbstractVerticalSpacingGrid) = @inbounds Δrᶠᶜᶠ(i, j, k, grid) * vertical_scaling(i, j, k, grid, f, c, f)
@inline Δzᶠᶜᶜ(i, j, k, grid::AbstractVerticalSpacingGrid) = @inbounds Δrᶠᶜᶜ(i, j, k, grid) * vertical_scaling(i, j, k, grid, f, c, c)
@inline Δzᶜᶠᶠ(i, j, k, grid::AbstractVerticalSpacingGrid) = @inbounds Δrᶜᶠᶠ(i, j, k, grid) * vertical_scaling(i, j, k, grid, c, f, f)
@inline Δzᶜᶠᶜ(i, j, k, grid::AbstractVerticalSpacingGrid) = @inbounds Δrᶜᶠᶜ(i, j, k, grid) * vertical_scaling(i, j, k, grid, f, c, c)
@inline Δzᶠᶠᶠ(i, j, k, grid::AbstractVerticalSpacingGrid) = @inbounds Δrᶠᶠᶠ(i, j, k, grid) * vertical_scaling(i, j, k, grid, f, f, f)
@inline Δzᶠᶠᶜ(i, j, k, grid::AbstractVerticalSpacingGrid) = @inbounds Δrᶠᶠᶜ(i, j, k, grid) * vertical_scaling(i, j, k, grid, f, f, c)

#####
##### Additional terms to be included in the momentum equations (fallbacks)
#####

@inline grid_slope_contribution_x(i, j, k, grid, args...) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid, args...) = zero(grid)

@inline ∂t_s_grid(i, j, k, grid) = zero(grid)

#####
##### Tracer update in generalized vertical coordinates 
##### We advance sθ but store θ once sⁿ⁺¹ is known
#####

@kernel function _ab2_step_tracer_generalized_spacing!(θ, grid, Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    FT = eltype(χ)
    C₁ = convert(FT, 1.5) + χ
    C₂ = convert(FT, 0.5) + χ

    sⁿ = vertical_scaling(i, j, k, grid, c, c, c)
    s⁻ = previous_vertical_scaling(i, j, k, grid, c, c, c)

    @inbounds begin
        ∂t_sθ = C₁ * sⁿ * Gⁿ[i, j, k] - C₂ * s⁻ * G⁻[i, j, k]
        
        # We store temporarily sθ in θ. the unscaled θ will be retrived later on with `unscale_tracers!`
        θ[i, j, k] = sⁿ * θ[i, j, k] + convert(FT, Δt) * ∂t_sθ
    end
end

ab2_step_tracer_field!(tracer_field, grid::AbstractVerticalSpacingGrid, Δt, χ, Gⁿ, G⁻) =
    launch!(architecture(grid), grid, :xyz, _ab2_step_tracer_generalized_spacing!, 
            tracer_field, 
            grid, 
            Δt, χ, Gⁿ, G⁻)

const EmptyTuples = Union{NamedTuple{(), Tuple{}}, Tuple{}}

unscale_tracers!(::EmptyTuples, ::AbstractVerticalSpacingGrid; kwargs...) = nothing

tracer_scaling_parameters(param::Symbol, tracers, grid) = KernelParameters((size(grid, 1), size(grid, 2), length(tracers)), (0, 0, 0))
tracer_scaling_parameters(param::KernelParameters{S, O}, tracers, grid) where {S, O} = KernelParameters((S..., length(tracers)), (O..., 0))

function unscale_tracers!(tracers, grid::AbstractVerticalSpacingGrid; parameters = :xy) 
    parameters = tracer_scaling_parameters(parameters, tracers, grid)
    
    launch!(architecture(grid), grid, parameters, _unscale_tracers!, tracers, grid, 
            Val(grid.Hz), Val(grid.Nz))
    
    return nothing
end
    
@kernel function _unscale_tracers!(tracers, grid, ::Val{Hz}, ::Val{Nz}) where {Hz, Nz}
    i, j, n = @index(Global, NTuple)

    @unroll for k in -Hz+1:Nz+Hz
        tracers[n][i, j, k] /= vertical_scaling(i, j, k, grid, c, c, c)
    end
end