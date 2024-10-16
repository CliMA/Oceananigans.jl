using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BuoyancyModels: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, halo_size, AbstractVerticalCoordinateUnderlyingGrid
using Oceananigans.ImmersedBoundaries
using Oceananigans.ImmersedBoundaries: ImmersedAbstractVerticalCoordinateGrid
using Oceananigans.Utils: getnamewrapper
using Oceananigans.Grids: with_halo, ∂t_grid, vertical_scaling, previous_vertical_scaling
using Adapt 
using Printf

import Oceananigans.Architectures: arch_array

const AbstractVerticalCoordinateGrid = Union{AbstractVerticalCoordinateUnderlyingGrid, ImmersedAbstractVerticalCoordinateGrid}

#####
##### General implementation
#####

update_grid!(model, grid; kwargs...) = nothing

#####
##### Additional terms to be included in the momentum equations (fallbacks)
#####

@inline grid_slope_contribution_x(i, j, k, grid, args...) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid, args...) = zero(grid)

#####
##### Tracer update in generalized vertical coordinates 
##### We advance sθ but store θ once sⁿ⁺¹ is known
#####

@kernel function _ab2_step_tracer_generalized_spacing!(θ, grid, Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    FT = eltype(χ)
    C₁ = convert(FT, 1.5) + χ
    C₂ = convert(FT, 0.5) + χ

    sⁿ = vertical_scaling(i, j, k, grid, Center(), Center(), Center())
    s⁻ = previous_vertical_scaling(i, j, k, grid, Center(), Center(), Center())

    @inbounds begin
        ∂t_sθ = C₁ * sⁿ * Gⁿ[i, j, k] - C₂ * s⁻ * G⁻[i, j, k]
        
        # We store temporarily sθ in θ. the unscaled θ will be retrived later on with `unscale_tracers!`
        θ[i, j, k] = sⁿ * θ[i, j, k] + convert(FT, Δt) * ∂t_sθ
    end
end

ab2_step_tracer_field!(tracer_field, grid::AbstractVerticalCoordinateGrid, Δt, χ, Gⁿ, G⁻) =
    launch!(architecture(grid), grid, :xyz, _ab2_step_tracer_generalized_spacing!, 
            tracer_field, 
            grid, 
            Δt, χ, Gⁿ, G⁻)

const EmptyTuples = Union{NamedTuple{(), Tuple{}}, Tuple{}}

# Fallbacks
unscale_tracers!(tracers, grid; kwargs...) = nothing
unscale_tracers!(::EmptyTuples, ::AbstractVerticalCoordinateGrid; kwargs...) = nothing

tracer_scaling_parameters(param::Symbol, tracers, grid) = KernelParameters((size(grid, 1), size(grid, 2), length(tracers)), (0, 0, 0))
tracer_scaling_parameters(param::KernelParameters{S, O}, tracers, grid) where {S, O} = KernelParameters((S..., length(tracers)), (O..., 0))

function unscale_tracers!(tracers, grid::AbstractVerticalCoordinateGrid; parameters = :xy) 
    parameters = tracer_scaling_parameters(parameters, tracers, grid)
    
    launch!(architecture(grid), grid, parameters, _unscale_tracers!, tracers, grid, 
            Val(grid.Hz), Val(grid.Nz))
    
    return nothing
end
    
@kernel function _unscale_tracers!(tracers, grid, ::Val{Hz}, ::Val{Nz}) where {Hz, Nz}
    i, j, n = @index(Global, NTuple)

    @unroll for k in -Hz+1:Nz+Hz
        tracers[n][i, j, k] /= vertical_scaling(i, j, k, grid, Center(), Center(), Center())
    end
end