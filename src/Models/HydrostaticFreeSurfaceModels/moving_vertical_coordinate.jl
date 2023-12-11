using Oceananigans
using Oceananigans.Grids
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, halo_size
using Oceananigans.ImmersedBoundaries
using Oceananigans.Utils: getnamewrapper
using Adapt 

struct ZCoordinate end

struct ZStarCoordinate{R, S, Z}
    reference :: R
      scaling :: S
   star_value :: Z
end

ZStarCoordinate() = ZStarCoordinate(nothing, nothing, nothing)

Adapt.adapt_structure(to, coord::ZStarCoordinate) = 
    ZStarCoordinate(Adapt.adapt(to, coord.reference),
                    Adapt.adapt(to, coord.scaling),
                    Adapt.adapt(to, coord.star_value))

const ZStarCoordinateRG  = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ZStarCoordinate}
const ZStarCoordinateLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ZStarCoordinate}

const ZStarCoordinateUnderlyingGrid = Union{ZStarCoordinateRG, ZStarCoordinateLLG}
const ZStarCoordinateImmersedGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarCoordinateUnderlyingGrid}

const ZStarCoordinateGrid = Union{ZStarCoordinateUnderlyingGrid, ZStarCoordinateImmersedGrid}

MovingCoordinateGrid(grid, coord) = grid

function MovingCoordinateGrid(grid::ImmersedBoundaryGrid, ::ZStarCoordinate)
    underlying_grid = MovingCoordinateGrid(grid.underlying_grid, ZStarCoordinate())
    active_cells_map = !isnothing(grid.active_cells_map)

    return ImmersedBoundaryGrid(underlying_grid, grid.immersed_boundary; active_cells_map)
end

function MovingCoordinateGrid(grid::AbstractUnderlyingGrid{FT, TX, TY, TZ}, ::ZStarCoordinate) where {FT, TX, TY, TZ}
    # Memory layout for Dz spacings is local in z
    ΔzF =  ZFaceField(grid)
    ΔzC = CenterField(grid)
    scaling = ZFaceField(grid, indices = (:, :, grid.Nz + 1))

    Δzᵃᵃᶠ = ZStarCoordinate(grid.Δzᵃᵃᶠ, scaling, ΔzF)
    Δzᵃᵃᶜ = ZStarCoordinate(grid.Δzᵃᵃᶜ, scaling, ΔzC)

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

# Fallback
update_vertical_coordinate!(model, grid; kwargs...) = nothing

function update_vertical_coordinate!(model, grid::ZStarCoordinateGrid; parameters = tuple(:xyz))
    η = model.free_surface.η
    
    # Scaling 
    scaling = grid.Δzᵃᵃᶠ.scaling

    # Moving coordinates
    Δzᵃᵃᶠ  = grid.Δzᵃᵃᶠ.star_value
    Δzᵃᵃᶜ  = grid.Δzᵃᵃᶜ.star_value

    # Reference coordinates
    Δz₀ᵃᵃᶠ = grid.Δzᵃᵃᶠ.reference
    Δz₀ᵃᵃᶜ = grid.Δzᵃᵃᶜ.reference

    # Update the scaling on the whole grid (from -H to N+H)
    launch!(architecture(grid), grid, horizontal_parameters(grid), _update_scaling!,
            scaling, η, grid)

    # Update vertical coordinate with available parameters 
    for params in parameters
        launch!(architecture(grid), grid, params, _update_z_star!, 
                Δzᵃᵃᶠ, Δzᵃᵃᶜ, Δz₀ᵃᵃᶠ, Δz₀ᵃᵃᶜ, scaling, grid.Nz)
    end

    return nothing
end

function horizontal_parameters(grid)
    halos = halo_size(grid)[1:2]
    total_size = size(grid)[1:2] .+ 2 .* halos
    return KernelParameters(total_size, .- halos)
end

@kernel function _update_scaling!(scaling, η, grid)
    i, j = @index(Global, NTuple)
    bottom = bottom_height(grid, i, j)
    @inbounds scaling[i, j, grid.Nz+1] = (bottom + η[i, j, grid.Nz+1]) / bottom
end

bottom_height(grid, i, j) = grid.Lz
bottom_height(grid::ImmersedBoundaryGrid, i, j) = @inbounds - grid.immersed_boundary.bottom_height[i, j, 1]

@kernel function _update_z_star!(ΔzF, ΔzC, ΔzF₀, ΔzC₀, scaling, Nz)
    i, j, k = @index(Global, NTuple)
    @inbounds ΔzF[i, j, k] = scaling[i, j, Nz+1] * ΔzF₀[k]
    @inbounds ΔzC[i, j, k] = scaling[i, j, Nz+1] * ΔzC₀[k]
end

@kernel function _update_z_star!(ΔzF, ΔzC, ΔzF₀::Number, ΔzC₀::Number, scaling, Nz)
    i, j, k = @index(Global, NTuple)
    @inbounds ΔzF[i, j, k] = scaling[i, j, Nz+1] * ΔzF₀
    @inbounds ΔzC[i, j, k] = scaling[i, j, Nz+1] * ΔzC₀
end

import Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ, Δzᶜᶠᶠ, Δzᶜᶠᶜ, Δzᶠᶜᶠ, Δzᶠᶜᶜ, Δzᶠᶠᶠ, Δzᶠᶠᶜ

# Very bad for GPU performance!!! (z-values are not coalesced in memory for z-derivatives anymore)
# TODO: make z-direction local in memory by not using Fields
@inline Δzᶜᶜᶠ(i, j, k, grid::ZStarCoordinateGrid) = @inbounds grid.Δzᵃᵃᶠ.star_value[i, j, k]
@inline Δzᶜᶜᶜ(i, j, k, grid::ZStarCoordinateGrid) = @inbounds grid.Δzᵃᵃᶜ.star_value[i, j, k]

@inline Δzᶜᶠᶠ(i, j, k, grid::ZStarCoordinateGrid) = ℑyᵃᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.star_value)
@inline Δzᶜᶠᶜ(i, j, k, grid::ZStarCoordinateGrid) = ℑyᵃᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.star_value)

@inline Δzᶠᶜᶠ(i, j, k, grid::ZStarCoordinateGrid) = ℑxᶠᵃᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.star_value)
@inline Δzᶠᶜᶜ(i, j, k, grid::ZStarCoordinateGrid) = ℑxᶠᵃᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.star_value)

@inline Δzᶠᶠᶠ(i, j, k, grid::ZStarCoordinateGrid) = ℑxyᶠᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.star_value)
@inline Δzᶠᶠᶜ(i, j, k, grid::ZStarCoordinateGrid) = ℑxyᶠᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.star_value)