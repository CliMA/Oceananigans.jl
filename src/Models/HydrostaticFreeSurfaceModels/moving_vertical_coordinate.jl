using Oceananigans.Grids: AbstractGrid

const RigidLidModel = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Nothing}

struct ZStarCoordinate{R, S, Z}
    reference :: R
      scaling :: S
   star_value :: Z
end

Adapt.adapt_structure(to, coord::ZStarCoordinate) = 
    ZStarCoordinate(Adapt.adapt(to, coord.reference),
                    Adapt.adapt(to, coord.scaling),
                    Adapt.adapt(to, coord.star_value))

const ZStarCoordinateRG  = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ZStarCoordinate}
const ZStarCoordinateLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ZStarCoordinate}

const ZStarCoordinateGrid = Union{ZStarCoordinateRG, ZStarCoordinateLLG}

function MovingCoordinateGrid(grid::AbstractGrid{FT, TX, TY, TZ}, ::ZStarCoordinate) where {FT, TX, TY, TZ}
    ΔzF =  ZFaceField(grid)
    ΔzC = CenterField(grid)
    scaling = ZFaceField(grid, indices = (:, :, grid.Nz + 1))

    Δzᵃᵃᶠ = ZStarCoordinate(grid.Δzᵃᵃᶠ, ΔzF, scaling)
    Δzᵃᵃᶜ = ZStarCoordinate(grid.Δzᵃᵃᶜ, ΔzC, scaling)

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
update_vertical_coordinate!(model, grid) = nothing

# Do not update if rigid lid!!
update_vertical_coordinate!(::RigidLidModel, ::ZStarCoordinateGrid) = nothing

function update_vertical_coordinate!(model, grid::ZStarCoordinateGrid)
    η = model.free_surface.η
    
    # Scaling 
    scaling = grid.Δzᵃᵃᶠ.scaling

    # Moving coordinates
    Δzᵃᵃᶠ  = grid.Δzᵃᵃᶠ.star_value
    Δzᵃᵃᶜ  = grid.Δzᵃᵃᶜ.star_value

    # Reference coordinates
    Δz₀ᵃᵃᶠ = grid.Δzᵃᵃᶠ.reference
    Δz₀ᵃᵃᶜ = grid.Δzᵃᵃᶜ.reference

    launch!(architecture(grid), grid, _update_scaling!, :xy, 
            scaling, η, grid.Lz, grid.Nz)

    # Update vertical coordinate
    launch!(architecture(grid), grid, _update_z_star!, :xyz, 
            Δzᵃᵃᶠ, Δzᵃᵃᶜ, Δz₀ᵃᵃᶠ, Δz₀ᵃᵃᶜ, scaling)

    return nothing
end

@kernel function update_scaling!(scaling, η, grid.Lz, grid.Nz)
    i, j = @index(Global, NTuple)
    @inbounds scaling[i, j, Nz+1] = (Lz + η[i, j, Nz+1]) / Lz
end

@kernel function _update_z_star!(ΔzF, ΔzC, ΔzF₀, ΔzC₀, scaling)
    i, j, k = @index(Global, NTuple)
    @inbounds ΔzF[i, j, k] = scaling[i, j, Nz+1] * ΔzF₀[k]
    @inbounds ΔzC[i, j, k] = scaling[i, j, Nz+1] * ΔzC₀[k]
end

@kernel function _update_z_star!(ΔzF, ΔzC, ΔzF₀::Number, ΔzC₀::Number, scaling)
    i, j, k = @index(Global, NTuple)
    @inbounds ΔzF[i, j, k] = scaling[i, j, Nz+1] * ΔzF₀
    @inbounds ΔzC[i, j, k] = scaling[i, j, Nz+1] * ΔzC₀
end

import Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ, Δzᶜᶠᶠ, Δzᶜᶠᶜ, Δzᶠᶜᶠ, Δzᶠᶜᶜ, Δzᶠᶠᶠ, Δzᶠᶠᶜ

@inline Δzᶜᶜᶠ(i, j, k, grid::ZStarCoordinateGrid) = @inbounds grid.Δzᵃᵃᶠ.star_value[i, j, k]
@inline Δzᶜᶜᶜ(i, j, k, grid::ZStarCoordinateGrid) = @inbounds grid.Δzᵃᵃᶜ.star_value[i, j, k]

@inline Δzᶜᶠᶠ(i, j, k, grid::ZStarCoordinateGrid) = ℑyᵃᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.star_value)
@inline Δzᶜᶠᶜ(i, j, k, grid::ZStarCoordinateGrid) = ℑyᵃᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.star_value)

@inline Δzᶠᶜᶠ(i, j, k, grid::ZStarCoordinateGrid) = ℑxᶠᵃᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.star_value)
@inline Δzᶠᶜᶜ(i, j, k, grid::ZStarCoordinateGrid) = ℑxᶠᵃᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.star_value)

@inline Δzᶠᶠᶠ(i, j, k, grid::ZStarCoordinateGrid) = ℑxyᶠᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.star_value)
@inline Δzᶠᶠᶜ(i, j, k, grid::ZStarCoordinateGrid) = ℑxyᶠᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.star_value)