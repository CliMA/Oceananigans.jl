using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BuoyancyModels: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, halo_size
using Oceananigans.ImmersedBoundaries
using Oceananigans.Utils: getnamewrapper
using Adapt 

struct ZCoordinate end

struct ZStarCoordinate{R, S, Z}
  reference :: R 
 star_value :: Z
    scaling :: S
 ∂t_scaling :: S
end

ZStarCoordinate() = ZStarCoordinate(nothing, nothing, nothing, nothing)

Adapt.adapt_structure(to, coord::ZStarCoordinate) = 
    ZStarCoordinate(Adapt.adapt(to, coord.reference),
                    Adapt.adapt(to, coord.scaling),
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
    active_cells_map = !isnothing(grid.interior_active_cells)

    return ImmersedBoundaryGrid(underlying_grid, grid.immersed_boundary; active_cells_map)
end

function MovingCoordinateGrid(grid::AbstractUnderlyingGrid{FT, TX, TY, TZ}, ::ZStarCoordinate) where {FT, TX, TY, TZ}
    # Memory layout for Dz spacings is local in z
    ΔzF =  ZFaceField(grid)
    ΔzC = CenterField(grid)
    scaling = ZFaceField(grid, indices = (:, :, grid.Nz + 1))
    ∂t_scaling = ZFaceField(grid, indices = (:, :, grid.Nz + 1))

    Δzᵃᵃᶠ = ZStarCoordinate(grid.Δzᵃᵃᶠ, ΔzF, scaling, ∂t_scaling)
    Δzᵃᵃᶜ = ZStarCoordinate(grid.Δzᵃᵃᶜ, ΔzC, scaling, ∂t_scaling)

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

function update_vertical_coordinate!(model, grid::ZStarCoordinateGrid, Δt; parameters = tuple(:xyz))
    η = model.free_surface.η
    
    # Scaling 
    scaling = grid.Δzᵃᵃᶠ.scaling
    ∂t_scaling = grid.Δzᵃᵃᶠ.previous_scaling

    # Moving coordinates
    Δzᵃᵃᶠ  = grid.Δzᵃᵃᶠ.star_value
    Δzᵃᵃᶜ  = grid.Δzᵃᵃᶜ.star_value

    # Reference coordinates
    Δz₀ᵃᵃᶠ = grid.Δzᵃᵃᶠ.reference
    Δz₀ᵃᵃᶜ = grid.Δzᵃᵃᶜ.reference

    # Update the scaling on the whole grid (from -H to N+H)
    launch!(architecture(grid), grid, horizontal_parameters(grid), _update_scaling!,
            scaling, ∂t_scaling, η, grid, Δt)

    # Update vertical coordinate with available parameters 
    for params in parameters
        launch!(architecture(grid), grid, params, _update_z_star!, 
                Δzᵃᵃᶠ, Δzᵃᵃᶜ, Δz₀ᵃᵃᶠ, Δz₀ᵃᵃᶜ, scaling, grid.Nz)
    end

    fill_halo_regions!((Δzᵃᵃᶠ, Δzᵃᵃᶜ); only_local_halos = true)
    
    return nothing
end

function horizontal_parameters(grid)
    halos = halo_size(grid)[1:2]
    total_size = size(grid)[1:2] .+ 2 .* halos
    return KernelParameters(total_size, .- halos)
end

@kernel function _update_scaling!(scaling, ∂t_scaling, η, grid, Δt)
    i, j = @index(Global, NTuple)
    bottom = bottom_height(i, j, grid)
    @inbounds begin
        h = (bottom + η[i, j, grid.Nz+1]) / bottom
        ∂t_scaling[i, j, grid.Nz+1] = (h - scaling[i, j, grid.Nz+1]) / Δt 
        scaling[i, j, grid.Nz+1] = h
    end
end

bottom_height(i, j, grid) = grid.Lz
bottom_height(i, j, grid::ImmersedBoundaryGrid) = @inbounds - grid.immersed_boundary.bottom_height[i, j, 1]

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

import Oceananigans.Architectures: arch_array

arch_array(arch, coord::ZStarCoordinate) = 
    ZStarCoordinate(arch_array(arch, coord.reference), coord.scaling, coord.star_value)

# Adding the slope to the momentum-RHS
@inline free_surface_slope_x(i, j, k, grid, args...) = nothing
@inline free_surface_slope_y(i, j, k, grid, args...) = nothing

@inline η_times_zᶜᶜᶜ(i, j, k, grid, η) = @inbounds η[i, j, grid.Nz+1] * (1 + grid.zᵃᵃᶜ[k] / bottom(i, j, grid))

@inline η_slope_xᶠᶜᶜ(i, j, k, grid, free_surface) = @inbounds ∂xᶠᶜᶜ(i, j, k, grid, η_times_zᶜᶜᶜ, free_surface.η)
@inline η_slope_yᶜᶠᶜ(i, j, k, grid, free_surface) = @inbounds ∂yᶜᶠᶜ(i, j, k, grid, η_times_zᶜᶜᶜ, free_surface.η)

@inline free_surface_slope_x(i, j, k, grid::ZStarCoordinateGrid, free_surface, buoyancy, model_fields) = 
    ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy, model_fields) * η_slope_xᶠᶜᶜ(i, j, k, grid, free_surface)

@inline free_surface_slope_y(i, j, k, grid::ZStarCoordinateGrid, free_surface, buoyancy, model_fields) = 
    ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy, model_fields) * η_slope_yᶜᶠᶜ(i, j, k, grid, free_surface)
