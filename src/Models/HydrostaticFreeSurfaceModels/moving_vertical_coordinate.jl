using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BuoyancyModels: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, halo_size
using Oceananigans.ImmersedBoundaries
using Oceananigans.Utils: getnamewrapper
using Adapt 
using Printf

""" geopotential-following vertical coordinate """
struct ZCoordinate end

"""
    ZStarCoordinate{R, S, Z} 

a _free surface following_ vertical coordinate system.
The fixed coordinate is stored in `reference`, while `star_value`
contains the actual free-surface-following z-coordinate
the `scaling` and `∂t_scaling` fields are the vertical derivative of
the vertical coordinate and it's time derivative
"""
struct ZStarCoordinate{R, S, Z}
    Δ★ :: R # Reference _non moving_ coordinate
     Δ :: Z # moving vertical coordinate
    s⁻ :: S # scaling term = ∂Δ★(Δ) at the start of the time step
    sⁿ :: S # scaling term = ∂Δ★(Δ) at the end of the time step
  ∂t_s :: S # Time derivative of the vertical coordinate scaling
end

ZStarCoordinate() = ZStarCoordinate(nothing, nothing, nothing, nothing, nothing)

import Oceananigans.Grids: coordinate_summary

coordinate_summary(Δ::ZStarCoordinate, name) = 
    @sprintf("Free-surface following with Δ%s=%s", name, prettysummary(Δ.Δz★))

Adapt.adapt_structure(to, coord::ZStarCoordinate) = 
    ZStarCoordinate(Adapt.adapt(to, coord.Δ★),
                    Adapt.adapt(to, coord.Δ),
                    Adapt.adapt(to, coord.s⁻),
                    Adapt.adapt(to, coord.sⁿ),
                    Adapt.adapt(to, coord.∂t_s))

import Oceananigans.Architectures: arch_array

arch_array(arch, coord::ZStarCoordinate) = 
    ZStarCoordinate(arch_array(arch, coord.Δ★), 
                    coord.Δ,
                    coord.s⁻,
                    coord.sⁿ,
                    coord.∂t_s)
                    

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

# Replacing the z-coordinate with a moving vertical coordinate, defined by its reference spacing,
# the actual vertical spacing and a scaling
function MovingCoordinateGrid(grid::AbstractUnderlyingGrid{FT, TX, TY, TZ}, ::ZStarCoordinate) where {FT, TX, TY, TZ}
    
    # Memory layout for Dz spacings is local in z
    ΔzF  =  ZFaceField(grid)
    ΔzC  = CenterField(grid)
    s⁻   =  ZFaceField(grid, indices = (:, :, grid.Nz+1))
    sⁿ   =  ZFaceField(grid, indices = (:, :, grid.Nz+1))
    ∂t_s =  ZFaceField(grid, indices = (:, :, grid.Nz+1))

    # Initial "at-rest" conditions
    launch!(architecture(grid), grid, :xy, _update_scaling!,
            sⁿ, s⁻, ∂t_s, ZeroField(grid), grid, 1)
    
    launch!(architecture(grid), grid, :xy,_update_z_star!, 
        ΔzF, ΔzC, grid.Δzᵃᵃᶠ, grid.Δzᵃᵃᶜ, sⁿ, Val(grid.Nz))

    fill_halo_regions!((ΔzF, ΔzC, s⁻, sⁿ, ∂t_s); only_local_halos = true)

    Δzᵃᵃᶠ = ZStarCoordinate(grid.Δzᵃᵃᶠ, ΔzF, s⁻, sⁿ, ∂t_s)
    Δzᵃᵃᶜ = ZStarCoordinate(grid.Δzᵃᵃᶜ, ΔzC, s⁻, sⁿ, ∂t_s)

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
update_vertical_coordinate!(model, grid, Δt; kwargs...) = nothing

function update_vertical_coordinate!(model, grid::ZStarCoordinateGrid, Δt; parameters = tuple(:xyz))
    η = model.free_surface.η
    
    # Scaling 
    s⁻ = grid.Δzᵃᵃᶠ.s⁻
    sⁿ = grid.Δzᵃᵃᶠ.sⁿ
    ∂t_s = grid.Δzᵃᵃᶠ.∂t_s

    # Moving coordinates
    Δzᵃᵃᶠ  = grid.Δzᵃᵃᶠ.Δ
    Δzᵃᵃᶜ  = grid.Δzᵃᵃᶜ.Δ

    # Reference coordinates
    Δz₀ᵃᵃᶠ = grid.Δzᵃᵃᶠ.Δ★
    Δz₀ᵃᵃᶜ = grid.Δzᵃᵃᶜ.Δ★

    # Update vertical coordinate with available parameters 
    for params in parameters
        update_coordinate_scaling!(sⁿ, s⁻, ∂t_s, params, model.free_surface, grid, Δt)

        launch!(architecture(grid), grid, horizontal_parameters(params), _update_z_star!, 
                Δzᵃᵃᶠ, Δzᵃᵃᶜ, Δz₀ᵃᵃᶠ, Δz₀ᵃᵃᶜ, sⁿ, Val(grid.Nz))
    end

    fill_halo_regions!((Δzᵃᵃᶠ, Δzᵃᵃᶜ, s⁻, sⁿ, ∂t_s); only_local_halos = true)
    
    return nothing
end

horizontal_parameters(::Symbol) = :xy
horizontal_parameters(::KernelParameters{W, O}) where {W, O} = KernelParameters(W[1:2], O[1:2])

update_coordinate_scaling!(sⁿ, s⁻, ∂t_s, params, fs, grid, Δt) = 
    launch!(architecture(grid), grid, horizontal_parameters(params), _update_scaling!,
            sⁿ, s⁻, ∂t_s, fs.η, grid, Δt)

@kernel function _update_scaling!(sⁿ, s⁻, ∂t_s, η, grid, Δt)
    i, j = @index(Global, NTuple)
    bottom = bottom_height(i, j, grid)
    @inbounds begin
        h = (bottom + η[i, j, grid.Nz+1]) / bottom

        # update current and previous scaling
        s⁻[i, j, grid.Nz+1] = sⁿ[i, j, grid.Nz+1]
        sⁿ[i, j, grid.Nz+1] = h

        # Scaling derivative
        ∂t_s[i, j, grid.Nz+1] = (h - s⁻[i, j, grid.Nz+1]) / Δt 
    end
end

bottom_height(i, j, grid) = grid.Lz
bottom_height(i, j, grid::ImmersedBoundaryGrid) = @inbounds - grid.immersed_boundary.bottom_height[i, j, 1]

@kernel function _update_z_star!(ΔzF, ΔzC, ΔzF₀, ΔzC₀, sⁿ, ::Val{Nz}) where Nz
    i, j = @index(Global, NTuple)
    @unroll for k in 1:Nz+1
        @inbounds ΔzF[i, j, k] = sⁿ[i, j, Nz+1] * ΔzF₀[k]
        @inbounds ΔzC[i, j, k] = sⁿ[i, j, Nz+1] * ΔzC₀[k]
    end
end

@kernel function _update_z_star!(ΔzF, ΔzC, ΔzF₀::Number, ΔzC₀::Number, sⁿ, ::Val{Nz}) where Nz
    i, j = @index(Global, NTuple)
    @unroll for k in 1:Nz+1
        @inbounds ΔzF[i, j, k] = sⁿ[i, j, Nz+1] * ΔzF₀
        @inbounds ΔzC[i, j, k] = sⁿ[i, j, Nz+1] * ΔzC₀
    end
end

@inline scaling(i, j, k, grid) = one(grid)
@inline scaling(i, j, k, grid::ZStarCoordinateGrid) = grid.Δzᵃᵃᶠ.sⁿ[i, j, grid.Nz+1]


import Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ, Δzᶜᶠᶠ, Δzᶜᶠᶜ, Δzᶠᶜᶠ, Δzᶠᶜᶜ, Δzᶠᶠᶠ, Δzᶠᶠᶜ
import Oceananigans.Operators: Vᶜᶜᶠ, Vᶜᶜᶜ, Vᶜᶠᶠ, Vᶜᶠᶜ, Vᶠᶜᶠ, Vᶠᶜᶜ, Vᶠᶠᶠ, Vᶠᶠᶜ

# Very bad for GPU performance!!! (z-values are not coalesced in memory for z-derivatives anymore)
# TODO: make z-direction local in memory by not using Fields
@inline Δzᶜᶜᶠ(i, j, k, grid::ZStarCoordinateGrid) = @inbounds grid.Δzᵃᵃᶠ.Δ[i, j, k]
@inline Δzᶜᶜᶜ(i, j, k, grid::ZStarCoordinateGrid) = @inbounds grid.Δzᵃᵃᶜ.Δ[i, j, k]

@inline Δzᶜᶠᶠ(i, j, k, grid::ZStarCoordinateGrid) = ℑyᵃᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.Δ)
@inline Δzᶜᶠᶜ(i, j, k, grid::ZStarCoordinateGrid) = ℑyᵃᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.Δ)

@inline Δzᶠᶜᶠ(i, j, k, grid::ZStarCoordinateGrid) = ℑxᶠᵃᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.Δ)
@inline Δzᶠᶜᶜ(i, j, k, grid::ZStarCoordinateGrid) = ℑxᶠᵃᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.Δ)

@inline Δzᶠᶠᶠ(i, j, k, grid::ZStarCoordinateGrid) = ℑxyᶠᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶠ.Δ)
@inline Δzᶠᶠᶜ(i, j, k, grid::ZStarCoordinateGrid) = ℑxyᶠᶠᵃ(i, j, k, grid, grid.Δzᵃᵃᶜ.Δ)

# Adding the slope to the momentum-RHS
@inline free_surface_slope_x(i, j, k, grid, args...) = zero(grid)
@inline free_surface_slope_y(i, j, k, grid, args...) = zero(grid)

@inline free_surface_slope_x(i, j, k, grid::ZStarCoordinateGrid, free_surface, ::Nothing, model_fields) = zero(grid)
@inline free_surface_slope_y(i, j, k, grid::ZStarCoordinateGrid, free_surface, ::Nothing, model_fields) = zero(grid)

@inline η_times_zᶜᶜᶜ(i, j, k, grid, η) = @inbounds η[i, j, grid.Nz+1] * (1 + grid.zᵃᵃᶜ[k] / bottom_height(i, j, grid))

@inline η_slope_xᶠᶜᶜ(i, j, k, grid, free_surface) = @inbounds ∂xᶠᶜᶜ(i, j, k, grid, η_times_zᶜᶜᶜ, free_surface.η)
@inline η_slope_yᶜᶠᶜ(i, j, k, grid, free_surface) = @inbounds ∂yᶜᶠᶜ(i, j, k, grid, η_times_zᶜᶜᶜ, free_surface.η)

@inline free_surface_slope_x(i, j, k, grid::ZStarCoordinateGrid, free_surface, buoyancy, model_fields) = 
    ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * η_slope_xᶠᶜᶜ(i, j, k, grid, free_surface)

@inline free_surface_slope_y(i, j, k, grid::ZStarCoordinateGrid, free_surface, buoyancy, model_fields) = 
    ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * η_slope_yᶜᶠᶜ(i, j, k, grid, free_surface)

#####
##### In the Z-star coordinate framework the prognostic field is sθ!  
#####

@kernel function ab2_step_tracer_zstar!(θ, sⁿ, s⁻, Nz, Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    FT = eltype(χ)
    one_point_five = convert(FT, 1.5)
    oh_point_five  = convert(FT, 0.5)

    @inbounds begin
        ∂t_θ = (one_point_five + χ) * Gⁿ[i, j, k] - (oh_point_five + χ) * G⁻[i, j, k]
        sθ   = sⁿ[i, j, Nz+1] * θ[i, j, k] + convert(FT, Δt) * ∂t_θ
        θ[i, j, k] = sθ / sⁿ[i, j, Nz+1]
    end
end

ab2_step_tracer_field!(tracer_field, grid::ZStarCoordinateGrid, Δt, χ, Gⁿ, G⁻) =
    launch!(architecture(grid), grid, :xyz, ab2_step_tracer_zstar!, 
            tracer_field, 
            grid.Δzᵃᵃᶠ.sⁿ, 
            grid.Δzᵃᵃᶠ.s⁻, 
            grid.Nz,
            Δt, χ, Gⁿ, G⁻)