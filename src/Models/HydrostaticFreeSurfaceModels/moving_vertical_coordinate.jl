using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BuoyancyModels: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, halo_size
using Oceananigans.ImmersedBoundaries
using Oceananigans.Utils: getnamewrapper
using Adapt 
using Printf

"""
    GeneralizedVerticalCoordinate{R, S, Z} 

spacings for a generalized vertical coordinate system.
The reference coordinate is stored in `Δr`, while `Δ` contains the z-coordinate.
The `s⁻`, `sⁿ` and `∂t_s` fields are the vertical derivative of the vertical coordinate (∂Δ/∂Δr)
at timestep `n-1` and `n` and it's time derivative.
`denomination` contains the "type" of generalized vertical coordinate, for example:
- Zstar: free-surface following
- sigma: terrain following
"""
struct GeneralizedVerticalCoordinate{D, R, Z, S}
  denomination :: D # The type of generalized coordinate
            Δr :: R # Reference _non moving_ vertical coordinate (one-dimensional)
             Δ :: Z # moving vertical coordinate (three-dimensional)
            s⁻ :: S # scaling term = ∂Δ/∂Δr at the start of the time step
            sⁿ :: S # scaling term = ∂Δ/∂Δr at the end of the time step
          ∂t_s :: S # Time derivative of the vertical coordinate scaling divided by the sⁿ
end

Adapt.adapt_structure(to, coord::GeneralizedVerticalCoordinate) = 
    GeneralizedVerticalCoordinate(nothing, 
                             Adapt.adapt(to, coord.Δr),
                             Adapt.adapt(to, coord.Δ),
                             Adapt.adapt(to, coord.s⁻),
                             Adapt.adapt(to, coord.sⁿ),
                             Adapt.adapt(to, coord.∂t_s))

import Oceananigans.Architectures: arch_array

arch_array(arch, coord::GeneralizedVerticalCoordinate) = 
    GeneralizedVerticalCoordinate(coord.denomination,
                             arch_array(arch, coord.Δr), 
                             coord.Δ,
                             coord.s⁻,
                             coord.sⁿ,
                             coord.∂t_s)

const MovingCoordinateRG{D}  = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:GeneralizedVerticalCoordinate{D}} where D
const MovingCoordinateLLG{D} = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:GeneralizedVerticalCoordinate{D}} where D

const MovingCoordinateUnderlyingGrid{D} = Union{MovingCoordinateRG{D}, MovingCoordinateLLG{D}} where D
const MovingCoordinateImmersedGrid{D} = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:MovingCoordinateUnderlyingGrid{D}} where D

const MovingCoordinateGrid{D} = Union{MovingCoordinateUnderlyingGrid{D}, MovingCoordinateImmersedGrid{D}} where D

""" free-surface following vertical coordinate """
struct ZStar end

""" geopotential following vertical coordinate """
struct Z end

const ZStarCoordinate = GeneralizedVerticalCoordinate{<:ZStar}

Grids.coordinate_summary(Δ::ZStarCoordinate, name) = 
    @sprintf("Free-surface following with Δ%s=%s", name, prettysummary(Δ.Δr))

const ZStarCoordinateGrid = MovingCoordinateGrid{<:ZStar}

MovingCoordinateGrid(grid, coord) = grid

function MovingCoordinateGrid(grid::ImmersedBoundaryGrid, ::ZStar)
    underlying_grid = MovingCoordinateGrid(grid.underlying_grid, ZStarCoordinate())
    active_cells_map = !isnothing(grid.interior_active_cells)

    return ImmersedBoundaryGrid(underlying_grid, grid.immersed_boundary; active_cells_map)
end

# Replacing the z-coordinate with a moving vertical coordinate, defined by its reference spacing,
# the actual vertical spacing and a scaling
function MovingCoordinateGrid(grid::AbstractUnderlyingGrid{FT, TX, TY, TZ}, ::ZStar) where {FT, TX, TY, TZ}
    
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

    Δzᵃᵃᶠ = GeneralizedVerticalCoordinate(ZStar(), grid.Δzᵃᵃᶠ, ΔzF, s⁻, sⁿ, ∂t_s)
    Δzᵃᵃᶜ = GeneralizedVerticalCoordinate(ZStar(), grid.Δzᵃᵃᶜ, ΔzC, s⁻, sⁿ, ∂t_s)

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
    ΔzF  = grid.Δzᵃᵃᶠ.Δ
    ΔzC  = grid.Δzᵃᵃᶜ.Δ

    # Reference coordinates
    Δz₀F = grid.Δzᵃᵃᶠ.Δr
    Δz₀C = grid.Δzᵃᵃᶜ.Δr

    # Update vertical coordinate with available parameters 
    for params in parameters
        update_coordinate_scaling!(sⁿ, s⁻, ∂t_s, params, model.free_surface, grid, Δt)

        launch!(architecture(grid), grid, horizontal_parameters(params), _update_z_star!, 
                ΔzF, ΔzC, Δz₀F, Δz₀C, sⁿ, Val(grid.Nz))
    end

    fill_halo_regions!((ΔzF, ΔzC, s⁻, sⁿ, ∂t_s); only_local_halos = true)
    
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
        ∂t_s[i, j, grid.Nz+1] = (h - s⁻[i, j, grid.Nz+1]) / Δt  / h
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
@inline horizontal_surface_slope_x(i, j, k, grid, args...) = zero(grid)
@inline horizontal_surface_slope_y(i, j, k, grid, args...) = zero(grid)

@inline horizontal_surface_slope_x(i, j, k, grid::ZStarCoordinateGrid, free_surface, ::Nothing, model_fields) = zero(grid)
@inline horizontal_surface_slope_y(i, j, k, grid::ZStarCoordinateGrid, free_surface, ::Nothing, model_fields) = zero(grid)

@inline η_times_zᶜᶜᶜ(i, j, k, grid, η) = @inbounds η[i, j, grid.Nz+1] * (1 + grid.zᵃᵃᶜ[k] / bottom_height(i, j, grid))

@inline η_slope_xᶠᶜᶜ(i, j, k, grid, free_surface) = @inbounds ∂xᶠᶜᶜ(i, j, k, grid, η_times_zᶜᶜᶜ, free_surface.η)
@inline η_slope_yᶜᶠᶜ(i, j, k, grid, free_surface) = @inbounds ∂yᶜᶠᶜ(i, j, k, grid, η_times_zᶜᶜᶜ, free_surface.η)

@inline horizontal_surface_slope_x(i, j, k, grid::ZStarCoordinateGrid, free_surface, buoyancy, model_fields) = 
    ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * η_slope_xᶠᶜᶜ(i, j, k, grid, free_surface)

@inline horizontal_surface_slope_y(i, j, k, grid::ZStarCoordinateGrid, free_surface, buoyancy, model_fields) = 
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
        θ[i, j, k] = s⁻[i, j, Nz+1] * θ[i, j, k] / sⁿ[i, j, Nz+1] + convert(FT, Δt) * ∂t_θ
    end
end

ab2_step_tracer_field!(tracer_field, grid::ZStarCoordinateGrid, Δt, χ, Gⁿ, G⁻) =
    launch!(architecture(grid), grid, :xyz, ab2_step_tracer_zstar!, 
            tracer_field, 
            grid.Δzᵃᵃᶠ.sⁿ, 
            grid.Δzᵃᵃᶠ.s⁻, 
            grid.Nz,
            Δt, χ, Gⁿ, G⁻)

# When performing divergence upwinding we must include the 
# metric term

import Oceananigans.Advection: metric_term

metric_term(i, j, k, grid::MovingCoordinateGrid) = grid.Δzᵃᵃᶜ.∂t_s[i, j, grid.Nz+1] * Vᶜᶜᶜ(i, j, k, grid)
