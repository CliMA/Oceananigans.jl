
#####
##### ZStar coordinate and associated grid types
#####

""" free-surface following vertical coordinate """
struct ZStar end

const ZStarSpacing = GeneralizedVerticalSpacing{<:ZStar}

Grids.coordinate_summary(Δ::ZStarSpacing, name) = 
    @sprintf("Free-surface following with Δ%s=%s", name, prettysummary(Δ.Δr))

const ZStarSpacingGrid = GeneralizedSpacingGrid{<:ZStar}

function GeneralizedSpacingGrid(grid::ImmersedBoundaryGrid, ::ZStar)
    underlying_grid  = GeneralizedSpacingGrid(grid.underlying_grid, ZStar())
    active_cells_map = !isnothing(grid.interior_active_cells)

    return ImmersedBoundaryGrid(underlying_grid, grid.immersed_boundary; active_cells_map)
end

# Replacing the z-coordinate with a moving vertical coordinate, defined by its reference spacing,
# the actual vertical spacing and a scaling
function GeneralizedSpacingGrid(grid::AbstractUnderlyingGrid{FT, TX, TY, TZ}, ::ZStar) where {FT, TX, TY, TZ}
    
    # Memory layout for Δz spacings should be local in z instead of x
    ΔzF   =  ZFaceField(grid)
    ΔzC   = CenterField(grid)
    s⁻    = Field{Center, Center, Nothing}(grid)
    sⁿ    = Field{Center, Center, Nothing}(grid)
    ∂t_∂s = Field{Center, Center, Nothing}(grid)
    
    # Initial "at-rest" conditions
    launch!(architecture(grid), grid, :xy, _update_zstar!, 
            sⁿ, s⁻, ΔzF, ΔzC, ZeroField(grid), grid, Val(grid.Nz))

    fill_halo_regions!((s⁻, sⁿ))
    fill_halo_regions!((ΔzF, ΔzC))

    Δzᵃᵃᶠ = GeneralizedVerticalSpacing(ZStar(), grid.Δzᵃᵃᶠ, ΔzF, s⁻, sⁿ, ∂t_∂s)
    Δzᵃᵃᶜ = GeneralizedVerticalSpacing(ZStar(), grid.Δzᵃᵃᶜ, ΔzC, s⁻, sⁿ, ∂t_∂s)

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
##### ZStar-specific vertical spacings update
#####

function update_vertical_spacing!(model, grid::ZStarSpacingGrid, Δt; parameters = :xy)
    
    # Scaling 
    s⁻ = grid.Δzᵃᵃᶠ.s⁻
    sⁿ = grid.Δzᵃᵃᶠ.sⁿ
    ∂t_∂s = grid.Δzᵃᵃᶠ.∂t_∂s

    # Generalized vertical spacing
    ΔzF  = grid.Δzᵃᵃᶠ.Δ
    ΔzC  = grid.Δzᵃᵃᶜ.Δ

    # Update vertical spacing with available parameters 
    # No need to fill the halo as the scaling is updated _IN_ the halos
    launch!(architecture(grid), grid, parameters, _update_zstar!, sⁿ, s⁻, ΔzF, ΔzC, 
                         model.free_surface.η, grid, Val(grid.Nz))
    
    # Update scaling time derivative
    update_∂t_∂s!(∂t_∂s, parameters, grid, sⁿ, s⁻, Δt, model.free_surface)

    return nothing
end

@kernel function _update_zstar!(sⁿ, s⁻, ΔzF, ΔzC, η, grid, ::Val{Nz}) where Nz
    i, j = @index(Global, NTuple)
    bottom = bottom_height(i, j, grid)
    @inbounds begin
        h = (bottom + η[i, j, grid.Nz+1]) / bottom

        # update current and previous scaling
        s⁻[i, j, 1] = sⁿ[i, j, 1]
        sⁿ[i, j, 1] = h
       
        @unroll for k in 1:Nz+1
            ΔzF[i, j, k] = h * Δzᶜᶜᶠ_reference(i, j, k, grid)
            ΔzC[i, j, k] = h * Δzᶜᶜᶜ_reference(i, j, k, grid)
        end
    end
end

update_∂t_∂s!(∂t_∂s, parameters, grid, sⁿ, s⁻, Δt, fs) = 
    launch!(architecture(grid), grid, parameters, _update_∂t_∂s!, ∂t_∂s, sⁿ, s⁻, Δt)

@kernel function _update_∂t_∂s!(∂t_∂s, sⁿ, s⁻, Δt)
    i, j = @index(Global, NTuple)
    ∂t_∂s[i, j, 1] = (sⁿ[i, j, 1] - s⁻[i, j, 1]) /  Δt
end

#####
##### ZStar-specific implementation of the additional terms to be included in the momentum equations
#####

@inline η_surfaceᶜᶜᶜ(i, j, k, grid, η) = @inbounds η[i, j, grid.Nz+1] * (1 + znode(i, j, k, grid, c, c, c) / bottom_height(i, j, grid))

@inline slope_xᶠᶜᶜ(i, j, k, grid, free_surface) = @inbounds ∂xᶠᶜᶜ(i, j, k, grid, η_surfaceᶜᶜᶜ, free_surface.η)
@inline slope_yᶜᶠᶜ(i, j, k, grid, free_surface) = @inbounds ∂yᶜᶠᶜ(i, j, k, grid, η_surfaceᶜᶜᶜ, free_surface.η)

@inline grid_slope_contribution_x(i, j, k, grid::ZStarSpacingGrid, free_surface, ::Nothing, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid::ZStarSpacingGrid, free_surface, ::Nothing, model_fields) = zero(grid)

@inline grid_slope_contribution_x(i, j, k, grid::ZStarSpacingGrid, free_surface, buoyancy, model_fields) = 
    ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * slope_xᶠᶜᶜ(i, j, k, grid, free_surface)

@inline grid_slope_contribution_y(i, j, k, grid::ZStarSpacingGrid, free_surface, buoyancy, model_fields) = 
    ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * slope_yᶜᶠᶜ(i, j, k, grid, free_surface)
