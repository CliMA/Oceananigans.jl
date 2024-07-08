using Oceananigans.Grids: xspacings, yspacings, zspacings

"""
    FlatExtrapolation

Zero gradient perpendicular velocity boundary condition.

For constant spacing:
```math
f′(xᵢ) ≈ f′(xᵢ₋₁) + f′′(xᵢ₋₁)(xᵢ₋₁ - xᵢ) + O(Δx²) = f′(xᵢ₋₁) + f′′(xᵢ₋₁)Δx + O(Δx²),

f′(xᵢ₋₁) ≈ (f(xᵢ) - f(xᵢ₋₂)) / 2Δx

f′(xᵢ) ≈ (f(xᵢ) - f(xᵢ₋₂)) / 2Δx + O(Δx) = 0 ∴ f(xᵢ) ≈ f(xᵢ₋₂) + O(Δx)

f′′(xᵢ₋₁) ≈ (f′(xᵢ) - f′(xᵢ₋₂)) / 2Δx = - f′(xᵢ₋₂) / 2Δx ≈ - (f(xᵢ₋₁) - f(xᵢ₋₃)) / (2Δx)²

∴ f(xᵢ) ≈ f(xᵢ₋₂) - (f(xᵢ₋₁) - f(xᵢ₋₃))/2 + O(Δx²)
```

When the grid spacing is not constant it works out that the factor of 1/2 changes to 
``Δx₋₁/(Δx₋₂ + Δx₋₃)`` instead.
"""
struct FlatExtrapolation{FT}
    relaxation_timescale :: FT
end

const FEOBC = BoundaryCondition{<:Open{<:FlatExtrapolation}}

function FlatExtrapolationOpenBoundaryCondition(val = nothing; relaxation_timescale = Inf, kwargs...)
    classification = Open(FlatExtrapolation(relaxation_timescale))
    
    return BoundaryCondition(classification, val; kwargs...)
end

@inline relax(l, m, c, bc, grid, clock, model_fields) =
    c + ifelse(isnothing(bc.condition)||!isfinite(clock.last_stage_Δt), 0,
        (getbc(bc, l, m, grid, clock, model_fields) - c) * min(1, clock.last_stage_Δt / bc.classification.matching_scheme.relaxation_timescale))

@inline spacing_factor(args...) = 1/2
@inline spacing_factor(Δ::AbstractArray, ::Val{:right}) = @inbounds Δ[end] / (Δ[end-1] + Δ[end-2])
@inline spacing_factor(Δ::AbstractArray, ::Val{:left})  = @inbounds Δ[1]   / (Δ[2]     + Δ[3]    )

@inline function _fill_west_open_halo!(j, k, grid, c, bc::FEOBC, loc, clock, model_fields)
    Δx = xspacings(grid, Center(), Center(), Center())

    unrelaxed = @inbounds c[3, j, k] - (c[2, j, k] - c[4, j, k]) * spacing_factor(Δx, Val(:left))

    @inbounds c[1, j, k] = relax(j, k, unrelaxed, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_east_open_halo!(j, k, grid, c, bc::FEOBC, loc, clock, model_fields)
    i = grid.Nx + 1

    Δx = xspacings(grid, Center(), Center(), Center())

    unrelaxed = @inbounds c[i - 2, j, k] - (c[i - 1, j, k] - c[i - 3, j, k]) * spacing_factor(Δx, Val(:right))

    @inbounds c[i, j, k] = relax(j, k, unrelaxed, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_south_open_halo!(i, k, grid, c, bc::FEOBC, loc, clock, model_fields)
    Δy = yspacings(grid, Center(), Center(), Center())

    unrelaxed = c[i, 3, k] - (c[i, 2, k] - c[i, 4, k]) * spacing_factor(Δy, Val(:left))

    @inbounds c[i, 1, k] = relax(i, k, unrelaxed, bc, grid, clock, model_fields)
    
    return nothing
end

@inline function _fill_north_open_halo!(i, k, grid, c, bc::FEOBC, loc, clock, model_fields)
    j = grid.Ny + 1

    Δy = yspacings(grid, Center(), Center(), Center())

    unrelaxed = @inbounds c[i, j - 2, k] - (c[i, j - 1, k] - c[i, j - 3, k]) * spacing_factor(Δy, Val(:right))

    @inbounds c[i, j, k] = relax(i, k, unrelaxed, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_bottom_open_halo!(i, j, grid, c, bc::FEOBC, loc, clock, model_fields)
    Δz = zspacings(grid, Center(), Center(), Center())

    unrelaxed = @inbounds c[i, j, 3] - (c[i, k, 2] - c[i, j, 4]) * spacing_factor(Δz, Val(:left))

    @inbounds c[i, j, 1] = relax(i, j, unrelaxed, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_top_open_halo!(i, j, grid, c, bc::FEOBC, loc, clock, model_fields)
    k = grid.Nz + 1

    Δz = zspacings(grid, Center(), Center(), Center())

    unrelaxed = @inbounds c[i, j, k - 2] - (c[i, j, k - 1] - c[i, j, k - 3]) * spacing_factor(Δz, Val(:right))

    @inbounds c[i, j, k] = relax(i, j, unrelaxed, bc, grid, clock, model_fields)

    return nothing
end