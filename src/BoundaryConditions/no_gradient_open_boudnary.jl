using Oceananigans.Operators: ∂xᶜᶜᶜ

"""
    FlatExtrapolation

Zero gradient perepndicular velocity boundary condition.

*Given constant spacing*
```math
f′(xᵢ) ≈ f′(xᵢ₋₁) + f′′(xᵢ₋₁)(xᵢ₋₁ - xᵢ) + O(Δx²) = f′(xᵢ₋₁) + f′′(xᵢ₋₁)Δx + O(Δx²),

f′(xᵢ₋₁) ≈ (f(xᵢ) - f(xᵢ₋₂)) / 2Δx

f′(xᵢ) ≈ (f(xᵢ) - f(xᵢ₋₂)) / 2Δx + O(Δx) = 0 ∴ f(xᵢ) ≈ f(xᵢ₋₂) + O(Δx)

f′′(xᵢ₋₁) ≈ (f′(xᵢ) - f′(xᵢ₋₂)) / 2Δx = - f′(xᵢ₋₂) / 2Δx ≈ - (f(xᵢ₋₁) - f(xᵢ₋₃)) / (2Δx)²

∴ f(xᵢ) ≈ f(xᵢ₋₂) + (f(xᵢ₋₁) - f(xᵢ₋₃))/2 + O(Δx²)
```
"""
struct FlatExtrapolation end

const FEOBC = BoundaryCondition{<:Open{<:FlatExtrapolation}}

function FlatExtrapolationOpenBoundaryCondition()
    classifcation = Open(FlatExtrapolation())
    
    return BoundaryCondition(classifcation, nothing)
end

@inline _fill_west_open_halo!(j, k, grid, c, bc::FEOBC, loc, clock, model_fields) = @inbounds c[0, j, k] = c[2, j, k] + (c[1, j, k] - c[3, j, k]) / 2

@inline function _fill_east_open_halo!(j, k, grid, c, bc::FEOBC, loc, clock, model_fields)
    i = grid.Nx + 1

    @inbounds c[i, j, k] =  c[i - 2, j, k] + (c[i - 1, j, k] - c[i - 3, j, k]) / 2

    return nothing
end

@inline _fill_south_open_halo!(i, k, grid, c, bc::FEOBC, loc, clock, model_fields) = @inbounds c[i, 0, k] = c[i, 2, k] + (c[i, 1, k] - c[i, 3, k]) / 2

@inline function _fill_north_open_halo!(i, k, grid, c, bc::FEOBC, loc, clock, model_fields)
    j = grid.Ny + 1

    @inbounds c[i, j, k] = c[i, j - 2, k] + (c[i, j - 1, k] - c[i, j - 3, k]) / 2

    return nothing
end

@inline _fill_bottom_open_halo!(i, j, grid, c, bc::FEOBC, loc, clock, model_fields) = @inbounds c[i, j, 0] = c[i, j, 2] + (c[i, k, 1] - c[i, j, 3]) / 2

@inline function _fill_top_open_halo!(i, j, grid, c, bc::FEOBC, loc, clock, model_fields)
    k = grid.Nz + 1

    @inbounds c[i, j, k] = c[i, j, k - 2] + (c[i, j, k - 1] - c[i, j, k - 3]) / 2

    return nothing
end