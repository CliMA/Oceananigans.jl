#####
##### Chain-rule-correct derivative operators for generalized vertical coordinates
#####
##### For grids with GeneralizedVerticalDiscretization (e.g., z⋆ coordinates),
##### horizontal derivatives include the chain-rule correction term that accounts
##### for tilted coordinate surfaces.
#####
##### The key relationship is:
#####   ∂ϕ/∂ξ|_z = ∂ϕ/∂ξ|_r - (∂z/∂ξ|_r)(∂ϕ/∂z)
#####
##### where (x, y, z) are physical coordinates and (ξ, η, r) are computational coordinates.
##### The ∂z/∂ξ|_r term captures the slope of the z-coordinate surfaces.
#####
##### See docs/src/numerical_implementation/generalized_vertical_coordinates.md for theory.
#####

using Oceananigans.Grids: AbstractGeneralizedVerticalGrid

# Alias for brevity
const AGV = AbstractGeneralizedVerticalGrid

#####
##### Helper functions to compute ∂z/∂ξ|_r and ∂z/∂η|_r at various staggerings
#####
##### These compute the horizontal slope of the physical z coordinate at constant r.
##### We use the difference operators δξ, δη (written δx, δy in code) to compute differences,
##### avoiding manual index arithmetic and ensuring consistency with the rest of the codebase.
#####

# For ∂ξᶠᶜᶜ: output is at (F, C, C), input z is at (C, C, C)
@inline ∂ξ_z_at_r_ᶠᶜᶜ(i, j, k, grid::AGV) = δxᶠᶜᶜ(i, j, k, grid, znode, Center(), Center(), Center()) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)

# For ∂ξᶜᶜᶜ: output is at (C, C, C), input z is at (F, C, C)
@inline ∂ξ_z_at_r_ᶜᶜᶜ(i, j, k, grid::AGV) = δxᶜᶜᶜ(i, j, k, grid, znode, Face(), Center(), Center()) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)

# For ∂η_z at (C, F, C): input z is at (C, C, C)
@inline ∂η_z_at_r_ᶜᶠᶜ(i, j, k, grid::AGV) = δyᶜᶠᶜ(i, j, k, grid, znode, Center(), Center(), Center()) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)

# For ∂η_z at (C, C, C): input z is at (C, F, C)
@inline ∂η_z_at_r_ᶜᶜᶜ(i, j, k, grid::AGV) = δyᶜᶜᶜ(i, j, k, grid, znode, Center(), Face(), Center()) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)

# For ∂ξ_z at (F, F, C): input z is at (C, F, C)
@inline ∂ξ_z_at_r_ᶠᶠᶜ(i, j, k, grid::AGV) = δxᶠᶠᶜ(i, j, k, grid, znode, Center(), Face(), Center()) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)

# For ∂η_z at (F, F, C): input z is at (F, C, C)
@inline ∂η_z_at_r_ᶠᶠᶜ(i, j, k, grid::AGV) = δyᶠᶠᶜ(i, j, k, grid, znode, Face(), Center(), Center()) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)

#####
##### Chain-rule-correct horizontal derivatives for generalized grids
#####
##### ∂ϕ/∂ξ|_z = ∂ϕ/∂ξ|_r - (∂z/∂ξ|_r)(∂ϕ/∂z)
#####

#####
##### ξ-derivatives (x-direction) with chain rule correction
#####

# ∂ξᶠᶜᶜ: Field at (C,C,C) → derivative at (F,C,C)
# Used for: ∂ξ(p) in pressure gradient for u-velocity
@inline function ∂xᶠᶜᶜ(i, j, k, grid::AGV, c)
    ∂ξ_at_r = δxᶠᶜᶜ(i, j, k, grid, c) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_c = ∂zᶠᶜᶜ(i, j, k, grid, c)
    ∂ξ_z = ∂ξ_z_at_r_ᶠᶜᶜ(i, j, k, grid)
    return ∂ξ_at_r - ∂ξ_z * ∂z_c
end

@inline function ∂xᶠᶜᶜ(i, j, k, grid::AGV, f::Function, args...)
    ∂ξ_at_r = δxᶠᶜᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_c = ∂zᶠᶜᶜ(i, j, k, grid, f, args...)
    ∂ξ_z = ∂ξ_z_at_r_ᶠᶜᶜ(i, j, k, grid)
    return ∂ξ_at_r - ∂ξ_z * ∂z_c
end

# ∂ξᶜᶜᶜ: Field at (F,C,C) → derivative at (C,C,C)
@inline function ∂xᶜᶜᶜ(i, j, k, grid::AGV, c)
    ∂ξ_at_r = δxᶜᶜᶜ(i, j, k, grid, c) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_c = ∂zᶜᶜᶜ(i, j, k, grid, c)
    ∂ξ_z = ∂ξ_z_at_r_ᶜᶜᶜ(i, j, k, grid)
    return ∂ξ_at_r - ∂ξ_z * ∂z_c
end

@inline function ∂xᶜᶜᶜ(i, j, k, grid::AGV, f::Function, args...)
    ∂ξ_at_r = δxᶜᶜᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_c = ∂zᶜᶜᶜ(i, j, k, grid, f, args...)
    ∂ξ_z = ∂ξ_z_at_r_ᶜᶜᶜ(i, j, k, grid)
    return ∂ξ_at_r - ∂ξ_z * ∂z_c
end

# ∂ξᶠᶠᶜ: Field at (C,F,C) → derivative at (F,F,C)
@inline function ∂xᶠᶠᶜ(i, j, k, grid::AGV, c)
    ∂ξ_at_r = δxᶠᶠᶜ(i, j, k, grid, c) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_c = ∂zᶠᶠᶜ(i, j, k, grid, c)
    ∂ξ_z = ∂ξ_z_at_r_ᶠᶠᶜ(i, j, k, grid)
    return ∂ξ_at_r - ∂ξ_z * ∂z_c
end

@inline function ∂xᶠᶠᶜ(i, j, k, grid::AGV, f::Function, args...)
    ∂ξ_at_r = δxᶠᶠᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_c = ∂zᶠᶠᶜ(i, j, k, grid, f, args...)
    ∂ξ_z = ∂ξ_z_at_r_ᶠᶠᶜ(i, j, k, grid)
    return ∂ξ_at_r - ∂ξ_z * ∂z_c
end

#####
##### η-derivatives (y-direction) with chain rule correction  
#####

# ∂ηᶜᶠᶜ: Field at (C,C,C) → derivative at (C,F,C)
# Used for: ∂η(p) in pressure gradient for v-velocity
@inline function ∂yᶜᶠᶜ(i, j, k, grid::AGV, c)
    ∂η_at_r = δyᶜᶠᶜ(i, j, k, grid, c) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_c = ∂zᶜᶠᶜ(i, j, k, grid, c)
    ∂η_z = ∂η_z_at_r_ᶜᶠᶜ(i, j, k, grid)
    return ∂η_at_r - ∂η_z * ∂z_c
end

@inline function ∂yᶜᶠᶜ(i, j, k, grid::AGV, f::Function, args...)
    ∂η_at_r = δyᶜᶠᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_c = ∂zᶜᶠᶜ(i, j, k, grid, f, args...)
    ∂η_z = ∂η_z_at_r_ᶜᶠᶜ(i, j, k, grid)
    return ∂η_at_r - ∂η_z * ∂z_c
end

# ∂ηᶜᶜᶜ: Field at (C,F,C) → derivative at (C,C,C)
@inline function ∂yᶜᶜᶜ(i, j, k, grid::AGV, c)
    ∂η_at_r = δyᶜᶜᶜ(i, j, k, grid, c) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_c = ∂zᶜᶜᶜ(i, j, k, grid, c)
    ∂η_z = ∂η_z_at_r_ᶜᶜᶜ(i, j, k, grid)
    return ∂η_at_r - ∂η_z * ∂z_c
end

@inline function ∂yᶜᶜᶜ(i, j, k, grid::AGV, f::Function, args...)
    ∂η_at_r = δyᶜᶜᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_c = ∂zᶜᶜᶜ(i, j, k, grid, f, args...)
    ∂η_z = ∂η_z_at_r_ᶜᶜᶜ(i, j, k, grid)
    return ∂η_at_r - ∂η_z * ∂z_c
end

# ∂ηᶠᶠᶜ: Field at (F,C,C) → derivative at (F,F,C)
@inline function ∂yᶠᶠᶜ(i, j, k, grid::AGV, c)
    ∂η_at_r = δyᶠᶠᶜ(i, j, k, grid, c) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_c = ∂zᶠᶠᶜ(i, j, k, grid, c)
    ∂η_z = ∂η_z_at_r_ᶠᶠᶜ(i, j, k, grid)
    return ∂η_at_r - ∂η_z * ∂z_c
end

@inline function ∂yᶠᶠᶜ(i, j, k, grid::AGV, f::Function, args...)
    ∂η_at_r = δyᶠᶠᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_c = ∂zᶠᶠᶜ(i, j, k, grid, f, args...)
    ∂η_z = ∂η_z_at_r_ᶠᶠᶜ(i, j, k, grid)
    return ∂η_at_r - ∂η_z * ∂z_c
end
