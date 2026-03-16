#####
##### Generalized coordinate derivatives for mutable vertical grids
#####
##### For z-star coordinates where z(ξ, η, r, t) = η_fs + σ·r, derivatives transform as:
#####
##### Horizontal derivatives (chain rule):
#####   ∂ϕ/∂x|_z = ∂ϕ/∂x|_r - (∂z/∂x|_r)(∂ϕ/∂z)
#####   ∂ϕ/∂y|_z = ∂ϕ/∂y|_r - (∂z/∂y|_r)(∂ϕ/∂z)
#####
##### Vertical derivatives (stretching):
#####   ∂ϕ/∂z = (1/σ)(∂ϕ/∂r)
#####
##### Note: Vertical derivatives are already correct because Δz = σ·Δr is used
##### in the spacing operators for mutable grids (see time_variable_grid_operators.jl).
#####
##### The grid slopes ∂z/∂x|_r and ∂z/∂y|_r are computed using difference operators
##### (not derivatives) to avoid recursion.
#####

using Oceananigans.Grids: znode, AbstractMutableGrid

const AMG = AbstractMutableGrid

#####
##### Grid slope functions: ∂z/∂x|_r and ∂z/∂y|_r at various staggerings
#####
##### We use difference operators (δx, δy) instead of derivative operators (∂x, ∂y)
##### to avoid infinite recursion, since we're overriding ∂x/∂y.
#####

# x-direction slopes at different staggerings
@inline ∂x_zᶠᶜᶜ(i, j, k, grid) = δxᶠᶜᶜ(i, j, k, grid, znode, C(), C(), C()) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline ∂x_zᶜᶜᶜ(i, j, k, grid) = δxᶜᶜᶜ(i, j, k, grid, znode, F(), C(), C()) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline ∂x_zᶠᶜᶠ(i, j, k, grid) = δxᶠᶜᶠ(i, j, k, grid, znode, C(), C(), F()) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
@inline ∂x_zᶜᶠᶜ(i, j, k, grid) = δxᶜᶠᶜ(i, j, k, grid, znode, F(), F(), C()) * Δx⁻¹ᶜᶠᶜ(i, j, k, grid)
@inline ∂x_zᶠᶠᶜ(i, j, k, grid) = δxᶠᶠᶜ(i, j, k, grid, znode, C(), F(), C()) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)

# y-direction slopes at different staggerings
@inline ∂y_zᶜᶠᶜ(i, j, k, grid) = δyᶜᶠᶜ(i, j, k, grid, znode, C(), C(), C()) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
@inline ∂y_zᶜᶜᶜ(i, j, k, grid) = δyᶜᶜᶜ(i, j, k, grid, znode, C(), F(), C()) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline ∂y_zᶜᶠᶠ(i, j, k, grid) = δyᶜᶠᶠ(i, j, k, grid, znode, C(), C(), F()) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
@inline ∂y_zᶠᶜᶜ(i, j, k, grid) = δyᶠᶜᶜ(i, j, k, grid, znode, F(), F(), C()) * Δy⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline ∂y_zᶠᶠᶜ(i, j, k, grid) = δyᶠᶠᶜ(i, j, k, grid, znode, F(), C(), C()) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)

#####
##### Disambiguation for Number arguments (derivative of a constant is zero)
#####

@inline ∂xᶠᶜᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂xᶜᶜᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂xᶠᶜᶠ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂xᶜᶠᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂xᶠᶠᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)

@inline ∂yᶜᶠᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂yᶜᶜᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂yᶜᶠᶠ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂yᶠᶜᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)
@inline ∂yᶠᶠᶜ(i, j, k, grid::AMG, c::Number) = zero(grid)

#####
##### Chain-rule-correct x-derivatives: ∂ϕ/∂x|_z = ∂ϕ/∂x|_r - (∂z/∂x|_r)(∂ϕ/∂z)
#####

# ∂xᶠᶜᶜ: tracer/buoyancy/pressure x-derivatives (most common)
@inline function ∂xᶠᶜᶜ(i, j, k, grid::AMG, ϕ)
    ∂x_at_r = δxᶠᶜᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, ϕ)
    ∂x_z = ∂x_zᶠᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function ∂xᶠᶜᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂x_at_r = δxᶠᶜᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, f, args...)
    ∂x_z = ∂x_zᶠᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

# ∂xᶜᶜᶜ: filtered velocity derivatives (Smagorinsky)
@inline function ∂xᶜᶜᶜ(i, j, k, grid::AMG, ϕ)
    ∂x_at_r = δxᶜᶜᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, ϕ)
    ∂x_z = ∂x_zᶜᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function ∂xᶜᶜᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂x_at_r = δxᶜᶜᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, f, args...)
    ∂x_z = ∂x_zᶜᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

# ∂xᶠᶜᶠ: w x-derivative
@inline function ∂xᶠᶜᶠ(i, j, k, grid::AMG, ϕ)
    ∂x_at_r = δxᶠᶜᶠ(i, j, k, grid, ϕ) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶠ(i, j, k, grid, ∂zᶜᶜᶜ, ϕ)
    ∂x_z = ∂x_zᶠᶜᶠ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function ∂xᶠᶜᶠ(i, j, k, grid::AMG, f::Function, args...)
    ∂x_at_r = δxᶠᶜᶠ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶠ(i, j, k, grid, ∂zᶜᶜᶜ, f, args...)
    ∂x_z = ∂x_zᶠᶜᶠ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

# ∂xᶜᶠᶜ: vorticity x-derivative (Leith)
@inline function ∂xᶜᶠᶜ(i, j, k, grid::AMG, ϕ)
    ∂x_at_r = δxᶜᶠᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶠᶠ, ϕ)
    ∂x_z = ∂x_zᶜᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function ∂xᶜᶠᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂x_at_r = δxᶜᶠᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶠᶠ, f, args...)
    ∂x_z = ∂x_zᶜᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

# ∂xᶠᶠᶜ: filtered v x-derivative
@inline function ∂xᶠᶠᶜ(i, j, k, grid::AMG, ϕ)
    ∂x_at_r = δxᶠᶠᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶠᶠ, ϕ)
    ∂x_z = ∂x_zᶠᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function ∂xᶠᶠᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂x_at_r = δxᶠᶠᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶠᶠ, f, args...)
    ∂x_z = ∂x_zᶠᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

#####
##### Chain-rule-correct y-derivatives: ∂ϕ/∂y|_z = ∂ϕ/∂y|_r - (∂z/∂y|_r)(∂ϕ/∂z)
#####

# ∂yᶜᶠᶜ: tracer/buoyancy/pressure y-derivatives (most common)
@inline function ∂yᶜᶠᶜ(i, j, k, grid::AMG, ϕ)
    ∂y_at_r = δyᶜᶠᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, ϕ)
    ∂y_z = ∂y_zᶜᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function ∂yᶜᶠᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂y_at_r = δyᶜᶠᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, f, args...)
    ∂y_z = ∂y_zᶜᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

# ∂yᶜᶜᶜ: filtered velocity derivatives
@inline function ∂yᶜᶜᶜ(i, j, k, grid::AMG, ϕ)
    ∂y_at_r = δyᶜᶜᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶜᶠᶠ, ϕ)
    ∂y_z = ∂y_zᶜᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function ∂yᶜᶜᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂y_at_r = δyᶜᶜᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶜᶠᶠ, f, args...)
    ∂y_z = ∂y_zᶜᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

# ∂yᶜᶠᶠ: w y-derivative
@inline function ∂yᶜᶠᶠ(i, j, k, grid::AMG, ϕ)
    ∂y_at_r = δyᶜᶠᶠ(i, j, k, grid, ϕ) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶠ(i, j, k, grid, ∂zᶜᶜᶜ, ϕ)
    ∂y_z = ∂y_zᶜᶠᶠ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function ∂yᶜᶠᶠ(i, j, k, grid::AMG, f::Function, args...)
    ∂y_at_r = δyᶜᶠᶠ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶠ(i, j, k, grid, ∂zᶜᶜᶜ, f, args...)
    ∂y_z = ∂y_zᶜᶠᶠ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

# ∂yᶠᶜᶜ: vorticity y-derivative
@inline function ∂yᶠᶜᶜ(i, j, k, grid::AMG, ϕ)
    ∂y_at_r = δyᶠᶜᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶠᶠᶠ, ϕ)
    ∂y_z = ∂y_zᶠᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function ∂yᶠᶜᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂y_at_r = δyᶠᶜᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶠᶠᶠ, f, args...)
    ∂y_z = ∂y_zᶠᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

# ∂yᶠᶠᶜ: filtered u y-derivative
@inline function ∂yᶠᶠᶜ(i, j, k, grid::AMG, ϕ)
    ∂y_at_r = δyᶠᶠᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶠᶜᶠ, ϕ)
    ∂y_z = ∂y_zᶠᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function ∂yᶠᶠᶜ(i, j, k, grid::AMG, f::Function, args...)
    ∂y_at_r = δyᶠᶠᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶠᶜᶠ, f, args...)
    ∂y_z = ∂y_zᶠᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

# Note: For z-reduced fields (fields with Nothing as z-location), the chain-rule
# correction term (∂z/∂x|_r)(∂ϕ/∂z) is automatically zero since ∂ϕ/∂z = 0 for such fields.
# Therefore, the general implementations above correctly return ∂ϕ/∂x|_z = ∂ϕ/∂x|_r.
