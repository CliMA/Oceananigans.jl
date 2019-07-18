#
# Velocity gradients
#

# Diagonal
∂x_u(i, j, k, grid, u) = ∂x_caa(i, j, k, grid, u)
∂y_v(i, j, k, grid, v) = ∂y_aca(i, j, k, grid, v)
∂z_w(i, j, k, grid, w) = ∂z_aac(i, j, k, grid, w)

# Off-diagonal
∂x_v(i, j, k, grid, v) = ∂x_faa(i, j, k, grid, v)
∂x_w(i, j, k, grid, w) = ∂x_faa(i, j, k, grid, w)

∂y_u(i, j, k, grid, u) = ∂y_afa(i, j, k, grid, u)
∂y_w(i, j, k, grid, w) = ∂y_afa(i, j, k, grid, w)

∂z_u(i, j, k, grid, u) = ∂z_aaf(i, j, k, grid, u)
∂z_v(i, j, k, grid, v) = ∂z_aaf(i, j, k, grid, v)

#
# Strain components
#

# ccc strain components
Σ₁₁(i, j, k, grid, u) = ∂x_u(i, j, k, grid, u)
Σ₂₂(i, j, k, grid, v) = ∂y_v(i, j, k, grid, v)
Σ₃₃(i, j, k, grid, w) = ∂z_w(i, j, k, grid, w)

tr_Σ(i, j, k, grid, u, v, w) =
    Σ₁₁(i, j, k, grid, u) + Σ₂₂(i, j, k, grid, v) + Σ₃₃(i, j, k, grid, w)

# ffc
Σ₁₂(i, j, k, grid::Grid{T}, u, v) where T =
    T(0.5) * (∂y_u(i, j, k, grid, u) + ∂x_v(i, j, k, grid, v))

# fcf
Σ₁₃(i, j, k, grid::Grid{T}, u, w) where T =
    T(0.5) * (∂z_u(i, j, k, grid, u) + ∂x_w(i, j, k, grid, w))

# cff
Σ₂₃(i, j, k, grid::Grid{T}, v, w) where T =
    T(0.5) * (∂z_v(i, j, k, grid, v) + ∂y_w(i, j, k, grid, w))

Σ₁₂²(i, j, k, grid, u, v) = Σ₁₂(i, j, k, grid, u, v)^2
Σ₁₃²(i, j, k, grid, u, w) = Σ₁₃(i, j, k, grid, u, w)^2
Σ₂₃²(i, j, k, grid, v, w) = Σ₂₃(i, j, k, grid, v, w)^2

#
# Renamed functions for consistent function signatures
#

∂x_u(i, j, k, grid, u, v, w) = ∂x_u(i, j, k, grid, u)
∂x_v(i, j, k, grid, u, v, w) = ∂x_v(i, j, k, grid, v)
∂x_w(i, j, k, grid, u, v, w) = ∂x_w(i, j, k, grid, w)

∂y_u(i, j, k, grid, u, v, w) = ∂y_u(i, j, k, grid, u)
∂y_v(i, j, k, grid, u, v, w) = ∂y_v(i, j, k, grid, v)
∂y_w(i, j, k, grid, u, v, w) = ∂y_w(i, j, k, grid, w)

∂z_u(i, j, k, grid, u, v, w) = ∂z_u(i, j, k, grid, u)
∂z_v(i, j, k, grid, u, v, w) = ∂z_v(i, j, k, grid, v)
∂z_w(i, j, k, grid, u, v, w) = ∂z_w(i, j, k, grid, w)

Σ₁₁(i, j, k, grid, u, v, w) = Σ₁₁(i, j, k, grid, u)
Σ₂₂(i, j, k, grid, u, v, w) = Σ₂₂(i, j, k, grid, v)
Σ₃₃(i, j, k, grid, u, v, w) = Σ₃₃(i, j, k, grid, w)

Σ₁₂(i, j, k, grid, u, v, w) = Σ₁₂(i, j, k, grid, u, v)
Σ₁₃(i, j, k, grid, u, v, w) = Σ₁₃(i, j, k, grid, u, w)
Σ₂₃(i, j, k, grid, u, v, w) = Σ₂₃(i, j, k, grid, v, w)

# Symmetry relations
Σ₂₁ = Σ₁₂
Σ₃₁ = Σ₁₃
Σ₃₂ = Σ₂₃

# Trace and squared strains
tr_Σ²(ijk...) = Σ₁₁(ijk...)^2 +  Σ₂₂(ijk...)^2 +  Σ₃₃(ijk...)^2

Σ₁₂²(i, j, k, grid, u, v, w) = Σ₁₂²(i, j, k, grid, u, v)
Σ₁₃²(i, j, k, grid, u, v, w) = Σ₁₃²(i, j, k, grid, u, w)
Σ₂₃²(i, j, k, grid, u, v, w) = Σ₂₃²(i, j, k, grid, v, w)
