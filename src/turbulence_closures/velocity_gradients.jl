#
# Velocity gradients
#

# Diagonal
@inline ∂x_u(i, j, k, grid, u) = ∂x_caa(i, j, k, grid, u)
@inline ∂y_v(i, j, k, grid, v) = ∂y_aca(i, j, k, grid, v)
@inline ∂z_w(i, j, k, grid, w) = ∂z_aac(i, j, k, grid, w)

# Off-diagonal
@inline ∂x_v(i, j, k, grid, v) = ∂x_faa(i, j, k, grid, v)
@inline ∂x_w(i, j, k, grid, w) = ∂x_faa(i, j, k, grid, w)

@inline ∂y_u(i, j, k, grid, u) = ∂y_afa(i, j, k, grid, u)
@inline ∂y_w(i, j, k, grid, w) = ∂y_afa(i, j, k, grid, w)

@inline ∂z_u(i, j, k, grid, u) = ∂z_aaf(i, j, k, grid, u)
@inline ∂z_v(i, j, k, grid, v) = ∂z_aaf(i, j, k, grid, v)

#
# Strain components
#

# ccc strain components
@inline Σ₁₁(i, j, k, grid, u) = ∂x_caa(i, j, k, grid, u)
@inline Σ₂₂(i, j, k, grid, v) = ∂y_aca(i, j, k, grid, v)
@inline Σ₃₃(i, j, k, grid, w) = ∂z_aac(i, j, k, grid, w)

@inline tr_Σ(i, j, k, grid, u, v, w) =
    Σ₁₁(i, j, k, grid, u) + Σ₂₂(i, j, k, grid, v) + Σ₃₃(i, j, k, grid, w)

# ffc
@inline Σ₁₂(i, j, k, grid::Grid{T}, u, v) where T =
    T(0.5) * (∂y_afa(i, j, k, grid, u) + ∂x_faa(i, j, k, grid, v))

# fcf
@inline Σ₁₃(i, j, k, grid::Grid{T}, u, w) where T =
    T(0.5) * (∂z_aaf(i, j, k, grid, u) + ∂x_faa(i, j, k, grid, w))

# cff
@inline Σ₂₃(i, j, k, grid::Grid{T}, v, w) where T =
    T(0.5) * (∂z_aaf(i, j, k, grid, v) + ∂y_afa(i, j, k, grid, w))

@inline Σ₁₂²(i, j, k, grid, u, v) = Σ₁₂(i, j, k, grid, u, v)^2
@inline Σ₁₃²(i, j, k, grid, u, w) = Σ₁₃(i, j, k, grid, u, w)^2
@inline Σ₂₃²(i, j, k, grid, v, w) = Σ₂₃(i, j, k, grid, v, w)^2

#
# Renamed functions for consistent function signatures
#

@inline ∂x_u(i, j, k, grid, u, v, w) = ∂x_u(i, j, k, grid, u)
@inline ∂x_v(i, j, k, grid, u, v, w) = ∂x_v(i, j, k, grid, v)
@inline ∂x_w(i, j, k, grid, u, v, w) = ∂x_w(i, j, k, grid, w)

@inline ∂y_u(i, j, k, grid, u, v, w) = ∂y_u(i, j, k, grid, u)
@inline ∂y_v(i, j, k, grid, u, v, w) = ∂y_v(i, j, k, grid, v)
@inline ∂y_w(i, j, k, grid, u, v, w) = ∂y_w(i, j, k, grid, w)

@inline ∂z_u(i, j, k, grid, u, v, w) = ∂z_u(i, j, k, grid, u)
@inline ∂z_v(i, j, k, grid, u, v, w) = ∂z_v(i, j, k, grid, v)
@inline ∂z_w(i, j, k, grid, u, v, w) = ∂z_w(i, j, k, grid, w)

@inline Σ₁₁(i, j, k, grid, u, v, w) = Σ₁₁(i, j, k, grid, u)
@inline Σ₂₂(i, j, k, grid, u, v, w) = Σ₂₂(i, j, k, grid, v)
@inline Σ₃₃(i, j, k, grid, u, v, w) = Σ₃₃(i, j, k, grid, w)

@inline Σ₁₂(i, j, k, grid, u, v, w) = Σ₁₂(i, j, k, grid, u, v)
@inline Σ₁₃(i, j, k, grid, u, v, w) = Σ₁₃(i, j, k, grid, u, w)
@inline Σ₂₃(i, j, k, grid, u, v, w) = Σ₂₃(i, j, k, grid, v, w)

# Symmetry relations
const Σ₂₁ = Σ₁₂
const Σ₃₁ = Σ₁₃
const Σ₃₂ = Σ₂₃

# Trace and squared strains
@inline tr_Σ²(ijk...) = Σ₁₁(ijk...)^2 +  Σ₂₂(ijk...)^2 +  Σ₃₃(ijk...)^2

@inline Σ₁₂²(i, j, k, grid, u, v, w) = Σ₁₂²(i, j, k, grid, u, v)
@inline Σ₁₃²(i, j, k, grid, u, v, w) = Σ₁₃²(i, j, k, grid, u, w)
@inline Σ₂₃²(i, j, k, grid, u, v, w) = Σ₂₃²(i, j, k, grid, v, w)
