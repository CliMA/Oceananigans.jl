
#
# Velocity gradients
#

# Diagonal
∂x_u(i, j, k, grid, u) = ∂x_faa(i, j, k, grid, u)
∂y_v(i, j, k, grid, v) = ∂y_afa(i, j, k, grid, v)
∂z_w(i, j, k, grid, w) = ∂z_aaf(i, j, k, grid, w)

∂x_v(i, j, k, grid, v) = ∂x_caa(i, j, k, grid, v)
∂x_w(i, j, k, grid, w) = ∂x_caa(i, j, k, grid, w)

∂y_u(i, j, k, grid, u) = ∂y_aca(i, j, k, grid, u)
∂y_w(i, j, k, grid, w) = ∂y_aca(i, j, k, grid, w)

∂z_u(i, j, k, grid, u) = ∂z_aac(i, j, k, grid, u)
∂z_v(i, j, k, grid, v) = ∂z_aac(i, j, k, grid, v)

#
# Strain
#

# ccc strain components
Σ₁₁(i, j, k, grid, u) = ∂x_u(i, j, k, grid, u)
Σ₂₂(i, j, k, grid, v) = ∂y_v(i, j, k, grid, v)
Σ₃₃(i, j, k, grid, w) = ∂z_w(i, j, k, grid, w)

# ccc
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

Σ₂₁ = Σ₁₂
Σ₃₁ = Σ₁₃
Σ₃₂ = Σ₂₃

tr_Σ²(ijk...) = Σ₁₁(ijk...)^2 +  Σ₂₂(ijk...)^2 +  Σ₃₃(ijk...)^2

Σ₁₂²(i, j, k, grid, u, v, w) = Σ₁₂²(i, j, k, grid, u, v)
Σ₁₃²(i, j, k, grid, u, v, w) = Σ₁₃²(i, j, k, grid, u, w)
Σ₂₃²(i, j, k, grid, u, v, w) = Σ₂₃²(i, j, k, grid, v, w)

#
# Interpolation of velocity gradient components to cell centers
#

Σ₁₁_ccc = ∂x_u_ccc = ∂x_u
Σ₂₂_ccc = ∂y_v_ccc = ∂y_v
Σ₃₃_ccc = ∂z_w_ccc = ∂z_w

Σ₁₂_ccc(i, j, k, grid, u, v, w) = ▶xy_ffc(i, j, k, grid, Σ₁₂, u, v, w)
Σ₁₃_ccc(i, j, k, grid, u, v, w) = ▶xz_fcf(i, j, k, grid, Σ₁₃, u, v, w)
Σ₂₃_ccc(i, j, k, grid, u, v, w) = ▶yz_cff(i, j, k, grid, Σ₂₃, u, v, w)

∂y_u_ccc(i, j, k, grid, u, v, w) = ▶xy_ffc(i, j, k, grid, ∂y_u, u, v, w)
∂x_v_ccc(i, j, k, grid, u, v, w) = ▶xy_ffc(i, j, k, grid, ∂x_v, u, v, w)

∂z_u_ccc(i, j, k, grid, u, v, w) = ▶xz_fcf(i, j, k, grid, ∂z_u, u, v, w)
∂x_w_ccc(i, j, k, grid, u, v, w) = ▶xz_fcf(i, j, k, grid, ∂x_w, u, v, w)

∂z_v_ccc(i, j, k, grid, u, v, w) = ▶yz_cff(i, j, k, grid, ∂z_v, u, v, w)
∂y_w_ccc(i, j, k, grid, u, v, w) = ▶yz_cff(i, j, k, grid, ∂y_w, u, v, w)
