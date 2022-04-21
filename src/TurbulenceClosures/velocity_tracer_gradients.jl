#####
##### Velocity gradients
#####

# Diagonal
@inline ∂x_u(i, j, k, grid, u) = ∂xᶜᶜᶜ(i, j, k, grid, u)
@inline ∂y_v(i, j, k, grid, v) = ∂yᶜᶜᶜ(i, j, k, grid, v)
@inline ∂z_w(i, j, k, grid, w) = ∂zᶜᶜᶜ(i, j, k, grid, w)

# Off-diagonal
@inline ∂x_v(i, j, k, grid, v) = ∂xᶠᶠᶜ(i, j, k, grid, v)
@inline ∂x_w(i, j, k, grid, w) = ∂xᶠᶜᶜ(i, j, k, grid, w)

@inline ∂y_u(i, j, k, grid, u) = ∂yᶠᶠᶜ(i, j, k, grid, u)
@inline ∂y_w(i, j, k, grid, w) = ∂yᶜᶠᶜ(i, j, k, grid, w)

@inline ∂z_u(i, j, k, grid, u) = ∂zᶠᶜᶠ(i, j, k, grid, u)
@inline ∂z_v(i, j, k, grid, v) = ∂zᶜᶠᶠ(i, j, k, grid, v)

#####
##### Strain components
#####

# ccc strain components
@inline Σ₁₁(i, j, k, grid, u) = ∂xᶜᶜᶜ(i, j, k, grid, u)
@inline Σ₂₂(i, j, k, grid, v) = ∂yᶜᶜᶜ(i, j, k, grid, v)
@inline Σ₃₃(i, j, k, grid, w) = ∂zᶜᶜᶜ(i, j, k, grid, w)

@inline tr_Σ(i, j, k, grid, u, v, w) =
    Σ₁₁(i, j, k, grid, u) + Σ₂₂(i, j, k, grid, v) + Σ₃₃(i, j, k, grid, w)

# ffc
@inline Σ₁₂(i, j, k, grid::AbstractGrid{FT}, u, v) where FT =
    FT(0.5) * (∂y_u(i, j, k, grid, u) + ∂x_v(i, j, k, grid, v))

# fcf
@inline Σ₁₃(i, j, k, grid::AbstractGrid{FT}, u, w) where FT =
    FT(0.5) * (∂z_u(i, j, k, grid, u) + ∂x_w(i, j, k, grid, w))

# cff
@inline Σ₂₃(i, j, k, grid::AbstractGrid{FT}, v, w) where FT =
    FT(0.5) * (∂z_v(i, j, k, grid, v) + ∂y_w(i, j, k, grid, w))

@inline Σ₁₂²(i, j, k, grid, u, v) = Σ₁₂(i, j, k, grid, u, v)^2
@inline Σ₁₃²(i, j, k, grid, u, w) = Σ₁₃(i, j, k, grid, u, w)^2
@inline Σ₂₃²(i, j, k, grid, v, w) = Σ₂₃(i, j, k, grid, v, w)^2

#####
##### Renamed functions for consistent function signatures
#####

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

#####
##### Same-location velocity products
#####

# ccc
@inline ∂x_u²(ijk...) = ∂x_u(ijk...)^2
@inline ∂y_v²(ijk...) = ∂y_v(ijk...)^2
@inline ∂z_w²(ijk...) = ∂z_w(ijk...)^2

# ffc
@inline ∂x_v²(ijk...) = ∂x_v(ijk...)^2
@inline ∂y_u²(ijk...) = ∂y_u(ijk...)^2

@inline ∂x_v_Σ₁₂(ijk...) = ∂x_v(ijk...) * Σ₁₂(ijk...)
@inline ∂y_u_Σ₁₂(ijk...) = ∂y_u(ijk...) * Σ₁₂(ijk...)

# fcf
@inline ∂z_u²(ijk...) = ∂z_u(ijk...)^2
@inline ∂x_w²(ijk...) = ∂x_w(ijk...)^2

@inline ∂x_w_Σ₁₃(ijk...) = ∂x_w(ijk...) * Σ₁₃(ijk...)
@inline ∂z_u_Σ₁₃(ijk...) = ∂z_u(ijk...) * Σ₁₃(ijk...)

# cff
@inline ∂z_v²(ijk...) = ∂z_v(ijk...)^2
@inline ∂y_w²(ijk...) = ∂y_w(ijk...)^2
@inline ∂z_v_Σ₂₃(ijk...) = ∂z_v(ijk...) * Σ₂₃(ijk...)
@inline ∂y_w_Σ₂₃(ijk...) = ∂y_w(ijk...) * Σ₂₃(ijk...)

#####
##### Tracer gradients
#####

@inline ∂x_c²(ijk...) = ∂xᶠᶜᶜ(ijk...)^2
@inline ∂y_c²(ijk...) = ∂yᶜᶠᶜ(ijk...)^2
@inline ∂z_c²(ijk...) = ∂zᶜᶜᶠ(ijk...)^2

#####
##### Normalized gradients
#####

# ccc
const norm_∂x_u = ∂x_u
const norm_∂y_v = ∂y_v
const norm_∂z_w = ∂z_w

# ffc
@inline norm_∂x_v(i, j, k, grid, v) =
    Δᶠx_ffc(i, j, k, grid) / Δᶠy_ffc(i, j, k, grid) * ∂x_v(i, j, k, grid, v)

@inline norm_∂y_u(i, j, k, grid, u) =
    Δᶠy_ffc(i, j, k, grid) / Δᶠx_ffc(i, j, k, grid) * ∂y_u(i, j, k, grid, u)

# fcf
@inline norm_∂x_w(i, j, k, grid, w) =
    Δᶠx_fcf(i, j, k, grid) / Δᶠz_fcf(i, j, k, grid) * ∂x_w(i, j, k, grid, w)

@inline norm_∂z_u(i, j, k, grid, u) =
    Δᶠz_fcf(i, j, k, grid) / Δᶠx_fcf(i, j, k, grid) * ∂z_u(i, j, k, grid, u)

# cff
@inline norm_∂y_w(i, j, k, grid, w) =
    Δᶠy_cff(i, j, k, grid) / Δᶠz_cff(i, j, k, grid) * ∂y_w(i, j, k, grid, w)

@inline norm_∂z_v(i, j, k, grid, v) =
    Δᶠz_cff(i, j, k, grid) / Δᶠy_cff(i, j, k, grid) * ∂z_v(i, j, k, grid, v)

# tracers
@inline norm_∂x_c(i, j, k, grid, c) = Δᶠx_fcc(i, j, k, grid) * ∂xᶠᶜᶜ(i, j, k, grid, c)
@inline norm_∂y_c(i, j, k, grid, c) = Δᶠy_cfc(i, j, k, grid) * ∂yᶜᶠᶜ(i, j, k, grid, c)
@inline norm_∂z_c(i, j, k, grid, c) = Δᶠz_ccf(i, j, k, grid) * ∂zᶜᶜᶠ(i, j, k, grid, c)

#####
##### Strain operators
#####

# ccc
@inline norm_Σ₁₁(i, j, k, grid, u) = norm_∂x_u(i, j, k, grid, u)
@inline norm_Σ₂₂(i, j, k, grid, v) = norm_∂y_v(i, j, k, grid, v)
@inline norm_Σ₃₃(i, j, k, grid, w) = norm_∂z_w(i, j, k, grid, w)

@inline norm_tr_Σ(i, j, k, grid, u, v, w) =
    norm_Σ₁₁(i, j, k, grid, u) + norm_Σ₂₂(i, j, k, grid, v) + norm_Σ₃₃(i, j, k, grid, w)

# ffc
@inline norm_Σ₁₂(i, j, k, grid::AbstractGrid{T}, u, v) where T =
    T(0.5) * (norm_∂y_u(i, j, k, grid, u) + norm_∂x_v(i, j, k, grid, v))

# fcf
@inline norm_Σ₁₃(i, j, k, grid::AbstractGrid{T}, u, w) where T =
    T(0.5) * (norm_∂z_u(i, j, k, grid, u) + norm_∂x_w(i, j, k, grid, w))

# cff
@inline norm_Σ₂₃(i, j, k, grid::AbstractGrid{T}, v, w) where T =
    T(0.5) * (norm_∂z_v(i, j, k, grid, v) + norm_∂y_w(i, j, k, grid, w))

@inline norm_Σ₁₂²(i, j, k, grid, u, v) = norm_Σ₁₂(i, j, k, grid, u, v)^2
@inline norm_Σ₁₃²(i, j, k, grid, u, w) = norm_Σ₁₃(i, j, k, grid, u, w)^2
@inline norm_Σ₂₃²(i, j, k, grid, v, w) = norm_Σ₂₃(i, j, k, grid, v, w)^2

# Consistent function signatures for convenience:
@inline norm_∂x_v(i, j, k, grid, u, v, w) = norm_∂x_v(i, j, k, grid, v)
@inline norm_∂x_w(i, j, k, grid, u, v, w) = norm_∂x_w(i, j, k, grid, w)

@inline norm_∂y_u(i, j, k, grid, u, v, w) = norm_∂y_u(i, j, k, grid, u)
@inline norm_∂y_w(i, j, k, grid, u, v, w) = norm_∂y_w(i, j, k, grid, w)

@inline norm_∂z_u(i, j, k, grid, u, v, w) = norm_∂z_u(i, j, k, grid, u)
@inline norm_∂z_v(i, j, k, grid, u, v, w) = norm_∂z_v(i, j, k, grid, v)

@inline norm_Σ₁₁(i, j, k, grid, u, v, w) = norm_Σ₁₁(i, j, k, grid, u)
@inline norm_Σ₂₂(i, j, k, grid, u, v, w) = norm_Σ₂₂(i, j, k, grid, v)
@inline norm_Σ₃₃(i, j, k, grid, u, v, w) = norm_Σ₃₃(i, j, k, grid, w)

@inline norm_Σ₁₂(i, j, k, grid, u, v, w) = norm_Σ₁₂(i, j, k, grid, u, v)
@inline norm_Σ₁₃(i, j, k, grid, u, v, w) = norm_Σ₁₃(i, j, k, grid, u, w)
@inline norm_Σ₂₃(i, j, k, grid, u, v, w) = norm_Σ₂₃(i, j, k, grid, v, w)

# Symmetry relations
const norm_Σ₂₁ = norm_Σ₁₂
const norm_Σ₃₁ = norm_Σ₁₃
const norm_Σ₃₂ = norm_Σ₂₃

# Trace and squared strains
@inline norm_tr_Σ²(ijk...) = norm_Σ₁₁(ijk...)^2 +  norm_Σ₂₂(ijk...)^2 +  norm_Σ₃₃(ijk...)^2

@inline norm_Σ₁₂²(i, j, k, grid, u, v, w) = norm_Σ₁₂²(i, j, k, grid, u, v)
@inline norm_Σ₁₃²(i, j, k, grid, u, v, w) = norm_Σ₁₃²(i, j, k, grid, u, w)
@inline norm_Σ₂₃²(i, j, k, grid, u, v, w) = norm_Σ₂₃²(i, j, k, grid, v, w)

#####
##### Same-location velocity products
#####

# ccc
@inline norm_∂x_u²(ijk...) = norm_∂x_u(ijk...)^2
@inline norm_∂y_v²(ijk...) = norm_∂y_v(ijk...)^2
@inline norm_∂z_w²(ijk...) = norm_∂z_w(ijk...)^2

# ffc
@inline norm_∂x_v²(ijk...) = norm_∂x_v(ijk...)^2
@inline norm_∂y_u²(ijk...) = norm_∂y_u(ijk...)^2

@inline norm_∂x_v_Σ₁₂(ijk...) = norm_∂x_v(ijk...) * norm_Σ₁₂(ijk...)
@inline norm_∂y_u_Σ₁₂(ijk...) = norm_∂y_u(ijk...) * norm_Σ₁₂(ijk...)

# fcf
@inline norm_∂z_u²(ijk...) = norm_∂z_u(ijk...)^2
@inline norm_∂x_w²(ijk...) = norm_∂x_w(ijk...)^2

@inline norm_∂x_w_Σ₁₃(ijk...) = norm_∂x_w(ijk...) * norm_Σ₁₃(ijk...)
@inline norm_∂z_u_Σ₁₃(ijk...) = norm_∂z_u(ijk...) * norm_Σ₁₃(ijk...)

# cff
@inline norm_∂z_v²(ijk...) = norm_∂z_v(ijk...)^2
@inline norm_∂y_w²(ijk...) = norm_∂y_w(ijk...)^2

@inline norm_∂z_v_Σ₂₃(ijk...) = norm_∂z_v(ijk...) * norm_Σ₂₃(ijk...)
@inline norm_∂y_w_Σ₂₃(ijk...) = norm_∂y_w(ijk...) * norm_Σ₂₃(ijk...)

#####
##### Tracer gradients squared
#####

@inline norm_∂x_c²(ijk...) = norm_∂x_c(ijk...)^2
@inline norm_∂y_c²(ijk...) = norm_∂y_c(ijk...)^2
@inline norm_∂z_c²(ijk...) = norm_∂z_c(ijk...)^2
