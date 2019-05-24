#
# Renamed operators with consistent signatures
#

∂x_faa(i, j, k, grid, u::AbstractArray) = δx_f2c(grid, u, i, j, k) / grid.Δx
∂y_afa(i, j, k, grid, v::AbstractArray) = δy_f2c(grid, v, i, j, k) / grid.Δy
∂z_aaf(i, j, k, grid, w::AbstractArray) = δz_f2c(grid, w, i, j, k) / grid.Δz
∂x_caa(i, j, k, grid, ϕ::AbstractArray) = δx_c2f(grid, ϕ, i, j, k) / grid.Δx
∂y_aca(i, j, k, grid, ϕ::AbstractArray) = δy_c2f(grid, ϕ, i, j, k) / grid.Δy
∂z_aac(i, j, k, grid, ϕ::AbstractArray) = δz_c2f(grid, ϕ, i, j, k) / grid.Δz

#
# Differential operators for functions
#

"""
    ∂x_caa(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `caa` in `x`, across `faa`.
"""
function ∂x_caa(i, j, k, grid, F::Function, args...)
    i⁻ = decmod1(i, grid.Nx)
    return (F(i, j, k, grid, args...) - F(i⁻, j, k, grid, args...)) / grid.Δx
end

"""
    ∂x_faa(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `faa` in `x`, across `caa`.
"""
function ∂x_faa(i, j, k, grid, F::Function, args...)
    i⁺ = incmod1(i, grid.Nx)
    return (F(i⁺, j, k, grid, args...) - F(i, j, k, grid, args...)) / grid.Δx
end

"""
    ∂y_aca(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aca` in `y`, across `afa`.
"""
function ∂y_aca(i, j, k, grid, F::Function, args...)
    j⁻ = decmod1(j, grid.Ny)
    return (F(i, j, k, grid, args...) - F(i, j⁻, k, grid, args...)) / grid.Δy
end

"""
    ∂y_afa(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `afa` in `y`, across `aca`.
"""
function ∂y_afa(i, j, k, grid, F::Function, args...)
    j⁺ = incmod1(j, grid.Ny)
    return (F(i, j⁺, k, grid, args...) - F(i, j, k, grid, args...)) / grid.Δy
end

"""
    ∂z_aac(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aac` in `z`, across `aaf`.
"""
function ∂z_aac(i, j, k, grid::Grid{T}, F::Function, args...) where T
    if k < 2
        return -zero(T)
    else
        return (F(i, j, k-1, grid, args...) - F(i, j, k, grid, args...)) / grid.Δz
    end
end

"""
    ∂z_aaf(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aaf` in `z`, across `aac`.
"""
function ∂z_aaf(i, j, k, grid::Grid{T}, F::Function, args...) where T
    if k >= grid.Nz
        return -zero(T)
    else
        return (F(i, j, k, grid, args...) - F(i, j, k+1, grid, args...)) / grid.Δz
    end
end

#
# Double differentiation
#

∂x²_faa(i, j, k, grid, u) = ∂x_caa(i, j, k, grid, ∂x_faa, u)
∂x²_caa(i, j, k, grid, ϕ) = ∂x_faa(i, j, k, grid, ∂x_caa, ϕ)

∂y²_afa(i, j, k, grid, v) = ∂y_aca(i, j, k, grid, ∂y_afa, v)
∂y²_aca(i, j, k, grid, ϕ) = ∂y_afa(i, j, k, grid, ∂y_aca, ϕ)

∂z²_aaf(i, j, k, grid, w) = ∂z_aac(i, j, k, grid, ∂z_aaf, w)
∂z²_aac(i, j, k, grid, ϕ) = ∂z_aaf(i, j, k, grid, ∂z_aac, ϕ)

#
# Interpolation operations for functions
#

"""
    ▶x_caa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `caa` to `faa`."
"""
function ▶x_caa(i, j, k, grid::Grid{T}, F::Function, args...) where T
    i⁻¹ = decmod1(i, grid.Nx)
    return T(0.5) * (F(i, j, k, grid, args...) + F(i⁻¹, j, k, grid, args...))
end

"""
    ▶y_aca(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aca` to `afa`.
"""
function ▶y_aca(i, j, k, grid::Grid{T}, F::Function, args...) where T
    j⁻¹ = decmod1(j, grid.Ny)
    return T(0.5) * (F(i, j, k, grid, args...) + F(i, j⁻¹, k, grid, args...))
end

"""
    ▶z_aac(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aac` to `aaf`.
"""
function ▶z_aac(i, j, k, grid::Grid{T}, F::Function, args...) where T
    if k < 2
        return T(0.5) * F(i, j, k, grid, args...)
    else
        return T(0.5) * (F(i, j, k, grid, args...) + F(i, j, k-1, grid, args...))
    end
end

"""
    ▶x_faa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `faa` to `caa`."
"""
function ▶x_faa(i, j, k, grid::Grid{T}, F::Function, args...) where T
    i⁺¹ = incmod1(i, grid.Nx)
    return T(0.5) * (F(i⁺¹, j, k, grid, args...) + F(i, j, k, grid, args...))
end

"""
    ▶y_afa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `afa` to `aca`."
"""
function ▶y_afa(i, j, k, grid::Grid{T}, F::Function, args...) where T
    j⁺¹ = incmod1(j, grid.Ny)
    return T(0.5) * (F(i, j⁺¹, k, grid, args...) + F(i, j, k, grid, args...))
end

"""
    ▶z_aaf(i, j, k, grid::Grid{T}, F, args...) where T

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aaf` to `aac`.
"""
function ▶z_aaf(i, j, k, grid::Grid{T}, F::Function, args...) where T
    if k >= grid.Nz
        return T(0.5) * F(i, j, k, grid, args...)
    else
        return T(0.5) * (F(i, j, k+1, grid, args...) + F(i, j, k, grid, args...))
    end
end

# Convenience operators for "interpolating constants"
▶x_caa(i, j, k, grid, F::Number, args...) = F
▶x_faa(i, j, k, grid, F::Number, args...) = F
▶y_aca(i, j, k, grid, F::Number, args...) = F
▶y_afa(i, j, k, grid, F::Number, args...) = F
▶z_aac(i, j, k, grid, F::Number, args...) = F
▶z_aaf(i, j, k, grid, F::Number, args...) = F

# 12 operators for 4-way interpolation.
▶xy_ffa(i, j, k, grid, F, args...) =
 ▶y_afa(i, j, k, grid,
 ▶x_faa, F, args...)

▶xy_cfa(i, j, k, grid, F, args...) =
 ▶y_afa(i, j, k, grid,
 ▶x_caa, F, args...)

▶xy_cca(i, j, k, grid, F, args...) =
 ▶y_aca(i, j, k, grid,
 ▶x_caa, F, args...)

▶xy_fca(i, j, k, grid, F, args...) =
 ▶y_aca(i, j, k, grid,
 ▶x_faa, F, args...)

▶xz_faf(i, j, k, grid, F, args...) =
 ▶z_aaf(i, j, k, grid,
 ▶x_faa, F, args...)

▶xz_caf(i, j, k, grid, F, args...) =
 ▶z_aaf(i, j, k, grid,
 ▶x_caa, F, args...)

▶xz_cac(i, j, k, grid, F, args...) =
 ▶z_aac(i, j, k, grid,
 ▶x_caa, F, args...)

▶xz_fac(i, j, k, grid, F, args...) =
 ▶z_aac(i, j, k, grid,
 ▶x_faa, F, args...)

▶yz_aff(i, j, k, grid, F, args...) =
 ▶z_aaf(i, j, k, grid,
 ▶y_afa, F, args...)

▶yz_acf(i, j, k, grid, F, args...) =
 ▶z_aaf(i, j, k, grid,
 ▶y_aca, F, args...)

▶yz_acc(i, j, k, grid, F, args...) =
 ▶z_aac(i, j, k, grid,
 ▶y_aca, F, args...)

▶yz_afc(i, j, k, grid, F, args...) =
 ▶z_aac(i, j, k, grid,
 ▶y_afa, F, args...)

"""
    ∂x_faa(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_x ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `faa`, where `a` is either `f` or `c`.
The quantity returned has the location `caa`.
"""
function ∂x_faa(κ::Function, Σ::Function, i, j, k, grid, closure, u, v, w, T, S)
    i⁺¹ = incmod1(i, grid.Nx)

    Σ⁺¹ = Σ(i⁺¹, j, k, grid, u, v, w)
    Σ⁻¹ = Σ(i,   j, k, grid, u, v, w)

    κ⁺¹ = κ(i⁺¹, j, k, grid, closure, u, v, w, T, S)
    κ⁻¹ = κ(i,   j, k, grid, closure, u, v, w, T, S)

    return (κ⁺¹ * Σ⁺¹ - κ⁻¹ * Σ⁻¹) / grid.Δx
end

"""
    ∂x_caa(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_x ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `caa`, where `a` is either `f` or `c`.
The quantity returned has the location `faa`.
"""
function ∂x_caa(κ::Function, Σ::Function, i, j, k, grid, closure, u, v, w, T, S)
    i⁻¹ = decmod1(i, grid.Nx)

    Σ⁺¹ = Σ(i,   j, k, grid, u, v, w)
    Σ⁻¹ = Σ(i⁻¹, j, k, grid, u, v, w)

    κ⁺¹ = κ(i,   j, k, grid, closure, u, v, w, T, S)
    κ⁻¹ = κ(i⁻¹, j, k, grid, closure, u, v, w, T, S)

    return (κ⁺¹ * Σ⁺¹ - κ⁻¹ * Σ⁻¹) / grid.Δx
end

"""
    ∂y_afa(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_y ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `afa`, where `a` is either `f` or `c`.
The quantity returned has the location `aca`.
"""
function ∂y_afa(κ::Function, Σ::Function, i, j, k, grid, closure, u, v, w, T, S)
    j⁺¹ = incmod1(j, grid.Ny)

    Σ⁺¹ = Σ(i, j⁺¹, k, grid, u, v, w)
    Σ⁻¹ = Σ(i, j,   k, grid, u, v, w)

    κ⁺¹ = κ(i, j⁺¹, k, grid, closure, u, v, w, T, S)
    κ⁻¹ = κ(i, j,   k, grid, closure, u, v, w, T, S)

    return (κ⁺¹ * Σ⁺¹ - κ⁻¹ * Σ⁻¹) / grid.Δy
end

"""
    ∂y_aca(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_y ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `aca`, where `a` is either `f` or `c`.
The quantity returned has the location `afa`.
"""
function ∂y_aca(κ::Function, Σ::Function, i, j, k, grid, closure, u, v, w, T, S)
    j⁻¹ = decmod1(j, grid.Ny)

    Σ⁺¹ = Σ(i, j,   k, grid, u, v, w)
    Σ⁻¹ = Σ(i, j⁻¹, k, grid, u, v, w)

    κ⁺¹ = κ(i, j,   k, grid, closure, u, v, w, T, S)
    κ⁻¹ = κ(i, j⁻¹, k, grid, closure, u, v, w, T, S)

    return (κ⁺¹ * Σ⁺¹ - κ⁻¹ * Σ⁻¹) / grid.Δy
end

"""
    ∂z_aaf(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_z ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `aaf`, where `a` is either `f` or `c`.
The quantity returned has the location `aac`.
"""
function ∂z_aaf(κ::Function, Σ::Function, i, j, k, grid, closure, u, v, w, T, S)
    Σ⁻¹ = Σ(i, j, k, grid, u, v, w)
    κ⁻¹ = κ(i, j, k, grid, closure, u, v, w, T, S)

    if k == grid.Nz
        return κ⁻¹ * Σ⁻¹ / grid.Δz
    else
        Σ⁺¹ = Σ(i, j, k+1, grid, u, v, w)
        κ⁺¹ = κ(i, j, k+1, grid, closure, u, v, w, T, S)

        # Reversed due to inverse convention for z-indexing
        return (κ⁻¹ * Σ⁻¹ - κ⁺¹ * Σ⁺¹) / grid.Δz
    end
end

"""
    ∂z_aac(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_z ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `aac`, where `a` is either `f` or `c`.
The quantity returned has the location `aaf`.
"""
function ∂z_aac(κ::Function, Σ::Function, i, j, k, grid, closure, u, v, w, T, S)
    Σ⁺¹ = Σ(i, j, k, grid, u, v, w)
    κ⁺¹ = κ(i, j, k, grid, closure, u, v, w, T, S)

    if k == 1
        return - κ⁺¹ * Σ⁺¹ / grid.Δz
    else
        Σ⁻¹ = Σ(i, j, k-1, grid, u, v, w)
        κ⁻¹ = κ(i, j, k-1, grid, closure, u, v, w, T, S)

        # Reversed due to inverse convention for z-indexing
        return (κ⁻¹ * Σ⁻¹ - κ⁺¹ * Σ⁺¹) / grid.Δz
    end
end

# At fcc
∂x_2ν_Σ₁₁(ijk...) = 2 * ∂x_caa(ν_ccc, Σ₁₁, ijk...)
∂y_2ν_Σ₁₂(ijk...) = 2 * ∂y_afa(ν_ffc, Σ₁₂, ijk...)
∂z_2ν_Σ₁₃(ijk...) = 2 * ∂z_aaf(ν_fcf, Σ₁₃, ijk...)

# At cfc
∂x_2ν_Σ₂₁(ijk...) = 2 * ∂x_faa(ν_ffc, Σ₂₁, ijk...)
∂y_2ν_Σ₂₂(ijk...) = 2 * ∂y_aca(ν_ccc, Σ₂₂, ijk...)
∂z_2ν_Σ₂₃(ijk...) = 2 * ∂z_aaf(ν_cff, Σ₂₃, ijk...)

# At ccf
∂x_2ν_Σ₃₁(ijk...) = 2 * ∂x_faa(ν_fcf, Σ₃₁, ijk...)
∂y_2ν_Σ₃₂(ijk...) = 2 * ∂y_afa(ν_cff, Σ₃₂, ijk...)
∂z_2ν_Σ₃₃(ijk...) = 2 * ∂z_aac(ν_ccc, Σ₃₃, ijk...)

"""
    ∂ⱼ_2ν_Σ₁ⱼ

Return the ``x``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₁₁) + ∂y(2 ν Σ₁₁) + ∂z(2 ν Σ₁₁)`

at the location `fcc`.
"""
∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, uvwTS...) = (
    ∂x_2ν_Σ₁₁(i, j, k, grid, closure, uvwTS...)
  + ∂y_2ν_Σ₁₂(i, j, k, grid, closure, uvwTS...)
  + ∂z_2ν_Σ₁₃(i, j, k, grid, closure, uvwTS...)
)

"""
    ∂ⱼ_2ν_Σ₂ⱼ

Return the ``y``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₂₁) + ∂y(2 ν Σ₂₂) + ∂z(2 ν Σ₂₂)`

at the location `ccf`.
"""
∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, uvwTS...) = (
    ∂x_2ν_Σ₂₁(i, j, k, grid, closure, uvwTS...)
  + ∂y_2ν_Σ₂₂(i, j, k, grid, closure, uvwTS...)
  + ∂z_2ν_Σ₂₃(i, j, k, grid, closure, uvwTS...)
)

"""
    ∂ⱼ_2ν_Σ₃ⱼ

Return the ``z``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₃₁) + ∂y(2 ν Σ₃₂) + ∂z(2 ν Σ₃₃)`

at the location `ccf`.
"""
∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, uvwTS...) = (
    ∂x_2ν_Σ₃₁(i, j, k, grid, closure, uvwTS...)
  + ∂y_2ν_Σ₃₂(i, j, k, grid, closure, uvwTS...)
  + ∂z_2ν_Σ₃₃(i, j, k, grid, closure, uvwTS...)
)

"""
    κ_∂x_ϕ(i, j, k, grid, ϕ, κ, closure, u, v, w, T, S)

Return `κ ∂x ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
function κ_∂x_ϕ(i, j, k, grid, ϕ, κ, closure, u, v, w, T, S)
    κ = ▶x_caa(i, j, k, grid, κ, closure, u, v, w, T, S)
    ∂x_ϕ = ∂x_caa(i, j, k, grid, ϕ)
    return κ * ∂x_ϕ
end

"""
    κ_∂y_ϕ(i, j, k, grid, ϕ, κ, closure, u, v, w, T, S)

Return `κ ∂y ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
function κ_∂y_ϕ(i, j, k, grid, ϕ, κ, closure, u, v, w, T, S)
    κ = ▶y_aca(i, j, k, grid, κ, closure, u, v, w, T, S)
    ∂y_ϕ = ∂y_aca(i, j, k, grid, ϕ)
    return κ * ∂y_ϕ
end

"""
    κ_∂z_ϕ(i, j, k, grid, ϕ, κ, closure, u, v, w, T, S)

Return `κ ∂z ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
function κ_∂z_ϕ(i, j, k, grid, ϕ, κ, closure, u, v, w, T, S)
    κ = ▶z_aac(i, j, k, grid, κ, closure, u, v, w, T, S)
    ∂z_ϕ = ∂z_aac(i, j, k, grid, ϕ)
    return κ * ∂z_ϕ
end

"""
    ∂x_κ_∂x_ϕ(κ, ϕ, i, j, k, grid, closure, u, v, w, T, S)

Return `(∂x κ ∂x) ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
∂x_κ_∂x_ϕ(i, j, k, grid, ϕκ...) = ∂x_faa(i, j, k, grid, κ_∂x_ϕ, ϕκ...)

"""
    ∂y_κ_∂y_ϕ(κ, ϕ, i, j, k, grid, closure, u, v, w, T, S)

Return `(∂y κ ∂y) ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
∂y_κ_∂y_ϕ(i, j, k, grid, ϕκ...) = ∂y_afa(i, j, k, grid, κ_∂y_ϕ, ϕκ...)

"""
    ∂z_κ_∂z_ϕ(κ, ϕ, i, j, k, grid, closure, u, v, w, T, S)

Return `(∂z κ ∂z) ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
∂z_κ_∂z_ϕ(i, j, k, grid, ϕκ...) = ∂z_aaf(i, j, k, grid, κ_∂z_ϕ, ϕκ...)

"""
    ∇_κ_∇_ϕ(ϕ, i, j, k, grid, closure, u, v, w, T, S)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ ϕ)` for the turbulence
`closure`, where `ϕ` is an array of scalar data located at cell centers.
"""
∇_κ_∇ϕ(i, j, k, grid, ϕ, closure::IsotropicDiffusivity, uvwTS...) = (
      ∂x_κ_∂x_ϕ(i, j, k, grid, ϕ, κ_ccc, closure, uvwTS...)
    + ∂y_κ_∂y_ϕ(i, j, k, grid, ϕ, κ_ccc, closure, uvwTS...)
    + ∂z_κ_∂z_ϕ(i, j, k, grid, ϕ, κ_ccc, closure, uvwTS...)
    )
