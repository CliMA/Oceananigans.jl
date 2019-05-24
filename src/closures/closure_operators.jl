#
# Differential operators for regular grids
#

∂x_caa(i, j, k, grid, u::AbstractArray) = δx_f2c(grid, u, i, j, k) / grid.Δx
∂y_aca(i, j, k, grid, v::AbstractArray) = δy_f2c(grid, v, i, j, k) / grid.Δy
∂z_aac(i, j, k, grid, w::AbstractArray) = δz_f2c(grid, w, i, j, k) / grid.Δz
∂x_faa(i, j, k, grid, ϕ::AbstractArray) = δx_c2f(grid, ϕ, i, j, k) / grid.Δx
∂y_afa(i, j, k, grid, ϕ::AbstractArray) = δy_c2f(grid, ϕ, i, j, k) / grid.Δy
∂z_aaf(i, j, k, grid, ϕ::AbstractArray) = δz_c2f(grid, ϕ, i, j, k) / grid.Δz

#
# Differentiation and interpolation operators for functions
#
# The geometry of the Oceananigans grid (in one dimension) is
#
# face   cell   face   cell   face
#
#         i-1            i 
#          ↓             ↓
#   |      ×      |      ×      |
#   ↑             ↑             ↑
#  i-1            i            i+1
#
# As a result the interpolation of a quantity ϕᵢ from a cell i to face i 
# (this operation is denoted "▶x_faa" in the code below) is 
#
# (▶ϕ)ᵢ = (ϕᵢ + ϕᵢ₋₁) / 2 .
#
# Conversely, the interpolation of a quantity uᵢ from face i to cell i is given by
#
# (▶u)ᵢ = (uᵢ₊₁ + uᵢ) / 2.
# 
# Derivative operators are defined similarly. Using the symbol "∂" to denote 
# differentiation, we have
#
# (∂ϕ)ᵢ = (ϕᵢ - ϕᵢ₋₁) / Δ
#
# for the differentation of a CellField ending on face i, and
#
# (∂u)ᵢ = (uᵢ₊₁ - uᵢ) / Δ
#
# for the differentation of a FaceField ending on cell i, where Δ is the spacing
# between cell points or face points, appropriately.
# 

#
# Differential operators
#

"""
    ∂x_faa(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `caa` in `x`, across `faa`.
"""
function ∂x_faa(i, j, k, grid, F::Function, args...)
    i⁻ = decmod1(i, grid.Nx)
    return (F(i, j, k, grid, args...) - F(i⁻, j, k, grid, args...)) / grid.Δx
end

"""
    ∂x_caa(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `faa` in `x`, across `caa`.
"""
function ∂x_caa(i, j, k, grid, F::Function, args...)
    i⁺ = incmod1(i, grid.Nx)
    return (F(i⁺, j, k, grid, args...) - F(i, j, k, grid, args...)) / grid.Δx
end

"""
    ∂y_afa(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aca` in `y`, across `afa`.
"""
function ∂y_afa(i, j, k, grid, F::Function, args...)
    j⁻ = decmod1(j, grid.Ny)
    return (F(i, j, k, grid, args...) - F(i, j⁻, k, grid, args...)) / grid.Δy
end

"""
    ∂y_aca(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `afa` in `y`, across `aca`.
"""
function ∂y_aca(i, j, k, grid, F::Function, args...)
    j⁺ = incmod1(j, grid.Ny)
    return (F(i, j⁺, k, grid, args...) - F(i, j, k, grid, args...)) / grid.Δy
end

"""
    ∂z_aaf(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aac` in `z`, across `aaf`.
"""
function ∂z_aaf(i, j, k, grid::Grid{T}, F::Function, args...) where T
    if k < 2
        return -zero(T)
    else
        return (F(i, j, k-1, grid, args...) - F(i, j, k, grid, args...)) / grid.Δz
    end
end

"""
    ∂z_aac(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aaf` in `z`, across `aac`.
"""
function ∂z_aac(i, j, k, grid::Grid{T}, F::Function, args...) where T
    if k == grid.Nz
        return -zero(T)
    else
        return (F(i, j, k, grid, args...) - F(i, j, k+1, grid, args...)) / grid.Δz
    end
end

#
# Double differentiation
#

∂x²_caa(i, j, k, grid, u) = ∂x_faa(i, j, k, grid, ∂x_caa, u)
∂x²_faa(i, j, k, grid, ϕ) = ∂x_caa(i, j, k, grid, ∂x_faa, ϕ)

∂y²_aca(i, j, k, grid, v) = ∂y_afa(i, j, k, grid, ∂y_aca, v)
∂y²_afa(i, j, k, grid, ϕ) = ∂y_aca(i, j, k, grid, ∂y_afa, ϕ)

∂z²_aac(i, j, k, grid, w) = ∂z_aaf(i, j, k, grid, ∂z_aac, w)
∂z²_aaf(i, j, k, grid, ϕ) = ∂z_aac(i, j, k, grid, ∂z_aaf, ϕ)

#
# Interpolation operations for functions
#

"""
    ▶x_faa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `caa` to `faa`."
"""
function ▶x_faa(i, j, k, grid::Grid{T}, F::Function, args...) where T
    i⁻¹ = decmod1(i, grid.Nx)
    return T(0.5) * (F(i, j, k, grid, args...) + F(i⁻¹, j, k, grid, args...))
end

"""
    ▶x_caa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `faa` to `caa`."
"""
function ▶x_caa(i, j, k, grid::Grid{T}, F::Function, args...) where T
    i⁺¹ = incmod1(i, grid.Nx)
    return T(0.5) * (F(i⁺¹, j, k, grid, args...) + F(i, j, k, grid, args...))
end

"""
    ▶y_afa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aca` to `afa`.
"""
function ▶y_afa(i, j, k, grid::Grid{T}, F::Function, args...) where T
    j⁻¹ = decmod1(j, grid.Ny)
    return T(0.5) * (F(i, j, k, grid, args...) + F(i, j⁻¹, k, grid, args...))
end

"""
    ▶y_aca(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `afa` to `aca`."
"""
function ▶y_aca(i, j, k, grid::Grid{T}, F::Function, args...) where T
    j⁺¹ = incmod1(j, grid.Ny)
    return T(0.5) * (F(i, j⁺¹, k, grid, args...) + F(i, j, k, grid, args...))
end

"""
    ▶z_aaf(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aac` to `aaf`.
"""
function ▶z_aaf(i, j, k, grid::Grid{T}, F::Function, args...) where T
    if k == 1
        return T(0.5) * F(i, j, k, grid, args...)
    else
        return T(0.5) * (F(i, j, k, grid, args...) + F(i, j, k-1, grid, args...))
    end
end

"""
    ▶z_aac(i, j, k, grid::Grid{T}, F, args...) where T

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aaf` to `aac`.
"""
function ▶z_aac(i, j, k, grid::Grid{T}, F::Function, args...) where T
    if k == grid.Nz
        return T(0.5) * F(i, j, k, grid, args...)
    else
        return T(0.5) * (F(i, j, k+1, grid, args...) + F(i, j, k, grid, args...))
    end
end

# Convenience operators for "interpolating constants"
▶x_faa(i, j, k, grid, F::Number, args...) = F
▶x_caa(i, j, k, grid, F::Number, args...) = F
▶y_afa(i, j, k, grid, F::Number, args...) = F
▶y_aca(i, j, k, grid, F::Number, args...) = F
▶z_aaf(i, j, k, grid, F::Number, args...) = F
▶z_aac(i, j, k, grid, F::Number, args...) = F

#
# Double interpolation: 12 operators
#

"""
    ▶xy_cca(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `y`, from `ffa` to `cca`.
"""
▶xy_cca(i, j, k, grid, F, args...) = ▶y_aca(i, j, k, grid, ▶x_caa, F, args...)

"""
    ▶xy_fca(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `y`, from `cfa` to `fca`.
"""
▶xy_fca(i, j, k, grid, F, args...) = ▶y_aca(i, j, k, grid, ▶x_faa, F, args...)

"""
    ▶xy_ffa(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `y`, from `cca` to `ffa`.
"""
▶xy_ffa(i, j, k, grid, F, args...) = ▶y_afa(i, j, k, grid, ▶x_faa, F, args...)

"""
    ▶xy_cfa(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `y`, from `fca` to `cfa`.
"""
▶xy_cfa(i, j, k, grid, F, args...) = ▶y_afa(i, j, k, grid, ▶x_caa, F, args...)

"""
    ▶xz_cac(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `z`, from `faf` to `cac`.
"""
▶xz_cac(i, j, k, grid, F, args...) = ▶z_aac(i, j, k, grid, ▶x_caa, F, args...)

"""
    ▶xz_fac(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `z`, from `caf` to `fac`.
"""
▶xz_fac(i, j, k, grid, F, args...) = ▶z_aac(i, j, k, grid, ▶x_faa, F, args...)

"""
    ▶xz_faf(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `z`, from `cac` to `faf`.
"""
▶xz_faf(i, j, k, grid, F, args...) = ▶z_aaf(i, j, k, grid, ▶x_faa, F, args...)

"""
    ▶xz_caf(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `z`, from `fac` to `caf`.
"""
▶xz_caf(i, j, k, grid, F, args...) = ▶z_aaf(i, j, k, grid, ▶x_caa, F, args...)

"""
    ▶yz_acc(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `y` and `z`, from `aff` to `acc`.
"""
▶yz_acc(i, j, k, grid, F, args...) = ▶z_aac(i, j, k, grid, ▶y_aca, F, args...)

"""
    ▶yz_afc(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `y` and `z`, from `acf` to `afc`.
"""
▶yz_afc(i, j, k, grid, F, args...) = ▶z_aac(i, j, k, grid, ▶y_afa, F, args...)

"""
    ▶yz_aff(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `y` and `z`, from `ffa` to `cca`.
"""
▶yz_aff(i, j, k, grid, F, args...) = ▶z_aaf(i, j, k, grid, ▶y_afa, F, args...)

"""
    ▶yz_acf(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `y` and `z`, from `afc` to `acf`.
"""
▶yz_acf(i, j, k, grid, F, args...) = ▶z_aaf(i, j, k, grid, ▶y_aca, F, args...)

"""
    ∂x_caa(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_x ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `faa`, where `a` is either `f` or `c`.
The quantity returned has the location `caa`.
"""
function ∂x_caa(κ::Function, Σ::Function, i, j, k, grid, closure, eos, g, u, v, w, T, S)
    i⁺¹ = incmod1(i, grid.Nx)

    Σ⁺¹ = Σ(i⁺¹, j, k, grid, u, v, w)
    Σ⁻¹ = Σ(i,   j, k, grid, u, v, w)

    κ⁺¹ = κ(i⁺¹, j, k, grid, closure, eos, g, u, v, w, T, S)
    κ⁻¹ = κ(i,   j, k, grid, closure, eos, g, u, v, w, T, S)

    return (κ⁺¹ * Σ⁺¹ - κ⁻¹ * Σ⁻¹) / grid.Δx
end

"""
    ∂x_faa(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_x ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `caa`, where `a` is either `f` or `c`.
The quantity returned has the location `faa`.
"""
function ∂x_faa(κ::Function, Σ::Function, i, j, k, grid, closure, eos, g, u, v, w, T, S)
    i⁻¹ = decmod1(i, grid.Nx)

    Σ⁺¹ = Σ(i,   j, k, grid, u, v, w)
    Σ⁻¹ = Σ(i⁻¹, j, k, grid, u, v, w)

    κ⁺¹ = κ(i,   j, k, grid, closure, eos, g, u, v, w, T, S)
    κ⁻¹ = κ(i⁻¹, j, k, grid, closure, eos, g, u, v, w, T, S)

    return (κ⁺¹ * Σ⁺¹ - κ⁻¹ * Σ⁻¹) / grid.Δx
end

"""
    ∂y_aca(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_y ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `afa`, where `a` is either `f` or `c`.
The quantity returned has the location `aca`.
"""
function ∂y_aca(κ::Function, Σ::Function, i, j, k, grid, closure, eos, g, u, v, w, T, S)
    j⁺¹ = incmod1(j, grid.Ny)

    Σ⁺¹ = Σ(i, j⁺¹, k, grid, u, v, w)
    Σ⁻¹ = Σ(i, j,   k, grid, u, v, w)

    κ⁺¹ = κ(i, j⁺¹, k, grid, closure, eos, g, u, v, w, T, S)
    κ⁻¹ = κ(i, j,   k, grid, closure, eos, g, u, v, w, T, S)

    return (κ⁺¹ * Σ⁺¹ - κ⁻¹ * Σ⁻¹) / grid.Δy
end

"""
    ∂y_afa(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_y ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `aca`, where `a` is either `f` or `c`.
The quantity returned has the location `afa`.
"""
function ∂y_afa(κ::Function, Σ::Function, i, j, k, grid, closure, eos, g, u, v, w, T, S)
    j⁻¹ = decmod1(j, grid.Ny)

    Σ⁺¹ = Σ(i, j,   k, grid, u, v, w)
    Σ⁻¹ = Σ(i, j⁻¹, k, grid, u, v, w)

    κ⁺¹ = κ(i, j,   k, grid, closure, eos, g, u, v, w, T, S)
    κ⁻¹ = κ(i, j⁻¹, k, grid, closure, eos, g, u, v, w, T, S)

    return (κ⁺¹ * Σ⁺¹ - κ⁻¹ * Σ⁻¹) / grid.Δy
end

"""
    ∂z_aac(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_z ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `aaf`, where `a` is either `f` or `c`.
The quantity returned has the location `aac`.
"""
function ∂z_aac(κ::Function, Σ::Function, i, j, k, grid, closure, eos, g, u, v, w, T, S)
    Σ⁻¹ = Σ(i, j, k, grid, u, v, w)
    κ⁻¹ = κ(i, j, k, grid, closure, eos, g, u, v, w, T, S)

    if k == grid.Nz
        return κ⁻¹ * Σ⁻¹ / grid.Δz
    else
        Σ⁺¹ = Σ(i, j, k+1, grid, u, v, w)
        κ⁺¹ = κ(i, j, k+1, grid, closure, eos, g, u, v, w, T, S)

        # Reversed due to inverse convention for z-indexing
        return (κ⁻¹ * Σ⁻¹ - κ⁺¹ * Σ⁺¹) / grid.Δz
    end
end

"""
    ∂z_aaf(κ, Σ, i, j, k, grid, closure, u, v, w, T, S)

Compute ``\\partial_z ( \\kappa \\Sigma )``, where `κ` and `Σ`
are diffusivity and strain functions that return quantities
at the location `aac`, where `a` is either `f` or `c`.
The quantity returned has the location `aaf`.
"""
function ∂z_aaf(κ::Function, Σ::Function, i, j, k, grid, closure, eos, g, u, v, w, T, S)
    Σ⁺¹ = Σ(i, j, k, grid, u, v, w)
    κ⁺¹ = κ(i, j, k, grid, closure, eos, g, u, v, w, T, S)

    if k == 1
        return - κ⁺¹ * Σ⁺¹ / grid.Δz
    else
        Σ⁻¹ = Σ(i, j, k-1, grid, u, v, w)
        κ⁻¹ = κ(i, j, k-1, grid, closure, eos, g, u, v, w, T, S)

        # Reversed due to inverse convention for z-indexing
        return (κ⁻¹ * Σ⁻¹ - κ⁺¹ * Σ⁺¹) / grid.Δz
    end
end

# At fcc
∂x_2ν_Σ₁₁(ijk...) = 2 * ∂x_faa(ν_ccc, Σ₁₁, ijk...)
∂y_2ν_Σ₁₂(ijk...) = 2 * ∂y_aca(ν_ffc, Σ₁₂, ijk...)
∂z_2ν_Σ₁₃(ijk...) = 2 * ∂z_aac(ν_fcf, Σ₁₃, ijk...)

# At cfc
∂x_2ν_Σ₂₁(ijk...) = 2 * ∂x_caa(ν_ffc, Σ₂₁, ijk...)
∂y_2ν_Σ₂₂(ijk...) = 2 * ∂y_afa(ν_ccc, Σ₂₂, ijk...)
∂z_2ν_Σ₂₃(ijk...) = 2 * ∂z_aac(ν_cff, Σ₂₃, ijk...)

# At ccf
∂x_2ν_Σ₃₁(ijk...) = 2 * ∂x_caa(ν_fcf, Σ₃₁, ijk...)
∂y_2ν_Σ₃₂(ijk...) = 2 * ∂y_aca(ν_cff, Σ₃₂, ijk...)
∂z_2ν_Σ₃₃(ijk...) = 2 * ∂z_aaf(ν_ccc, Σ₃₃, ijk...)

"""
    ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, eos, g, u, v, w, T, S)

Return the ``x``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₁₁) + ∂y(2 ν Σ₁₁) + ∂z(2 ν Σ₁₁)`

at the location `fcc`.
"""
∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, args...) = (
    ∂x_2ν_Σ₁₁(i, j, k, grid, closure, args...)
  + ∂y_2ν_Σ₁₂(i, j, k, grid, closure, args...)
  + ∂z_2ν_Σ₁₃(i, j, k, grid, closure, args...)
)

"""
    ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, eos, g, u, v, w, T, S)

Return the ``y``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₂₁) + ∂y(2 ν Σ₂₂) + ∂z(2 ν Σ₂₂)`

at the location `ccf`.
"""
∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, args...) = (
    ∂x_2ν_Σ₂₁(i, j, k, grid, closure, args...)
  + ∂y_2ν_Σ₂₂(i, j, k, grid, closure, args...)
  + ∂z_2ν_Σ₂₃(i, j, k, grid, closure, args...)
)

"""
    ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, eos, g, u, v, w, T, S)

Return the ``z``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₃₁) + ∂y(2 ν Σ₃₂) + ∂z(2 ν Σ₃₃)`

at the location `ccf`.
"""
∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, args...) = (
    ∂x_2ν_Σ₃₁(i, j, k, grid, closure, args...)
  + ∂y_2ν_Σ₃₂(i, j, k, grid, closure, args...)
  + ∂z_2ν_Σ₃₃(i, j, k, grid, closure, args...)
)

"""
    κ_∂x_ϕ(i, j, k, grid, ϕ, κ, closure, eos, g, u, v, w, T, S)

Return `κ ∂x ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
function κ_∂x_ϕ(i, j, k, grid, ϕ, κ, closure, args...)
    κ = ▶x_faa(i, j, k, grid, κ, closure, args...)
    ∂x_ϕ = ∂x_faa(i, j, k, grid, ϕ)
    return κ * ∂x_ϕ
end

"""
    κ_∂y_ϕ(i, j, k, grid, ϕ, κ, closure, eos, g, u, v, w, T, S)

Return `κ ∂y ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
function κ_∂y_ϕ(i, j, k, grid, ϕ, κ, closure, args...)
    κ = ▶y_afa(i, j, k, grid, κ, closure, args...)
    ∂y_ϕ = ∂y_afa(i, j, k, grid, ϕ)
    return κ * ∂y_ϕ
end

"""
    κ_∂z_ϕ(i, j, k, grid, ϕ, κ, closure, eos, g, u, v, w, T, S)

Return `κ ∂z ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
function κ_∂z_ϕ(i, j, k, grid, ϕ, κ, closure, args...)
    κ = ▶z_aaf(i, j, k, grid, κ, closure, args...)
    ∂z_ϕ = ∂z_aaf(i, j, k, grid, ϕ)
    return κ * ∂z_ϕ
end

"""
    ∂x_κ_∂x_ϕ(i, j, k, grid, ϕ, κ, closure, eos, g, u, v, w, T, S)

Return `(∂x κ ∂x) ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
∂x_κ_∂x_ϕ(i, j, k, grid, ϕκ...) = ∂x_caa(i, j, k, grid, κ_∂x_ϕ, ϕκ...)

"""
    ∂y_κ_∂y_ϕ(i, j, k, grid, ϕ, κ, closure, eos, g, u, v, w, T, S)

Return `(∂y κ ∂y) ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
∂y_κ_∂y_ϕ(i, j, k, grid, ϕκ...) = ∂y_aca(i, j, k, grid, κ_∂y_ϕ, ϕκ...)

"""
    ∂z_κ_∂z_ϕ(i, j, k, grid, ϕ, κ, closure, eos, g, u, v, w, T, S)

Return `(∂z κ ∂z) ϕ`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `ϕ` is an array of scalar
data located at cell centers.
"""
∂z_κ_∂z_ϕ(i, j, k, grid, ϕκ...) = ∂z_aac(i, j, k, grid, κ_∂z_ϕ, ϕκ...)

"""
    ∇_κ_∇_ϕ(i, j, k, grid, ϕ, closure, eos, g, u, v, w, T, S)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ ϕ)` for the turbulence
`closure`, where `ϕ` is an array of scalar data located at cell centers.
"""
∇_κ_∇ϕ(i, j, k, grid, ϕ, closure::IsotropicDiffusivity, args...) = (
      ∂x_κ_∂x_ϕ(i, j, k, grid, ϕ, κ_ccc, closure, args...)
    + ∂y_κ_∂y_ϕ(i, j, k, grid, ϕ, κ_ccc, closure, args...)
    + ∂z_κ_∂z_ϕ(i, j, k, grid, ϕ, κ_ccc, closure, args...)
    )
