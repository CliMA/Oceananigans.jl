#
# Differential operators for regular grids
#

@inline function ∂x_caa(i, j, k, grid, u::AbstractArray) 
    i⁺ = incmod1(i, grid.Nx)    
    return @inbounds (u[i⁺, j, k] - u[i, j, k]) / grid.Δx
end

@inline function ∂x_faa(i, j, k, grid, ϕ::AbstractArray) 
    i⁻ = decmod1(i, grid.Nx)    
    return @inbounds (ϕ[i, j, k] - ϕ[i⁻, j, k]) / grid.Δx
end

@inline function ∂y_aca(i, j, k, grid, v::AbstractArray) 
    j⁺ = incmod1(j, grid.Ny)    
    return @inbounds (v[i, j⁺, k] - v[i, j, k]) / grid.Δy
end

@inline function ∂y_afa(i, j, k, grid, ϕ::AbstractArray) 
    j⁻ = decmod1(j, grid.Ny)    
    return @inbounds (ϕ[i, j, k] - ϕ[i, j⁻, k]) / grid.Δy
end

@inline function ∂z_aac(i, j, k, grid, w::AbstractArray) 
    if k == grid.Nz
        return @inbounds w[i, j, k] / grid.Δz
    else
        return @inbounds (w[i, j, k] - w[i, j, k+1]) / grid.Δz
    end
end

@inline function ∂z_aaf(i, j, k, grid::Grid{T}, ϕ::AbstractArray) where T
    if k == 1
        return -zero(T)
    else
        return @inbounds (ϕ[i, j, k-1] - ϕ[i, j, k]) / grid.Δz
    end
end
    
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
@inline function ∂x_faa(i, j, k, grid, F::Function, args...)
    i⁻ = decmod1(i, grid.Nx)
    return (F(i, j, k, grid, args...) - F(i⁻, j, k, grid, args...)) / grid.Δx
end

"""
    ∂x_caa(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `faa` in `x`, across `caa`.
"""
@inline function ∂x_caa(i, j, k, grid, F::Function, args...)
    i⁺ = incmod1(i, grid.Nx)
    return (F(i⁺, j, k, grid, args...) - F(i, j, k, grid, args...)) / grid.Δx
end

"""
    ∂y_afa(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aca` in `y`, across `afa`.
"""
@inline function ∂y_afa(i, j, k, grid, F::Function, args...)
    j⁻ = decmod1(j, grid.Ny)
    return (F(i, j, k, grid, args...) - F(i, j⁻, k, grid, args...)) / grid.Δy
end

"""
    ∂y_aca(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `afa` in `y`, across `aca`.
"""
@inline function ∂y_aca(i, j, k, grid, F::Function, args...)
    j⁺ = incmod1(j, grid.Ny)
    return (F(i, j⁺, k, grid, args...) - F(i, j, k, grid, args...)) / grid.Δy
end

"""
    ∂z_aaf(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aac` in `z`, across `aaf`.
"""
@inline function ∂z_aaf(i, j, k, grid::Grid{T}, F::Function, args...) where T
    if k == 1
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
@inline function ∂z_aac(i, j, k, grid, F::Function, args...)
    if k == grid.Nz
        return F(i, j, k, grid, args...) / grid.Δz
    else
        return (F(i, j, k, grid, args...) - F(i, j, k+1, grid, args...)) / grid.Δz
    end
end

#
# Double differentiation
#

@inline ∂x²_caa(i, j, k, grid, ϕ) = ∂x_caa(i, j, k, grid, ∂x_faa, ϕ)
@inline ∂x²_faa(i, j, k, grid, u) = ∂x_faa(i, j, k, grid, ∂x_caa, u)

@inline ∂y²_aca(i, j, k, grid, ϕ) = ∂y_aca(i, j, k, grid, ∂y_afa, ϕ)
@inline ∂y²_afa(i, j, k, grid, v) = ∂y_afa(i, j, k, grid, ∂y_aca, v)

@inline ∂z²_aac(i, j, k, grid, ϕ) = ∂z_aac(i, j, k, grid, ∂z_aaf, ϕ)
@inline ∂z²_aaf(i, j, k, grid, w) = ∂z_aaf(i, j, k, grid, ∂z_aac, w)

#
# Interpolation operations for functions
#

"""
    ▶x_faa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `caa` to `faa`."
"""
@inline function ▶x_faa(i, j, k, grid::Grid{T}, F::Function, args...) where T
    i⁻¹ = decmod1(i, grid.Nx)
    return T(0.5) * (F(i, j, k, grid, args...) + F(i⁻¹, j, k, grid, args...))
end

"""
    ▶x_caa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `faa` to `caa`."
"""
@inline function ▶x_caa(i, j, k, grid::Grid{T}, F::Function, args...) where T
    i⁺¹ = incmod1(i, grid.Nx)
    return T(0.5) * (F(i⁺¹, j, k, grid, args...) + F(i, j, k, grid, args...))
end

"""
    ▶y_afa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aca` to `afa`.
"""
@inline function ▶y_afa(i, j, k, grid::Grid{T}, F::Function, args...) where T
    j⁻¹ = decmod1(j, grid.Ny)
    return T(0.5) * (F(i, j, k, grid, args...) + F(i, j⁻¹, k, grid, args...))
end

"""
    ▶y_aca(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `afa` to `aca`."
"""
@inline function ▶y_aca(i, j, k, grid::Grid{T}, F::Function, args...) where T
    j⁺¹ = incmod1(j, grid.Ny)
    return T(0.5) * (F(i, j⁺¹, k, grid, args...) + F(i, j, k, grid, args...))
end

"""
    ▶z_aaf(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aac` to `aaf`.
"""
@inline function ▶z_aaf(i, j, k, grid::Grid{T}, F::Function, args...) where T
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
@inline function ▶z_aac(i, j, k, grid::Grid{T}, F::Function, args...) where T
    if k == grid.Nz
        return T(0.5) * F(i, j, k, grid, args...)
    else
        return T(0.5) * (F(i, j, k+1, grid, args...) + F(i, j, k, grid, args...))
    end
end

# Convenience operators for "interpolating constants"
@inline ▶x_faa(i, j, k, grid, F::Number, args...) = F
@inline ▶x_caa(i, j, k, grid, F::Number, args...) = F
@inline ▶y_afa(i, j, k, grid, F::Number, args...) = F
@inline ▶y_aca(i, j, k, grid, F::Number, args...) = F
@inline ▶z_aaf(i, j, k, grid, F::Number, args...) = F
@inline ▶z_aac(i, j, k, grid, F::Number, args...) = F

#
# Double interpolation: 12 operators
#

"""
    ▶xy_cca(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `y`, from `ffa` to `cca`.
"""
@inline ▶xy_cca(i, j, k, grid, F, args...) = ▶y_aca(i, j, k, grid, ▶x_caa, F, args...)

"""
    ▶xy_fca(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `y`, from `cfa` to `fca`.
"""
@inline ▶xy_fca(i, j, k, grid, F, args...) = ▶y_aca(i, j, k, grid, ▶x_faa, F, args...)

"""
    ▶xy_ffa(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `y`, from `cca` to `ffa`.
"""
@inline ▶xy_ffa(i, j, k, grid, F, args...) = ▶y_afa(i, j, k, grid, ▶x_faa, F, args...)

"""
    ▶xy_cfa(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `y`, from `fca` to `cfa`.
"""
@inline ▶xy_cfa(i, j, k, grid, F, args...) = ▶y_afa(i, j, k, grid, ▶x_caa, F, args...)

"""
    ▶xz_cac(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `z`, from `faf` to `cac`.
"""
@inline ▶xz_cac(i, j, k, grid, F, args...) = ▶z_aac(i, j, k, grid, ▶x_caa, F, args...)

"""
    ▶xz_fac(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `z`, from `caf` to `fac`.
"""
@inline ▶xz_fac(i, j, k, grid, F, args...) = ▶z_aac(i, j, k, grid, ▶x_faa, F, args...)

"""
    ▶xz_faf(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `z`, from `cac` to `faf`.
"""
@inline ▶xz_faf(i, j, k, grid, F, args...) = ▶z_aaf(i, j, k, grid, ▶x_faa, F, args...)

"""
    ▶xz_caf(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x` and `z`, from `fac` to `caf`.
"""
@inline ▶xz_caf(i, j, k, grid, F, args...) = ▶z_aaf(i, j, k, grid, ▶x_caa, F, args...)

"""
    ▶yz_acc(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `y` and `z`, from `aff` to `acc`.
"""
@inline ▶yz_acc(i, j, k, grid, F, args...) = ▶z_aac(i, j, k, grid, ▶y_aca, F, args...)

"""
    ▶yz_afc(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `y` and `z`, from `acf` to `afc`.
"""
@inline ▶yz_afc(i, j, k, grid, F, args...) = ▶z_aac(i, j, k, grid, ▶y_afa, F, args...)

"""
    ▶yz_aff(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `y` and `z`, from `ffa` to `cca`.
"""
@inline ▶yz_aff(i, j, k, grid, F, args...) = ▶z_aaf(i, j, k, grid, ▶y_afa, F, args...)

"""
    ▶yz_acf(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `y` and `z`, from `afc` to `acf`.
"""
@inline ▶yz_acf(i, j, k, grid, F, args...) = ▶z_aaf(i, j, k, grid, ▶y_aca, F, args...)

@inline ν_Σᵢⱼ(i, j, k, grid, ν, Σᵢⱼ, closure, eos, g, u, v, w, T, S) =
    ν(i, j, k, grid, closure, eos, g, u, v, w, T, S) * Σᵢⱼ(i, j, k, grid, u, v, w)

# At fcc
@inline ∂x_2ν_Σ₁₁(i, j, k, grid, args...) = 2 * ∂x_faa(i, j, k, grid, ν_Σᵢⱼ, ν_ccc, Σ₁₁, args...)
@inline ∂y_2ν_Σ₁₂(i, j, k, grid, args...) = 2 * ∂y_aca(i, j, k, grid, ν_Σᵢⱼ, ν_ffc, Σ₁₂, args...)
@inline ∂z_2ν_Σ₁₃(i, j, k, grid, args...) = 2 * ∂z_aac(i, j, k, grid, ν_Σᵢⱼ, ν_fcf, Σ₁₃, args...)

# At cfcᵢⱼ
@inline ∂x_2ν_Σ₂₁(i, j, k, grid, args...) = 2 * ∂x_caa(i, j, k, grid, ν_Σᵢⱼ, ν_ffc, Σ₂₁, args...)
@inline ∂y_2ν_Σ₂₂(i, j, k, grid, args...) = 2 * ∂y_afa(i, j, k, grid, ν_Σᵢⱼ, ν_ccc, Σ₂₂, args...)
@inline ∂z_2ν_Σ₂₃(i, j, k, grid, args...) = 2 * ∂z_aac(i, j, k, grid, ν_Σᵢⱼ, ν_cff, Σ₂₃, args...)

# At ccfᵢⱼ
@inline ∂x_2ν_Σ₃₁(i, j, k, grid, args...) = 2 * ∂x_caa(i, j, k, grid, ν_Σᵢⱼ, ν_fcf, Σ₃₁, args...)
@inline ∂y_2ν_Σ₃₂(i, j, k, grid, args...) = 2 * ∂y_aca(i, j, k, grid, ν_Σᵢⱼ, ν_cff, Σ₃₂, args...)
@inline ∂z_2ν_Σ₃₃(i, j, k, grid, args...) = 2 * ∂z_aaf(i, j, k, grid, ν_Σᵢⱼ, ν_ccc, Σ₃₃, args...)

"""
    ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, eos, g, u, v, w, T, S)

Return the ``x``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₁₁) + ∂y(2 ν Σ₁₁) + ∂z(2 ν Σ₁₁)`

at the location `fcc`.
"""
@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, args...) = (
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
@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, args...) = (
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
@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, args...) = (
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
@inline function κ_∂x_ϕ(i, j, k, grid, ϕ, κ, closure, args...)
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
@inline function κ_∂y_ϕ(i, j, k, grid, ϕ, κ, closure, args...)
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
@inline function κ_∂z_ϕ(i, j, k, grid, ϕ, κ, closure, args...)
    κ = ▶z_aaf(i, j, k, grid, κ, closure, args...)
    ∂z_ϕ = ∂z_aaf(i, j, k, grid, ϕ)
    return κ * ∂z_ϕ
end

"""
    ∇_κ_∇_ϕ(i, j, k, grid, ϕ, closure, eos, g, u, v, w, T, S)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ ϕ)` for the turbulence
`closure`, where `ϕ` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇ϕ(i, j, k, grid, ϕ, closure::IsotropicDiffusivity, args...) = (
      ∂x_caa(i, j, k, grid, κ_∂x_ϕ, ϕ, κ_ccc, closure, args...)
    + ∂y_aca(i, j, k, grid, κ_∂y_ϕ, ϕ, κ_ccc, closure, args...)
    + ∂z_aac(i, j, k, grid, κ_∂z_ϕ, ϕ, κ_ccc, closure, args...)
    )
