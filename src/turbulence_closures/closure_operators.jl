#
# Differential operators for regular grids
#

@inline ∂x_caa(i, j, k, grid, u::AbstractArray) = @inbounds (u[i+1, j, k] - u[i, j, k]) / grid.Δx
@inline ∂x_faa(i, j, k, grid, c::AbstractArray) = @inbounds (c[i, j, k] - c[i-1, j, k]) / grid.Δx

@inline ∂y_aca(i, j, k, grid, v::AbstractArray) = @inbounds (v[i, j+1, k] - v[i, j, k]) / grid.Δy
@inline ∂y_afa(i, j, k, grid, c::AbstractArray) = @inbounds (c[i, j, k] - c[i, j-1, k]) / grid.Δy

@inline ∂z_aac(i, j, k, grid, w::AbstractArray) = @inbounds (w[i, j, k] - w[i, j, k+1]) / grid.Δz
@inline ∂z_aaf(i, j, k, grid, c::AbstractArray) = @inbounds (c[i, j, k-1] - c[i, j, k]) / grid.Δz

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
# As a result the interpolation of a quantity cᵢ from a cell i to face i
# (this operation is denoted "▶x_faa" in the code below) is
#
# (▶c)ᵢ = (cᵢ + cᵢ₋₁) / 2 .
#
# Conversely, the interpolation of a quantity uᵢ from face i to cell i is given by
#
# (▶u)ᵢ = (uᵢ₊₁ + uᵢ) / 2.
#
# Derivative operators are defined similarly. Using the symbol "∂" to denote
# differentiation, we have
#
# (∂c)ᵢ = (cᵢ - cᵢ₋₁) / Δ
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
@inline ∂x_faa(i, j, k, grid, F::TF, args...) where TF<:Function =
    (F(i, j, k, grid, args...) - F(i-1, j, k, grid, args...)) / grid.Δx

"""
    ∂x_caa(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `faa` in `x`, across `caa`.
"""
@inline ∂x_caa(i, j, k, grid, F::TF, args...) where TF<:Function =
    (F(i+1, j, k, grid, args...) - F(i, j, k, grid, args...)) / grid.Δx

"""
    ∂y_afa(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aca` in `y`, across `afa`.
"""
@inline ∂y_afa(i, j, k, grid, F::TF, args...) where TF<:Function =
    (F(i, j, k, grid, args...) - F(i, j-1, k, grid, args...)) / grid.Δy

"""
    ∂y_aca(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `afa` in `y`, across `aca`.
"""
@inline ∂y_aca(i, j, k, grid, F::TF, args...) where TF<:Function =
    (F(i, j+1, k, grid, args...) - F(i, j, k, grid, args...)) / grid.Δy

"""
    ∂z_aaf(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aac` in `z`, across `aaf`.
"""
@inline ∂z_aaf(i, j, k, grid::Grid, F::TF, args...) where TF<:Function =
    (F(i, j, k-1, grid, args...) - F(i, j, k, grid, args...)) / grid.Δz

"""
    ∂z_aac(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aaf` in `z`, across `aac`.
"""
@inline ∂z_aac(i, j, k, grid, F::TF, args...) where TF<:Function =
    (F(i, j, k, grid, args...) - F(i, j, k+1, grid, args...)) / grid.Δz

#
# Double differentiation
#

@inline ∂x²_caa(i, j, k, grid, c) = ∂x_caa(i, j, k, grid, ∂x_faa, c)
@inline ∂x²_faa(i, j, k, grid, u) = ∂x_faa(i, j, k, grid, ∂x_caa, u)

@inline ∂y²_aca(i, j, k, grid, c) = ∂y_aca(i, j, k, grid, ∂y_afa, c)
@inline ∂y²_afa(i, j, k, grid, v) = ∂y_afa(i, j, k, grid, ∂y_aca, v)

@inline ∂z²_aac(i, j, k, grid, c) = ∂z_aac(i, j, k, grid, ∂z_aaf, c)
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
@inline ▶x_faa(i, j, k, grid::Grid{T}, F::TF, args...) where {T, TF<:Function} =
    T(0.5) * (F(i, j, k, grid, args...) + F(i-1, j, k, grid, args...))

"""
    ▶x_caa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `faa` to `caa`."
"""
@inline ▶x_caa(i, j, k, grid::Grid{T}, F::TF, args...) where {T, TF<:Function} =
    return T(0.5) * (F(i+1, j, k, grid, args...) + F(i, j, k, grid, args...))

"""
    ▶y_afa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aca` to `afa`.
"""
@inline ▶y_afa(i, j, k, grid::Grid{T}, F::TF, args...) where {T, TF<:Function} =
    return T(0.5) * (F(i, j, k, grid, args...) + F(i, j-1, k, grid, args...))

"""
    ▶y_aca(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `afa` to `aca`."
"""
@inline ▶y_aca(i, j, k, grid::Grid{T}, F::TF, args...) where {T, TF<:Function} =
    T(0.5) * (F(i, j+1, k, grid, args...) + F(i, j, k, grid, args...))

"""
    ▶z_aaf(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aac` to `aaf`.
"""
@inline ▶z_aaf(i, j, k, grid::Grid{T}, F::TF, args...) where {T, TF<:Function} =
    T(0.5) * (F(i, j, k, grid, args...) + F(i, j, k-1, grid, args...))

"""
    ▶z_aac(i, j, k, grid::Grid{T}, F, args...) where T

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aaf` to `aac`.
"""
@inline ▶z_aac(i, j, k, grid::Grid{T}, F::TF, args...) where {T, TF<:Function} =
    T(0.5) * (F(i, j, k+1, grid, args...) + F(i, j, k, grid, args...))

# Convenience operators for "interpolating constants"
@inline ▶x_faa(i, j, k, grid, F::Number, args...) = F
@inline ▶x_caa(i, j, k, grid, F::Number, args...) = F
@inline ▶y_afa(i, j, k, grid, F::Number, args...) = F
@inline ▶y_aca(i, j, k, grid, F::Number, args...) = F
@inline ▶z_aaf(i, j, k, grid, F::Number, args...) = F
@inline ▶z_aac(i, j, k, grid, F::Number, args...) = F

@inline ▶x_faa(i, j, k, grid::Grid{T}, F::AbstractArray, args...) where T =
    @inbounds T(0.5) * (F[i, j, k] + F[i-1, j, k])

@inline ▶x_caa(i, j, k, grid::Grid{T}, F::AbstractArray, args...) where T =
    @inbounds T(0.5) * (F[i, j, k] + F[i+1, j, k])

@inline ▶y_afa(i, j, k, grid::Grid{T}, F::AbstractArray, args...) where T =
    @inbounds T(0.5) * (F[i, j, k] + F[i, j-1, k])

@inline ▶y_aca(i, j, k, grid::Grid{T}, F::AbstractArray, args...) where T =
    @inbounds T(0.5) * (F[i, j, k] + F[i, j+1, k])

@inline ▶z_aaf(i, j, k, grid::Grid{T}, F::AbstractArray, args...) where T =
    @inbounds T(0.5) * (F[i, j, k] + F[i, j, k-1])

@inline ▶z_aac(i, j, k, grid::Grid{T}, w::AbstractArray, args...) where T =
    @inbounds T(0.5) * (w[i, j, k] + w[i, j, k+1])

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

"""
    ▶xyc_ffc(i, j, k, grid, F, args...)

Interpolate the function

    `F(i, j, k, grid, args...)`

in `x`, `y`, and `z` from `ccf` to `ffc`.
"""
@inline ▶xyz_ffc(i, j, k, grid, F, args...) = ▶x_faa(i, j, k, grid, ▶y_afa, ▶z_aac, F, args...)
@inline ▶xyz_ccf(i, j, k, grid, F, args...) = ▶x_caa(i, j, k, grid, ▶y_aca, ▶z_aaf, F, args...)

"""
    ν_Σᵢⱼ(i, j, k, grid, ν, Σᵢⱼ, closure, eos, g, u, v, w, T, S)

Multiply the viscosity function

    `ν(i, j, k, grid, closure, eos, g, u, v, w, T, S)`

with the strain tensor component function

    `Σᵢⱼ(i, j, k, grid, u, v, w)`

at index `i, j, k`.
"""
@inline ν_Σᵢⱼ(i, j, k, grid, ν::TN, Σᵢⱼ::TS, closure, eos, grav, u, v, w, T, S) where {TN, TS} =
    ν(i, j, k, grid, closure, eos, grav, u, v, w, T, S) * Σᵢⱼ(i, j, k, grid, u, v, w)

@inline ν_Σᵢⱼ_ccc(i, j, k, grid, ν::TN, Σᵢⱼ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ν[i, j, k] * Σᵢⱼ(i, j, k, grid, u, v, w)

@inline ν_Σᵢⱼ_ffc(i, j, k, grid, ν::TN, Σᵢⱼ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ▶xy_ffa(i, j, k, grid, ν) * Σᵢⱼ(i, j, k, grid, u, v, w)

@inline ν_Σᵢⱼ_fcf(i, j, k, grid, ν::TN, Σᵢⱼ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ▶xz_faf(i, j, k, grid, ν) * Σᵢⱼ(i, j, k, grid, u, v, w)

@inline ν_Σᵢⱼ_cff(i, j, k, grid, ν::TN, Σᵢⱼ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ▶yz_aff(i, j, k, grid, ν) * Σᵢⱼ(i, j, k, grid, u, v, w)

#
# Stress divergences
#

# At fcc
@inline ∂x_2ν_Σ₁₁(i, j, k, grid, closure, u, v, w, diffusivities) =
    2 * ∂x_faa(i, j, k, grid, ν_Σᵢⱼ_ccc, diffusivities.νₑ, Σ₁₁, u, v, w)

@inline ∂y_2ν_Σ₁₂(i, j, k, grid, closure, u, v, w, diffusivities) =
    2 * ∂y_aca(i, j, k, grid, ν_Σᵢⱼ_ffc, diffusivities.νₑ, Σ₁₂, u, v, w)

@inline ∂z_2ν_Σ₁₃(i, j, k, grid, closure, u, v, w, diffusivities) =
    2 * ∂z_aac(i, j, k, grid, ν_Σᵢⱼ_fcf, diffusivities.νₑ, Σ₁₃, u, v, w)

# At cfc
@inline ∂x_2ν_Σ₂₁(i, j, k, grid, closure, u, v, w, diffusivities) =
    2 * ∂x_caa(i, j, k, grid, ν_Σᵢⱼ_ffc, diffusivities.νₑ, Σ₂₁, u, v, w)

@inline ∂y_2ν_Σ₂₂(i, j, k, grid, closure, u, v, w, diffusivities) =
    2 * ∂y_afa(i, j, k, grid, ν_Σᵢⱼ_ccc, diffusivities.νₑ, Σ₂₂, u, v, w)

@inline ∂z_2ν_Σ₂₃(i, j, k, grid, closure, u, v, w, diffusivities) =
    2 * ∂z_aac(i, j, k, grid, ν_Σᵢⱼ_cff, diffusivities.νₑ, Σ₂₃, u, v, w)

# At ccf
@inline ∂x_2ν_Σ₃₁(i, j, k, grid, closure, u, v, w, diffusivities) =
    2 * ∂x_caa(i, j, k, grid, ν_Σᵢⱼ_fcf, diffusivities.νₑ, Σ₃₁, u, v, w)

@inline ∂y_2ν_Σ₃₂(i, j, k, grid, closure, u, v, w, diffusivities) =
    2 * ∂y_aca(i, j, k, grid, ν_Σᵢⱼ_cff, diffusivities.νₑ, Σ₃₂, u, v, w)

@inline ∂z_2ν_Σ₃₃(i, j, k, grid, closure, u, v, w, diffusivities) =
    2 * ∂z_aaf(i, j, k, grid, ν_Σᵢⱼ_ccc, diffusivities.νₑ, Σ₃₃, u, v, w)

#
# Without precomputed diffusivities
#

@inline ∂x_2ν_Σ₁₁(i, j, k, grid, closure, eos, grav, u, v, w, T, S) =
    2 * ∂x_faa(i, j, k, grid, ν_Σᵢⱼ, ν_ccc, Σ₁₁, closure, eos, grav, u, v, w, T, S)

@inline ∂y_2ν_Σ₁₂(i, j, k, grid, closure, eos, grav, u, v, w, T, S) =
    2 * ∂y_aca(i, j, k, grid, ν_Σᵢⱼ, ν_ffc, Σ₁₂, closure, eos, grav, u, v, w, T, S)

@inline ∂z_2ν_Σ₁₃(i, j, k, grid, closure, eos, grav, u, v, w, T, S) =
    2 * ∂z_aac(i, j, k, grid, ν_Σᵢⱼ, ν_fcf, Σ₁₃, closure, eos, grav, u, v, w, T, S)

# At cfc
@inline ∂x_2ν_Σ₂₁(i, j, k, grid, closure, eos, grav, u, v, w, T, S) =
    2 * ∂x_caa(i, j, k, grid, ν_Σᵢⱼ, ν_ffc, Σ₂₁, closure, eos, grav, u, v, w, T, S)

@inline ∂y_2ν_Σ₂₂(i, j, k, grid, closure, eos, grav, u, v, w, T, S) =
    2 * ∂y_afa(i, j, k, grid, ν_Σᵢⱼ, ν_ccc, Σ₂₂, closure, eos, grav, u, v, w, T, S)

@inline ∂z_2ν_Σ₂₃(i, j, k, grid, closure, eos, grav, u, v, w, T, S) =
    2 * ∂z_aac(i, j, k, grid, ν_Σᵢⱼ, ν_cff, Σ₂₃, closure, eos, grav, u, v, w, T, S)

# At ccf
@inline ∂x_2ν_Σ₃₁(i, j, k, grid, closure, eos, grav, u, v, w, T, S) =
    2 * ∂x_caa(i, j, k, grid, ν_Σᵢⱼ, ν_fcf, Σ₃₁, closure, eos, grav, u, v, w, T, S)

@inline ∂y_2ν_Σ₃₂(i, j, k, grid, closure, eos, grav, u, v, w, T, S) =
    2 * ∂y_aca(i, j, k, grid, ν_Σᵢⱼ, ν_cff, Σ₃₂, closure, eos, grav, u, v, w, T, S)

@inline ∂z_2ν_Σ₃₃(i, j, k, grid, closure, eos, grav, u, v, w, T, S) =
    2 * ∂z_aaf(i, j, k, grid, ν_Σᵢⱼ, ν_ccc, Σ₃₃, closure, eos, grav, u, v, w, T, S)

"""
    κ_∂x_c(i, j, k, grid, c, κ, closure, eos, g, u, v, w, T, S)

Return `κ ∂x c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂x_c(i, j, k, grid, c, κ, closure, args...)
    κ = ▶x_faa(i, j, k, grid, κ, closure, args...)
    ∂x_c = ∂x_faa(i, j, k, grid, c)
    return κ * ∂x_c
end

"""
    κ_∂y_c(i, j, k, grid, c, κ, closure, eos, g, u, v, w, T, S)

Return `κ ∂y c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂y_c(i, j, k, grid, c, κ, closure, args...)
    κ = ▶y_afa(i, j, k, grid, κ, closure, args...)
    ∂y_c = ∂y_afa(i, j, k, grid, c)
    return κ * ∂y_c
end

"""
    κ_∂z_c(i, j, k, grid, c, κ, closure, eos, g, u, v, w, T, S)

Return `κ ∂z c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂z_c(i, j, k, grid, c, κ, closure, args...)
    κ = ▶z_aaf(i, j, k, grid, κ, closure, args...)
    ∂z_c = ∂z_aaf(i, j, k, grid, c)
    return κ * ∂z_c
end

"""
    ∇_κ_∇_c(i, j, k, grid, c, closure, eos, g, u, v, w, T, S)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ c)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇c(i, j, k, grid, c, closure::IsotropicDiffusivity, eos, g, u, v, w, T, S) = (
      ∂x_caa(i, j, k, grid, κ_∂x_c, c, κ_ccc, closure, eos, g, u, v, w, T, S)
    + ∂y_aca(i, j, k, grid, κ_∂y_c, c, κ_ccc, closure, eos, g, u, v, w, T, S)
    + ∂z_aac(i, j, k, grid, κ_∂z_c, c, κ_ccc, closure, eos, g, u, v, w, T, S)
    )

"""
    ∇_κ_∇_c(i, j, k, grid, c, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ c)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇c(i, j, k, grid, c, closure::IsotropicDiffusivity, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_c, c, diffusivities.κₑ, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_c, c, diffusivities.κₑ, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_c, c, diffusivities.κₑ, closure)
)

"""
    ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, u, v, w, diffusivities)

Return the ``x``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₁₁) + ∂y(2 ν Σ₁₁) + ∂z(2 ν Σ₁₁)`

at the location `fcc`.
"""
@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, u, v, w, diffusivities) = (
      ∂x_2ν_Σ₁₁(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂y_2ν_Σ₁₂(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂z_2ν_Σ₁₃(i, j, k, grid, closure, u, v, w, diffusivities)
)

"""
    ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, u, v, w, diffusivities)

Return the ``y``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₂₁) + ∂y(2 ν Σ₂₂) + ∂z(2 ν Σ₂₂)`

at the location `ccf`.
"""
@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, u, v, w, diffusivities) = (
      ∂x_2ν_Σ₂₁(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂y_2ν_Σ₂₂(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂z_2ν_Σ₂₃(i, j, k, grid, closure, u, v, w, diffusivities)
)

"""
    ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, diffusivities)

Return the ``z``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₃₁) + ∂y(2 ν Σ₃₂) + ∂z(2 ν Σ₃₃)`

at the location `ccf`.
"""
@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::IsotropicDiffusivity, u, v, w, diffusivities) = (
      ∂x_2ν_Σ₃₁(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂y_2ν_Σ₃₂(i, j, k, grid, closure, u, v, w, diffusivities)
    + ∂z_2ν_Σ₃₃(i, j, k, grid, closure, u, v, w, diffusivities)
)
