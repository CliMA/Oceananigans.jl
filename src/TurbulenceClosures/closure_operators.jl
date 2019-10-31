#####
##### Differential operators for regular grids
#####

@inline ∂x_caa(i, j, k, grid, u, args...) = @inbounds (u[i+1, j, k] - u[i, j, k]) / grid.Δx
@inline ∂x_faa(i, j, k, grid, c, args...) = @inbounds (c[i, j, k] - c[i-1, j, k]) / grid.Δx

@inline ∂y_aca(i, j, k, grid, v, args...) = @inbounds (v[i, j+1, k] - v[i, j, k]) / grid.Δy
@inline ∂y_afa(i, j, k, grid, c, args...) = @inbounds (c[i, j, k] - c[i, j-1, k]) / grid.Δy

@inline ∂z_aac(i, j, k, grid, w, args...) = @inbounds (w[i, j, k+1] - w[i, j, k]) / grid.Δz
@inline ∂z_aaf(i, j, k, grid, c, args...) = @inbounds (c[i, j, k] - c[i, j, k-1]) / grid.Δz

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

#####
##### Differential operators
#####

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
@inline ∂z_aaf(i, j, k, grid::AbstractGrid, F::TF, args...) where TF<:Function =
    (F(i, j, k, grid, args...) - F(i, j, k-1, grid, args...)) / grid.Δz


"""
    ∂z_aac(i, j, k, grid, F, args...)

Differentiate the function or callable object

    `F(i, j, k, grid, args...)`

located at `aaf` in `z`, across `aac`.
"""
@inline ∂z_aac(i, j, k, grid, F::TF, args...) where TF<:Function =
    (F(i, j, k+1, grid, args...) - F(i, j, k, grid, args...)) / grid.Δz

#####
##### Double differentiation
#####

# With arrays
@inline ∂x²_caa(i, j, k, grid, c::AbstractArray) = ∂x_caa(i, j, k, grid, ∂x_faa, c)
@inline ∂x²_faa(i, j, k, grid, u::AbstractArray) = ∂x_faa(i, j, k, grid, ∂x_caa, u)

@inline ∂y²_aca(i, j, k, grid, c::AbstractArray) = ∂y_aca(i, j, k, grid, ∂y_afa, c)
@inline ∂y²_afa(i, j, k, grid, v::AbstractArray) = ∂y_afa(i, j, k, grid, ∂y_aca, v)

@inline ∂z²_aac(i, j, k, grid, c::AbstractArray) = ∂z_aac(i, j, k, grid, ∂z_aaf, c)
@inline ∂z²_aaf(i, j, k, grid, w::AbstractArray) = ∂z_aaf(i, j, k, grid, ∂z_aac, w)

@inline ∇h²_cca(i, j, k, grid, c::AbstractArray) = ∂x²_caa(i, j, k, grid, c) + ∂y²_aca(i, j, k, grid, c)
@inline ∇h²_fca(i, j, k, grid, u::AbstractArray) = ∂x²_faa(i, j, k, grid, u) + ∂y²_aca(i, j, k, grid, u)
@inline ∇h²_cfa(i, j, k, grid, v::AbstractArray) = ∂x²_caa(i, j, k, grid, v) + ∂y²_afa(i, j, k, grid, v)

# With functions and arbitrary arguments
@inline ∂x²_caa(i, j, k, grid, F::FU, args...) where FU <: Function = ∂x_caa(i, j, k, grid, ∂x_faa, F, args...)
@inline ∂x²_faa(i, j, k, grid, F::FU, args...) where FU <: Function = ∂x_faa(i, j, k, grid, ∂x_caa, F, args...)

@inline ∂y²_aca(i, j, k, grid, F::FU, args...) where FU <: Function = ∂y_aca(i, j, k, grid, ∂y_afa, F, args...)
@inline ∂y²_afa(i, j, k, grid, F::FU, args...) where FU <: Function = ∂y_afa(i, j, k, grid, ∂y_aca, F, args...)

@inline ∂z²_aac(i, j, k, grid, F::FU, args...) where FU <: Function = ∂z_aac(i, j, k, grid, ∂z_aaf, F, args...)
@inline ∂z²_aaf(i, j, k, grid, F::FU, args...) where FU <: Function = ∂z_aaf(i, j, k, grid, ∂z_aac, F, args...)

@inline ∇h²_cca(i, j, k, grid, F::FU, args...) where FU <: Function = ∂x²_caa(i, j, k, grid, F, args...) + ∂y²_aca(i, j, k, grid, F, args...)
@inline ∇h²_fca(i, j, k, grid, F::FU, args...) where FU <: Function = ∂x²_faa(i, j, k, grid, F, args...) + ∂y²_aca(i, j, k, grid, F, args...)
@inline ∇h²_cfa(i, j, k, grid, F::FU, args...) where FU <: Function = ∂x²_caa(i, j, k, grid, F, args...) + ∂y²_afa(i, j, k, grid, F, args...)

#####
##### Fourth-order differentiation
#####

@inline ∂x⁴_caa(i, j, k, grid, c::AbstractArray) = ∂x²_caa(i, j, k, grid, ∂x²_caa, c)
@inline ∂x⁴_faa(i, j, k, grid, u::AbstractArray) = ∂x²_faa(i, j, k, grid, ∂x²_faa, u)

@inline ∂y⁴_aca(i, j, k, grid, c::AbstractArray) = ∂y²_aca(i, j, k, grid, ∂y²_aca, c)
@inline ∂y⁴_afa(i, j, k, grid, v::AbstractArray) = ∂y²_afa(i, j, k, grid, ∂y²_afa, v)

@inline ∂z⁴_aac(i, j, k, grid, c::AbstractArray) = ∂z²_aac(i, j, k, grid, ∂z²_aac, c)
@inline ∂z⁴_aaf(i, j, k, grid, w::AbstractArray) = ∂z²_aaf(i, j, k, grid, ∂z²_aaf, w)

@inline ∇h⁴_cca(i, j, k, grid, c::AbstractArray) = ∇h²_cca(i, j, k, grid, ∇h²_cca, c)
@inline ∇h⁴_fca(i, j, k, grid, c::AbstractArray) = ∇h²_fca(i, j, k, grid, ∇h²_fca, c)
@inline ∇h⁴_cfa(i, j, k, grid, c::AbstractArray) = ∇h²_cfa(i, j, k, grid, ∇h²_cfa, c)

 
#####
##### Interpolation operations for functions
#####

"""
    ▶x_faa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `caa` to `faa`."
"""
@inline ▶x_faa(i, j, k, grid::RegularCartesianGrid{T}, F::TF, args...) where {T, TF<:Function} =
    T(0.5) * (F(i, j, k, grid, args...) + F(i-1, j, k, grid, args...))

"""
    ▶x_caa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `faa` to `caa`."
"""
@inline ▶x_caa(i, j, k, grid::RegularCartesianGrid{T}, F::TF, args...) where {T, TF<:Function} =
    return T(0.5) * (F(i+1, j, k, grid, args...) + F(i, j, k, grid, args...))

"""
    ▶y_afa(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aca` to `afa`.
"""
@inline ▶y_afa(i, j, k, grid::RegularCartesianGrid{T}, F::TF, args...) where {T, TF<:Function} =
    return T(0.5) * (F(i, j, k, grid, args...) + F(i, j-1, k, grid, args...))

"""
    ▶y_aca(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `afa` to `aca`."
"""
@inline ▶y_aca(i, j, k, grid::RegularCartesianGrid{T}, F::TF, args...) where {T, TF<:Function} =
    T(0.5) * (F(i, j+1, k, grid, args...) + F(i, j, k, grid, args...))

"""
    ▶z_aaf(i, j, k, grid, F, args...)

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aac` to `aaf`.
"""
@inline ▶z_aaf(i, j, k, grid::RegularCartesianGrid{T}, F::TF, args...) where {T, TF<:Function} =
    T(0.5) * (F(i, j, k, grid, args...) + F(i, j, k-1, grid, args...))

"""
    ▶z_aac(i, j, k, grid::RegularCartesianGrid{T}, F, args...) where T

Interpolate the function or callable object

    `F(i, j, k, grid, args...)`

from `aaf` to `aac`.
"""
@inline ▶z_aac(i, j, k, grid::RegularCartesianGrid{T}, F::TF, args...) where {T, TF<:Function} =
    T(0.5) * (F(i, j, k+1, grid, args...) + F(i, j, k, grid, args...))

# Convenience operators for "interpolating constants"
@inline ▶x_faa(i, j, k, grid, a::Number) = a
@inline ▶x_caa(i, j, k, grid, a::Number) = a
@inline ▶y_afa(i, j, k, grid, a::Number) = a
@inline ▶y_aca(i, j, k, grid, a::Number) = a
@inline ▶z_aaf(i, j, k, grid, a::Number) = a
@inline ▶z_aac(i, j, k, grid, a::Number) = a

@inline ▶x_faa(i, j, k, grid::RegularCartesianGrid{FT}, c, args...) where FT =
    @inbounds FT(0.5) * (c[i, j, k] + c[i-1, j, k])

@inline ▶x_caa(i, j, k, grid::RegularCartesianGrid{FT}, u, args...) where FT =
    @inbounds FT(0.5) * (u[i, j, k] + u[i+1, j, k])

@inline ▶y_afa(i, j, k, grid::RegularCartesianGrid{FT}, c, args...) where FT = 
    @inbounds FT(0.5) * (c[i, j, k] + c[i, j-1, k])

@inline ▶y_aca(i, j, k, grid::RegularCartesianGrid{FT}, v, args...) where FT =
    @inbounds FT(0.5) * (v[i, j, k] + v[i, j+1, k])

@inline ▶z_aaf(i, j, k, grid::RegularCartesianGrid{FT}, c, args...) where FT =
    @inbounds FT(0.5) * (c[i, j, k] + c[i, j, k-1])

@inline ▶z_aac(i, j, k, grid::RegularCartesianGrid{FT}, w, args...) where FT =
    @inbounds FT(0.5) * (w[i, j, k] + w[i, j, k+1])

#####
##### Double interpolation: 12 operators
#####

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

# Interpolation in three-dimensions:
@inline ▶xyz_ccc(i, j, k, grid, F, args...) = ▶x_caa(i, j, k, grid, ▶y_aca, ▶z_aac, F, args...)
@inline ▶xyz_fff(i, j, k, grid, F, args...) = ▶x_faa(i, j, k, grid, ▶y_afa, ▶z_aaf, F, args...)

@inline ▶xyz_ccf(i, j, k, grid, F, args...) = ▶x_caa(i, j, k, grid, ▶y_aca, ▶z_aaf, F, args...)
@inline ▶xyz_cfc(i, j, k, grid, F, args...) = ▶x_caa(i, j, k, grid, ▶y_afa, ▶z_aac, F, args...)
@inline ▶xyz_fcc(i, j, k, grid, F, args...) = ▶x_faa(i, j, k, grid, ▶y_aca, ▶z_aac, F, args...)

@inline ▶xyz_cff(i, j, k, grid, F, args...) = ▶x_caa(i, j, k, grid, ▶y_afa, ▶z_aaf, F, args...)
@inline ▶xyz_fcf(i, j, k, grid, F, args...) = ▶x_faa(i, j, k, grid, ▶y_aca, ▶z_aaf, F, args...)
@inline ▶xyz_ffc(i, j, k, grid, F, args...) = ▶x_faa(i, j, k, grid, ▶y_afa, ▶z_aac, F, args...)


"""
    ν_σᶜᶜᶜ(i, j, k, grid, ν, σᶜᶜᶜ, u, v, w)

Multiply the array `ν` located at `ᶜᶜᶜ` by a function 

    `σᶜᶜᶜ(i, j, k, grid, u, v, w)`

at index `i, j, k` and location `ᶜᶜᶜ`.
"""
@inline ν_σᶜᶜᶜ(i, j, k, grid, ν::TN, σᶜᶜᶜ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ν[i, j, k] * σᶜᶜᶜ(i, j, k, grid, u, v, w)

"""
    ν_σᶠᶠᶜ(i, j, k, grid, ν, σᶠᶠᶜ, u, v, w)

Multiply the array `ν` located at `ᶜᶜᶜ` by a function 

    `σᶠᶠᶜ(i, j, k, grid, u, v, w)`

at index `i, j, k` and location `ᶠᶠᶜ`.
"""
@inline ν_σᶠᶠᶜ(i, j, k, grid, ν::TN, σᶠᶠᶜ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ▶xy_ffa(i, j, k, grid, ν) * σᶠᶠᶜ(i, j, k, grid, u, v, w)

# These functions are analogous to the two above, but for different locations:
@inline ν_σᶠᶜᶠ(i, j, k, grid, ν::TN, σᶠᶜᶠ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ▶xz_faf(i, j, k, grid, ν) * σᶠᶜᶠ(i, j, k, grid, u, v, w)

@inline ν_σᶜᶠᶠ(i, j, k, grid, ν::TN, σᶜᶠᶠ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ▶yz_aff(i, j, k, grid, ν) * σᶜᶠᶠ(i, j, k, grid, u, v, w)

#####
##### Stress divergences
#####

# At fcc
@inline ∂x_2ν_Σ₁₁(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂x_faa(i, j, k, grid, ν_σᶜᶜᶜ, diffusivities.νₑ, Σ₁₁, U.u, U.v, U.w)

@inline ∂y_2ν_Σ₁₂(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂y_aca(i, j, k, grid, ν_σᶠᶠᶜ, diffusivities.νₑ, Σ₁₂, U.u, U.v, U.w)

@inline ∂z_2ν_Σ₁₃(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂z_aac(i, j, k, grid, ν_σᶠᶜᶠ, diffusivities.νₑ, Σ₁₃, U.u, U.v, U.w)

# At cfc
@inline ∂x_2ν_Σ₂₁(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂x_caa(i, j, k, grid, ν_σᶠᶠᶜ, diffusivities.νₑ, Σ₂₁, U.u, U.v, U.w)

@inline ∂y_2ν_Σ₂₂(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂y_afa(i, j, k, grid, ν_σᶜᶜᶜ, diffusivities.νₑ, Σ₂₂, U.u, U.v, U.w)

@inline ∂z_2ν_Σ₂₃(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂z_aac(i, j, k, grid, ν_σᶜᶠᶠ, diffusivities.νₑ, Σ₂₃, U.u, U.v, U.w)

# At ccf
@inline ∂x_2ν_Σ₃₁(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂x_caa(i, j, k, grid, ν_σᶠᶜᶠ, diffusivities.νₑ, Σ₃₁, U.u, U.v, U.w)

@inline ∂y_2ν_Σ₃₂(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂y_aca(i, j, k, grid, ν_σᶜᶠᶠ, diffusivities.νₑ, Σ₃₂, U.u, U.v, U.w)

@inline ∂z_2ν_Σ₃₃(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂z_aaf(i, j, k, grid, ν_σᶜᶜᶜ, diffusivities.νₑ, Σ₃₃, U.u, U.v, U.w)

"""
    κ_∂x_c(i, j, k, grid, c, κ, closure, args...)

Return `κ ∂x c`, where `κ` is an array or function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂x_c(i, j, k, grid, κ, c, closure, args...)
    κ = ▶x_faa(i, j, k, grid, κ, closure, args...)
    ∂x_c = ∂x_faa(i, j, k, grid, c)
    return κ * ∂x_c
end

"""
    κ_∂y_c(i, j, k, grid, c, κ, closure, args...)

Return `κ ∂y c`, where `κ` is an array or function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂y_c(i, j, k, grid, κ, c, closure, args...)
    κ = ▶y_afa(i, j, k, grid, κ, closure, args...)
    ∂y_c = ∂y_afa(i, j, k, grid, c)
    return κ * ∂y_c
end

"""
    κ_∂z_c(i, j, k, grid, c, κ, closure, buoyancy, u, v, w, T, S)

Return `κ ∂z c`, where `κ` is an array or function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂z_c(i, j, k, grid, κ, c, closure, args...)
    κ = ▶z_aaf(i, j, k, grid, κ, closure, args...)
    ∂z_c = ∂z_aaf(i, j, k, grid, c)
    return κ * ∂z_c
end

"""
    ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, U, diffusivities)

Return the ``x``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₁₁) + ∂y(2 ν Σ₁₁) + ∂z(2 ν Σ₁₁)`

at the location `fcc`.
"""
@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::IsotropicViscosity, U, diffusivities) = (
      ∂x_2ν_Σ₁₁(i, j, k, grid, closure, U, diffusivities)
    + ∂y_2ν_Σ₁₂(i, j, k, grid, closure, U, diffusivities)
    + ∂z_2ν_Σ₁₃(i, j, k, grid, closure, U, diffusivities)
)

"""
    ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, U, diffusivities)

Return the ``y``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₂₁) + ∂y(2 ν Σ₂₂) + ∂z(2 ν Σ₂₂)`

at the location `ccf`.
"""
@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::IsotropicViscosity, U, diffusivities) = (
      ∂x_2ν_Σ₂₁(i, j, k, grid, closure, U, diffusivities)
    + ∂y_2ν_Σ₂₂(i, j, k, grid, closure, U, diffusivities)
    + ∂z_2ν_Σ₂₃(i, j, k, grid, closure, U, diffusivities)
)

"""
    ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, diffusivities)

Return the ``z``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₃₁) + ∂y(2 ν Σ₃₂) + ∂z(2 ν Σ₃₃)`

at the location `ccf`.
"""
@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::IsotropicViscosity, U, diffusivities) = (
      ∂x_2ν_Σ₃₁(i, j, k, grid, closure, U, diffusivities)
    + ∂y_2ν_Σ₃₂(i, j, k, grid, closure, U, diffusivities)
    + ∂z_2ν_Σ₃₃(i, j, k, grid, closure, U, diffusivities)
)
