#####
##### Lagrange basis functions for linear/bilinear/trilinear interpolation
#####

# Trilinear Lagrange polynomials (also used for 1D and 2D with η=0, ζ=0)
@inline ϕ₁(ξ, η, ζ) = (1 - ξ) * (1 - η) * (1 - ζ)
@inline ϕ₂(ξ, η, ζ) = (1 - ξ) * (1 - η) *      ζ
@inline ϕ₃(ξ, η, ζ) = (1 - ξ) *      η  * (1 - ζ)
@inline ϕ₄(ξ, η, ζ) = (1 - ξ) *      η  *      ζ
@inline ϕ₅(ξ, η, ζ) =      ξ  * (1 - η) * (1 - ζ)
@inline ϕ₆(ξ, η, ζ) =      ξ  * (1 - η) *      ζ
@inline ϕ₇(ξ, η, ζ) =      ξ  *      η  * (1 - ζ)
@inline ϕ₈(ξ, η, ζ) =      ξ  *      η  *      ζ

#####
##### Interpolator tuple helper
#####

"""
    interpolator(fractional_idx)

Return an "interpolator tuple" from the fractional index `fractional_idx`,
defined as the 3-tuple `(i⁻, i⁺, ξ)` where:

- `i⁻` is the index to the left (floor of fractional_idx)
- `i⁺` is the index to the right (i⁻ + 1)
- `ξ` is the fractional distance from `i⁻`, such that `ξ ∈ [0, 1)`

This function is used for linear interpolation in lookup tables and fields.

Note: Uses `Base.unsafe_trunc` instead of `trunc` for GPU compatibility.
See https://github.com/CliMA/Oceananigans.jl/issues/828
"""
@inline function interpolator(fractional_idx)
    # For why we use Base.unsafe_trunc instead of trunc see:
    # https://github.com/CliMA/Oceananigans.jl/issues/828
    # https://github.com/CliMA/Oceananigans.jl/pull/997
    i⁻ = Base.unsafe_trunc(Int, fractional_idx)
    i⁺ = i⁻ + 1
    ξ = mod(fractional_idx, 1)
    return (i⁻, i⁺, ξ)
end

@inline interpolator(::Nothing) = (1, 1, 0)

#####
##### Generic interpolation functions
#####

"""
    _interpolate(data, ix, iy, iz, in...)

Perform trilinear interpolation on 3D (or higher-dimensional) `data` using
interpolator tuples `ix`, `iy`, `iz` of the form `(i⁻, i⁺, ξ)`.

Additional indices `in...` are passed through for higher-dimensional arrays.
"""
@inline function _interpolate(data, ix, iy, iz, in...)
    i⁻, i⁺, ξ = ix
    j⁻, j⁺, η = iy
    k⁻, k⁺, ζ = iz

    return @inbounds ϕ₁(ξ, η, ζ) * getindex(data, i⁻, j⁻, k⁻, in...) +
                     ϕ₂(ξ, η, ζ) * getindex(data, i⁻, j⁻, k⁺, in...) +
                     ϕ₃(ξ, η, ζ) * getindex(data, i⁻, j⁺, k⁻, in...) +
                     ϕ₄(ξ, η, ζ) * getindex(data, i⁻, j⁺, k⁺, in...) +
                     ϕ₅(ξ, η, ζ) * getindex(data, i⁺, j⁻, k⁻, in...) +
                     ϕ₆(ξ, η, ζ) * getindex(data, i⁺, j⁻, k⁺, in...) +
                     ϕ₇(ξ, η, ζ) * getindex(data, i⁺, j⁺, k⁻, in...) +
                     ϕ₈(ξ, η, ζ) * getindex(data, i⁺, j⁺, k⁺, in...)
end

"""
    _interpolate(data, ix, iy)

Perform bilinear interpolation on 2D `data` using interpolator tuples `ix`, `iy`.
"""
@inline function _interpolate(data, ix, iy)
    i⁻, i⁺, ξ = ix
    j⁻, j⁺, η = iy

    return @inbounds (1 - ξ) * (1 - η) * data[i⁻, j⁻] +
                          ξ  * (1 - η) * data[i⁺, j⁻] +
                     (1 - ξ) *      η  * data[i⁻, j⁺] +
                          ξ  *      η  * data[i⁺, j⁺]
end

"""
    _interpolate(data, ix)

Perform linear interpolation on 1D `data` using interpolator tuple `ix`.
"""
@inline function _interpolate(data, ix)
    i⁻, i⁺, ξ = ix
    return @inbounds (1 - ξ) * data[i⁻] + ξ * data[i⁺]
end
