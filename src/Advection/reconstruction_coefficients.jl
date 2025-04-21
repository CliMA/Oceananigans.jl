using Oceananigans.Grids: ξnodes, ηnodes, rnodes

# Generic reconstruction methods valid for all reconstruction schemes
# Unroll the functions to pass the coordinates in case of a stretched grid
""" same as [`symmetric_interpolate_xᶠᵃᵃ`](@ref) but on `Center`s instead of `Face`s """
@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, ψ, args...) = symmetric_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, args...)
""" same as [`symmetric_interpolate_yᵃᶠᵃ`](@ref) but on `Center`s instead of `Face`s """
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, ψ, args...) = symmetric_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, args...)
""" same as [`symmetric_interpolate_zᵃᵃᶠ`](@ref) but on `Center`s instead of `Face`s """
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, ψ, args...) = symmetric_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, args...)

""" same as [`biased_interpolate_xᶠᵃᵃ`](@ref) but on `Center`s instead of `Face`s """
@inline biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, bias, ψ, args...)  = biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, bias, ψ, args...)
""" same as [`biased_interpolate_yᵃᶠᵃ`](@ref) but on `Center`s instead of `Face`s """
@inline biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, bias, ψ, args...)  = biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, bias, ψ, args...)
""" same as [`biased_interpolate_zᵃᵃᶠ`](@ref) but on `Center`s instead of `Face`s """
@inline biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, bias, ψ, args...)  = biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, bias, ψ, args...)

struct FirstDerivative end
struct SecondDerivative end
struct Primitive end

num_prod(i, m, l, r, xr, xi, shift, op, order, args...)            = @inbounds prod(xr[i+shift] - xi[op(i, r-q+1)]  for q=0:order if (q != m && q != l))
num_prod(i, m, l, r, xr, xi, shift, op, order, ::FirstDerivative)  = @inbounds 2*xr[i+shift] - sum(xi[op(i, r-q+1)] for q=0:order if (q != m && q != l))
num_prod(i, m, l, r, xr, xi, shift, op, order, ::SecondDerivative) = 2

@inline function num_prod(i, m, l, r, xr, xi, shift, op, order, ::Primitive) 
    s = sum(xi[op(i, r-q+1)]  for q=0:order if (q != m && q != l))
    p = prod(xi[op(i, r-q+1)] for q=0:order if (q != m && q != l))

    return xr[i+shift]^3 / 3 - s * xr[i+shift]^2 / 2 + p * xr[i+shift]
end

"""
    stencil_coefficients(i, r, xr, xi; shift = 0, op = Base.:(-), order = 3, der = nothing)

Return coefficients for finite-volume polynomial reconstruction of order `order` at stencil `r`.

Positional Arguments
====================

- `xi`: the locations of the reconstructing value, i.e. either the center coordinate,
  for centered quantities or face coordinate for staggered
- `xr`: the opposite of the reconstruction location desired, i.e., if a recostruction at
  `Center`s is required xr is the face coordinate

On a uniform `grid`, the coefficients are independent of the `xr` and `xi` values.
"""
@inline function stencil_coefficients(FT, i, r, xr, xi; shift = 0, op = Base.:(-), order = 3, der = nothing)
    coeffs = zeros(BigFloat, order)
    @inbounds begin
        for j in 0:order-1
            for m in j+1:order
                numerator   = sum(num_prod(i, m, l, r, xr, xi, shift, op, order, der) for l=0:order if l != m)
                denominator = prod(xi[op(i, r-m+1)] - xi[op(i, r-l+1)] for l=0:order if l != m)
                coeffs[j+1] += numerator / denominator * (xi[op(i, r-j)] - xi[op(i, r-j+1)])
            end
        end
    end

    coeffs = FT.(coeffs)[1:end-1]

    return tuple(coeffs..., 1-sum(coeffs)) # Coefficients should sum to 1!
end

"""
    uniform_reconstruction_coefficients(FT, Val(bias), buffer)

Returns coefficients for finite volume reconstruction used in linear advection schemes (`Centered` and `UpwindBiased`). 
`FT` is the floating type (e.g. `Float32`, `Float64`), `bias` is either `:symmetric`, `:left`, or `:right`, 
and `buffer` is the buffer size which determines the order of the reconstruction.

examples:
```julia
julia> using Oceananigans.Advection: uniform_reconstruction_coefficients

julia> uniform_reconstruction_coefficients(Float64, Val(:symmetric), 1)
(0.5, 0.5)

julia> uniform_reconstruction_coefficients(Float32, Val(:left), 3)
(-0.05f0, 0.45f0, 0.78333336f0, -0.21666667f0, 0.033333335f0)

julia> uniform_reconstruction_coefficients(Float16, Val(:right), 4)
(Float16(-0.00714), Float16(0.0595), Float16(-0.2405), Float16(0.76), Float16(0.51), Float16(-0.09045), Float16(0.00952))
```
"""
uniform_reconstruction_coefficients(FT, ::Val{:symmetric}, buffer) = stencil_coefficients(FT, 50, buffer-1, collect(1:100), collect(1:100); order = 2buffer)
uniform_reconstruction_coefficients(FT, ::Val{:left}, buffer)      = buffer==1 ? (one(FT),) : stencil_coefficients(FT, 50, buffer-2, collect(1:100), collect(1:100); order = 2buffer-1)
uniform_reconstruction_coefficients(FT, ::Val{:right}, buffer)     = buffer==1 ? (one(FT),) : stencil_coefficients(FT, 50, buffer-1, collect(1:100), collect(1:100); order = 2buffer-1)

""" 
    calc_reconstruction_stencil(FT, buffer, shift, dir, func::Bool = false)

Stencils for reconstruction calculations (note that WENO has its own reconstruction stencils)

The first argument is the `buffer`, not the `order`! 
- `order = 2 * buffer` for Centered reconstruction
- `order = 2 * buffer - 1` for Upwind reconstruction
   
Examples
========

```jldoctest
julia> using Oceananigans.Advection: calc_reconstruction_stencil

julia> calc_reconstruction_stencil(Float32, 1, :right, :x)
:(+(1.0f0 * ψ[i + 0, j, k]))

julia> calc_reconstruction_stencil(Float64, 1, :left, :x)
:(+(1.0 * ψ[i + -1, j, k]))

julia> calc_reconstruction_stencil(Float64, 1, :symmetric, :y)
:(0.5 * ψ[i, j + -1, k] + 0.5 * ψ[i, j + 0, k])

julia> calc_reconstruction_stencil(Float32, 2, :symmetric, :x)
:(-0.083333254f0 * ψ[i + -2, j, k] + 0.5833333f0 * ψ[i + -1, j, k] + 0.5833333f0 * ψ[i + 0, j, k] + -0.083333336f0 * ψ[i + 1, j, k])

julia> calc_reconstruction_stencil(Float32, 3, :left, :x)
:(0.0333333f0 * ψ[i + -3, j, k] + -0.21666667f0 * ψ[i + -2, j, k] + 0.78333336f0 * ψ[i + -1, j, k] + 0.45f0 * ψ[i + 0, j, k] + -0.05f0 * ψ[i + 1, j, k])
```
"""
@inline function calc_reconstruction_stencil(FT, buffer, shift, dir, func::Bool = false)
    N = buffer * 2
    order = shift == :symmetric ? N : N - 1
    if shift != :symmetric
        N = N .- 1
    end
    rng = 1:N
    if shift == :right
        rng = rng .+ 1
    end
    stencil_full = Vector(undef, N)
    coeff = uniform_reconstruction_coefficients(FT, Val(shift), buffer)
    for (idx, n) in enumerate(rng)
        c = n - buffer - 1
        C = coeff[order - idx + 1]
        if func
            stencil_full[idx] = dir == :x ?
                                :($C * ψ(i + $c, j, k, grid, args...)) :
                                dir == :y ?
                                :($C * ψ(i, j + $c, k, grid, args...)) :
                                :($C * ψ(i, j, k + $c, grid, args...))
        else
            stencil_full[idx] =  dir == :x ? 
                                :($C * ψ[i + $c, j, k]) :
                                dir == :y ?
                                :($C * ψ[i, j + $c, k]) :
                                :($C * ψ[i, j, k + $c])
        end
    end
    return Expr(:call, :+, stencil_full...)
end