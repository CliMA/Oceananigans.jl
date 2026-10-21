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
    stencil_coefficients(FT, i, r, xr, xi; shift = 0, op = Base.:(-), order = 3, der = nothing)

Return coefficients for finite-volume polynomial reconstruction of order `order` at stencil `r`.

Positional Arguments
====================

- `FT`: the floating-point type of the returned coefficients (e.g. `Float32` or `Float64`).
- `i`: the index around which the reconstruction stencil is centered.
- `r`: the stencil number.
- `xr`: the opposite of the reconstruction location desired, i.e., if a reconstruction at
  `Center`s is required xr is the face coordinate
- `xi`: the locations of the reconstructing value, i.e. either the center coordinate,
  for centered quantities or face coordinate for staggered

Keyword Arguments
=================

- `shift`: offset added to the index `i` when locating the reconstruction point `xr[i+shift]`. Default: `0`.
- `op`: operation combining `i` with the stencil offset to index `xi`/`xr` (e.g. `Base.:(-)` for a
  left-biased stencil). Default: `Base.:(-)`.
- `order`: order of the polynomial reconstruction, equal to the number of returned coefficients. Default: `3`.
- `der`: quantity the coefficients reconstruct — `nothing` for the function value, or `FirstDerivative()`,
  `SecondDerivative()`, or `Primitive()` (the integral). Default: `nothing`.

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
$(TYPEDSIGNATURES)

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
