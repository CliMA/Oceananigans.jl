"""
    newton_div(inv_FT, a, b::FT)

Compute an approximation of `a / b` that uses `inv_FT` type to compute
`1/b`, and then performs a single Newton iteration to add a few more bits of precision
afterwards.
"""
@inline function newton_div(inv_FT, a, b::FT) where FT
    # Low precision division:
    b_low = convert(inv_FT, b)
    inv_b = Base.FastMath.inv_fast(b_low)

    # compute x = a / b using the low-precision approximation for 1/b
    x = a * convert(FT, inv_b)

    # Improve the approximation with a single Newton iteration: x += (a – x*b) / b
    x = fma(fma(x, -b, a), inv_b, x)

    return x
end

# Fallback for no precision lowering
@inline newton_div(::Type{FT}, a, b::FT) where FT = a * Base.FastMath.inv_fast(b)

# Note that the implementation may be overridden for some specific backends
# Refer to extension modules (e.g. `OceananigansCUDAExt`)
