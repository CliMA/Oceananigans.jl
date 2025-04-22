"""
    newton_div(a, b, inv_FT)

Compute an approximation of `a / b`` that uses `inv_FT` to compute
`1/b`, and then performs a single Newton iteration to add a few more bits of precision
afterwards.
"""
@inline function newton_div(inv_FT, a, b::FT) where FT
    # Low precision division:
    b = convert(inv_FT, b)
    inv_b = Base.FastMath.inv_fast(b)

    # compute x = a / b with low-precision approximation for 1/b
    x = a * convert(FT, inv_b)

    # Improve the approximaiton with a single Newton iteration: x += (a – x*b) / b
    x = fma(fma(x, -b, a), inv_b, x)

    return x
end

@inline newton_div(::Type{Float32}, a, b::Float32}) = a * Base.FastMath.inv_fast(b)