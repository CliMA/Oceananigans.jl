"""
    newton_div(a, b, FT=Float32)

Compute an approximation of `a / b`` that uses Float32 to compute
`1/b`, and then performs a single Newton iteration to add a few more bits of precision
afterwards.
"""
function newton_div(a, b, FT=Float32)
    # Low precision division:
    b = convert(FT, b)
    inv_b = Base.FastMath.inv_fast(b)

    # compute x = a / b with low-precision approximation for 1/b
    x = a * inv_b 

    # Improve the approximaiton with a single Newton iteration: x += (a – x*b) / b
    x = fma(fma(x, -b, a), inv_b, x)

    return x
end