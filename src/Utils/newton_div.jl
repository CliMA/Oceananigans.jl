function newton_div(FT, a, b)
    b = convert(FT, b)
    inv_b = Base.FastMath.inv_fast(b)
    x = a * inv_b
    x = fma(fma(x, -b, a), inv_b, x)  # single Newton iteration x += (a – x*b) / b
    return x
end