"""
Abstract supertype grouping configuration options for `newton_div`.
Use one of the concrete subtypes as the first argument to `newton_div`
to select the implementation.
"""
abstract type NewtonDivConfig end

"""
Configuration selecting regular, full-precision division.
"""
struct NoNewtonDiv <: NewtonDivConfig end

"""
Configuration selecting approximate division via one Newton iteration.
The reciprocal `1/b` is first evaluated in a lower-precision type `FT`
to obtain a fast initial guess, then refined with a single Newton step
in the original precision.
"""
struct NewtonDivWithConversion{FT} <: NewtonDivConfig end

"""
Configuration selecting a backend-optimized implementation of approximate division.
The actual algorithm may differ across CPU and different GPU backends.
"""
struct BackendOptimizedNewtonDiv <: NewtonDivConfig end

"""
    newton_div(::Type{T}, a, b)

Compute an approximate division `a/b` using a method specified by selector type `T`.
"""
function newton_div end

@inline function newton_div(::Type{NewtonDivWithConversion{inv_FT}}, a, b::FT) where {inv_FT, FT}
    # Low precision division:
    b_low = convert(inv_FT, b)
    inv_b = Base.FastMath.inv_fast(b_low)

    # compute x = a / b using the low-precision approximation for 1/b
    x = a * convert(FT, inv_b)

    # Improve the approximation with a single Newton iteration: x += (a – x*b) / b
    x = fma(fma(x, -b, a), inv_b, x)

    return x
end

# Case of matching precisions
@inline newton_div(::Type{NewtonDivWithConversion{FT}}, a, b::FT) where FT = a * Base.FastMath.inv_fast(b)

# Exact division if requested
@inline newton_div(::Type{NoNewtonDiv}, a, b) = a / b
