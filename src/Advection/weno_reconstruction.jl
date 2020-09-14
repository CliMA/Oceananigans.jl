"""
WENO reconstruction schemes following the lecture notes by Shu (1998) and
the review article by Shu (2009).

Shu (1998), "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory
    Schemes for Hyperbolic Conservation Laws", In: Advanced Numerical Approximation
    of Nonlinear Hyperbolic Equations, edited by Cockburn et al., pp. 325–432.
    DOI: https://doi.org/10.1007/BFb0096355

Shu (2009) "High Order Weighted Essentially Nonoscillatory Schemes for Convection
    Dominated Problems", SIAM Review 51(1), pp. 82–126.
    DOI: https://doi.org/10.1137/070679065
"""

# WENO weights on a uniform grid.
# Equation (2.21) from Shu (1998) lecture notes.

# Product in the numerator
Up(r, k, l, m) = prod([r-q+1 for q in 0:k if q ∉ [m, l]])

# Sum in the numerator
Us(r, k, m) = sum([Up(r, k, l, m) for l in 0:k if l != m])

# Denominator
D(k, m) = prod([m - l for l in 0:k if l != m])

# Individual coefficient
c_rj(k, r, j) = sum([Us(r, k, m)//D(k, m) for m in j+1:k])

# Array of coefficients
coefficients(k, r) = [c_rj(k, r, j) for j in 0:k-1]

# Optimal WENO reconstruction weights that reproduce the interpolant of order 2k-1.
function optimal_weights(k)
    C = zeros(Rational, 2k-1, k)
    b = coefficients(2k-1, k-1)

    for n in 0:k-1
        C[n+1:n+k, n+1] .= coefficients(k, k-1-n)
    end

    Γ = C \ b
    return rationalize.(Γ, tol=√eps(Float64))
end

using SymPy

x(j) = j

ℓ(ξ, j, k, r) = prod((ξ - x(m-r)) / (x(j-r) - x(m-r)) for m in 0:k if m != j)

L(ξ, k, r, ϕ) = sum(ℓ(ξ, j, k, r) * ϕ[j+1] for j in 0:k)

β(ξ, k, r) = sum(integrate(diff(L(ξ, k, r, ϕ), ξ, l)^2, (ξ, -1/2, 1/2)) for l in 1:k)
