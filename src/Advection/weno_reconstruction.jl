"""
WENO reconstruction schemes following the lecture notes by Shu (1998).

Shu (1998), "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory
    Schemes for Hyperbolic Conservation Laws", In: Advanced Numerical Approximation
    of Nonlinear Hyperbolic Equations, edited by Cockburn et al., pp. 325–432.
    DOI: https://doi.org/10.1007/BFb0096355
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
    return rationalize.(C \ b; tol=√eps(Float64))
end

function print_interpolant(k, r)
    cs = coefficients(k, r)

    ssign(n) = ifelse(n >= 0, "+", "-")
    ssubscript(n) = ifelse(n == 0, "i", "i" * ssign(n) * string(abs(n)))

    print("u(i+1//2) = ")
    for (j, c) in enumerate(cs)
        c_s = ssign(c) * " " * string(abs(c))
        ss_s = ssubscript(j-r-1)
        print(c_s * " u(" * ss_s * ") ")
    end
    print("\n")
end

function print_interpolants(k)
    for r in -1:k-1
        print_interpolant(k, r)
    end
    Γ = optimal_weights(k)
    println("Optimal weights γᵣ: $Γ")
end

# Recreating entries from Table 2.1 of Shu (1998) lecture notes.

println("WENO-5 [Compare with Table 2.1 of Shu (1998) and equation (2.15) from Shu (2009)]:")
print_interpolants(3)

println("\nWENO-7 [Compare with Table 2.1 of Shu (1998)]:")
print_interpolants(4)

println("\nWENO-9 [Compare with Table 2.1 of Shu (1998) and equation (2.14) of Shu (2009)]:")
print_interpolants(5)
