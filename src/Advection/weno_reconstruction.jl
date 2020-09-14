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

#####
##### ENO reconstruction coefficients and WENO weights on a uniform grid
##### Equation (2.21) from the Shu (1998) lecture notes.
#####

"""
    UΠ(k, r, m, l)

Return the product in the numerator of equation (2.21) of Shu (1998) for the `l`th
term of the `m`th Lagrange basis polynomial of an ENO reconstruction scheme with
order `k` and left shift `r`.
"""
UΠ(k, r, l, m) = prod([r-q+1 for q in 0:k if q ∉ (m, l)])

"""
    U(k, r, m)

Return the numerator in equation (2.21) of Shu (1998) for the `m`th Lagrange basis
polynomial of an ENO reconstruction scheme with order `k` and left shift `r`.
"""
U(k, r, m) = sum([UΠ(k, r, m, l) for l in 0:k if l != m])

"""
    D(k, m)

Return the denominator in equation (2.21) of Shu (1998) for the `m`th Lagrange basis
polynomial of an ENO reconstruction scheme with order `k`.
"""
D(k, m) = prod([m - l for l in 0:k if l != m])

"""
    eno_coefficient(k, r)

Return the `j`th ENO coefficient used to reconstruct a value at the point x(i+½) with
order of accuracy `k` (stencil size) and left shift `r`.
"""
eno_coefficient(k, r, j) = sum([U(k, r, m)//D(k, m) for m in j+1:k])

"""
    eno_coefficients(k, r)

Return an array of ENO coefficients to reconstruct a value at the point x(i+½) with
order of accuracy `k` (stencil size) and left shift `r`. Note that when combined
these produce a WENO scheme of order 2k-1.
"""
eno_coefficients(k, r) = [eno_coefficient(k, r, j) for j in 0:k-1]

"""
    optimal_weno_weights(k)

Return the optimal weights that can be used to weigh ENO reconstruction schemes of
order `k` to produce a WENO scheme of order 2k-1.
"""
function optimal_weno_weights(k)
    C = zeros(Rational, 2k-1, k)
    b = eno_coefficients(2k-1, k-1)

    for n in 0:k-1
        C[n+1:n+k, n+1] .= eno_coefficients(k, k-1-n)
    end

    Γ = C \ b
    return rationalize.(Γ, tol=√eps(Float64))
end
