"""
    poisson_eigenvalues(N, L, dim, ::Periodic)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with periodic boundary conditions along the dimension `dim` with `N` grid
points and domain extent `L`.
"""
function poisson_eigenvalues(N, L, dim, ::Periodic)
    inds = reshape(1:N, reshaped_size(N, dim)...)
    return @. (2sin((inds - 1) * π / N) / (L / N))^2
end

"""
    poisson_eigenvalues(N, L, dim, ::Bounded)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with staggered Neumann boundary conditions along the dimension `dim` with
`N` grid points and domain extent `L`.
"""
function poisson_eigenvalues(N, L, dim, ::Bounded)
    inds = reshape(1:N, reshaped_size(N, dim)...)
    return @. (2sin((inds - 1) * π / 2N) / (L / N))^2
end

"""
    poisson_eigenvalues(N, L, dim, ::Flat)

No need to do any transforms along `Flat` dimensions so just return zeros
since they will be added to the eigenvalues for other dimensions.
"""
poisson_eigenvalues(N, L, dim, ::Flat) = zeros(1, 1, 1)
