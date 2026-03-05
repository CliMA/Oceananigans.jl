"""
    poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Periodic)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with periodic boundary conditions along the dimension `dim` with `N` grid
points and domain extent `L`. `grid` is passed for dispatch and possible
future use for setting float type of the eigenvalues.
"""
function poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Periodic)
    inds = reshape(1:N, reshaped_size(N, dim)...)
    return @. (2sin((inds - 1) * π / N) / (L / N))^2
end

"""
    poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Bounded)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with staggered Neumann boundary conditions along the dimension `dim` with
`N` grid points and domain extent `L`. `grid` is passed for dispatch and
possible future use for setting float type of the eigenvalues.
"""
function poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Bounded)
    inds = reshape(1:N, reshaped_size(N, dim)...)
    return @. (2sin((inds - 1) * π / 2N) / (L / N))^2
end

"""
    poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Flat)

Return N-element array of `0.0` reshaped to three-dimensions.
This is also the first `poisson_eigenvalue` for `Bounded` and `Periodic` directions.
`grid` is passed for dispatch and possible future use for setting float type of the
eigenvalues.
"""
poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Flat) = reshape(zeros(N), reshaped_size(N, dim)...)
