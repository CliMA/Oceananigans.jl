"""
    poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Periodic)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with periodic boundary conditions along the dimension `dim` with `N` grid
points and domain extent `L`. Eigenvalues match the grid float type.
"""
function poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Periodic)
    inds = reshape(1:N, reshaped_size(N, dim)...)
    return convert.(eltype(grid), @. (2sin((inds - 1) * π / N) / (L / N))^2)
end

"""
    poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Bounded)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with staggered Neumann boundary conditions along the dimension `dim` with
`N` grid points and domain extent `L`. Eigenvalues match the grid float type.
"""
function poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Bounded)
    inds = reshape(1:N, reshaped_size(N, dim)...)
    return convert.(eltype(grid), @. (2sin((inds - 1) * π / 2N) / (L / N))^2)
end

"""
    poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Flat)

Return N-element array of `0.0` reshaped to three-dimensions.
This is also the first `poisson_eigenvalue` for `Bounded` and `Periodic`
directions. Eigenvalues match the grid float type.
"""
poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Flat) = convert.(eltype(grid), reshape(zeros(N), reshaped_size(N, dim)...))
