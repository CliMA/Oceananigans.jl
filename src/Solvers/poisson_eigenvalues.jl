"""
    poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Periodic)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with periodic boundary conditions along the dimension `dim` with `N` grid
points and domain extent `L`. Eigenvalues are returned as Float64 for all
grid float types as this was found to be more stable. `grid` is passed for
dispatch and possible future use for setting float type of eigenvalues.
"""
function poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Periodic)
    inds = reshape(1:N, reshaped_size(N, dim)...)
    return convert.(eltype(grid), @. (2sin((inds - 1) * π / N) / (L / N))^2)
end

"""
    poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Bounded)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with staggered Neumann boundary conditions along the dimension `dim` with
`N` grid points and domain extent `L`. Eigenvalues are returned as Float64
for all grid float types as this was found to be more stable. `grid` is
passed for dispatch and possible future use for setting float type of
eigenvalues.
"""
function poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Bounded)
    inds = reshape(1:N, reshaped_size(N, dim)...)
    return convert.(eltype(grid), @. (2sin((inds - 1) * π / 2N) / (L / N))^2)
end

"""
    poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Flat)

Return N-element array of `0.0` reshaped to three-dimensions.
This is also the first `poisson_eigenvalue` for `Bounded` and `Periodic`
directions. Eigenvalues are returned as Float64 for all grid float types
as this was found to be more stable. `grid` is passed for dispatch and
possible future use for setting float type of eigenvalues.
"""
poisson_eigenvalues(grid::AbstractGrid, N, L, dim, ::Flat) = convert.(eltype(grid), reshape(zeros(N), reshaped_size(N, dim)...))
