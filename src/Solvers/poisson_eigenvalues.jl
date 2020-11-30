reshaped_size(N, dim) = dim == 1 ? (N, 1, 1) :
                        dim == 2 ? (1, N, 1) :
                        dim == 3 ? (1, 1, N) : nothing

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

# For Flat dimensions
λx(grid::AbstractGrid{FT, <:Flat}, ::Nothing) where FT = reshape([zero(FT)], 1, 1, 1)
λy(grid::AbstractGrid{FT, TX, <:Flat}, ::Nothing) where {FT, TX} = reshape([zero(FT)], 1, 1, 1)
λz(grid::AbstractGrid{FT, TX, TY, <:Flat}, ::Nothing) where {FT, TX, TY} = reshape([zero(FT)], 1, 1, 1)
