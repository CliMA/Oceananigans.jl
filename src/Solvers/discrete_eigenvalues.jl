using Oceananigans: PBC, NFBC

"""
    λi(grid, ::PBC)

Return an Nx×1×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the x-dimension on `grid`.
"""
function λi(grid, ::PBC)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    is = reshape(1:Nx, Nx, 1, 1)
    return @. (2sin((is-1)*π/Nx) / (Lx/Nx))^2
end

"""
    λj(grid, ::PBC)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the y-dimension on `grid`.
"""
function λj(grid, ::PBC)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    return @. (2sin((js-1)*π/Ny) / (Ly/Ny))^2
end

"""
    λj(grid, ::NFBC)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the y-dimension on `grid`.
"""
function λj(grid, ::NFBC)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    return @. (2sin((js-1)*π/(2Ny)) / (Ly/Ny))^2
end

"""
    λk(grid, ::NFBC)

Return an 1×1×Nz array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the z-dimension on `grid`.
"""
function λk(grid, ::NFBC)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    ks = reshape(1:Nz, 1, 1, Nz)
    return @. (2sin((ks-1)*π/(2Nz)) / (Lz/Nz))^2
end
