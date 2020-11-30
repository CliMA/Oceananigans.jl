"""
    λx(grid, ::Periodic)

Return an Nx×1×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the x-dimension on `grid`.
"""
function λx(grid, ::Periodic)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    is = reshape(1:Nx, Nx, 1, 1)
    return @. (2sin((is - 1) * π / Nx) / (Lx / Nx))^2
end

"""
    λx(grid, ::Bounded)

Return an Nx×1×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the x-dimension on `grid`.
"""
function λx(grid, ::Bounded)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    is = reshape(1:Nx, Nx, 1, 1)
    return @. (2sin((is - 1) * π / 2Nx) / (Lx / Nx))^2
end

"""
    λy(grid, ::Periodic)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the y-dimension on `grid`.
"""
function λy(grid, ::Periodic)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    return @. (2sin((js - 1) * π / Ny) / (Ly / Ny))^2
end

"""
    λy(grid, ::Bounded)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the y-dimension on `grid`.
"""
function λy(grid, ::Bounded)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    return @. (2sin((js - 1) * π / 2Ny) / (Ly / Ny))^2
end

"""
    λz(grid, ::Periodic)

Return an 1×1×Nz array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the z-dimension on `grid`.
"""
function λz(grid, ::Periodic)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    ks = reshape(1:Nz, 1, 1, Nz)
    return @. (2sin((ks - 1) * π / Nz) / (Lz / Nz))^2
end

"""
    λz(grid, ::Bounded)

Return an 1×1×Nz array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the z-dimension on `grid`.
"""
function λz(grid, ::Bounded)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    ks = reshape(1:Nz, 1, 1, Nz)
    return @. (2sin((ks - 1) * π / 2Nz) / (Lz / Nz))^2
end

# For Flat dimensions
λx(grid::AbstractGrid{FT, <:Flat}, ::Nothing) where FT = reshape([zero(FT)], 1, 1, 1)
λy(grid::AbstractGrid{FT, TX, <:Flat}, ::Nothing) where {FT, TX} = reshape([zero(FT)], 1, 1, 1)
λz(grid::AbstractGrid{FT, TX, TY, <:Flat}, ::Nothing) where {FT, TX, TY} = reshape([zero(FT)], 1, 1, 1)
