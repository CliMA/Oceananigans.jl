struct RegularCartesianGrid{T <: AbstractFloat} <: Grid
    dim::Int
    # Number of grid points in (x,y,z).
    Nx::Int
    Ny::Int
    Nz::Int
    # Domain size [m].
    Lx::T
    Ly::T
    Lz::T
    # Grid spacing [m].
    dx::T
    dy::T
    dz::T
    # Cell face areas [m²].
    Ax::T
    Ay::T
    Az::T
    # Volume of a cell [m³].
    V::T
end

# example: g = RegularCartesianGrid((16, 16, 8), (2π, 2π, 2π))
function RegularCartesianGrid(N, L, T=Float64)
    dim = 3

    Nx, Ny, Nz = N
    Lx, Ly, Lz = L

    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    Ax = dx*dz
    Ay = dx*dz
    Az = dx*dy

    V = dx*dy*dz

    RegularCartesianGrid{T}(dim, Nx, Ny, Nz, Lx, Ly, Lz, dx, dy, dz, Ax, Ay, Az, V)
end
