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
    Δx::T
    Δy::T
    Δz::T
    # Cell face areas [m²].
    Ax::T
    Ay::T
    Az::T
    # Volume of a cell [m³].
    V::T
    # Range of coordinates at the centers of the cells.
    xCR
    yCR
    zCR
    # Array of coordinates at the centers of the cells.
    xCA
    yCA
    zCA
    # Range of grid coordinates at the faces of the cells. Note that there are
    # Nˣ+1 faces in the x̂-dimension, Nʸ+1 in the ŷ, and Nᶻ+1 in the ẑ.
    xFR
    yFR
    zFR
    # Array of grid coordinates at the faces of the cells.
    xFA
    yFA
    zFA
end

# example: g = RegularCartesianGrid((16, 16, 8), (2π, 2π, 2π))
function RegularCartesianGrid(N, L, T=Float64)
    dim = 3

    Nx, Ny, Nz = N
    Lx, Ly, Lz = L

    Δx = Lx / Nx
    Δy = Ly / Ny
    Δz = Lz / Nz

    Ax = Δy*Δz
    Ay = Δx*Δz
    Az = Δx*Δy

    V = Δx*Δy*Δz

    xCR = Δx/2:Δx:Lx
    yCR = Δy/2:Δy:Ly
    zCR = -Δz/2:-Δz:-Lz

    xCA = repeat(reshape(xCR, Nx, 1,  1),  1,  Ny, Nz)
    yCA = repeat(reshape(yCR, 1,  Ny, 1),  Nx, 1,  Nz)
    zCA = repeat(reshape(zCR, 1,  1,  Nz), Nx, Ny, 1)

    xFR = 0:Δx:Lx
    yFR = 0:Δy:Ly
    zFR = 0:-Δz:-Lz

    xFA = repeat(reshape(xFR, Nx+1, 1, 1), 1, Ny+1, Nz+1)
    yFA = repeat(reshape(yFR, 1, Ny+1, 1), Nx+1, 1, Nz+1)
    zFA = repeat(reshape(zFR, 1, 1, Nz+1), Nx+1, Ny+1, 1)

    RegularCartesianGrid{T}(dim, Nx, Ny, Nz, Lx, Ly, Lz, Δx, Δy, Δz, Ax, Ay, Az, V, xCR, yCR, zCR, xCA, yCA, zCA, xFR, yFR, zFR, xFA, yFA, zFA)
end
