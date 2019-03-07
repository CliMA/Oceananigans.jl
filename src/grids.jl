import Base: size, show

using Oceananigans: ModelMetadata

"""
    RegularCartesianGrid

A Cartesian grid with regularly spaces cells and faces so that \$Δx\$, \$Δy\$,
and \$Δz\$ are constants. Fields are stored using floating-point values of type
T.
"""
struct RegularCartesianGrid{T<:AbstractFloat,A} <: Grid
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
    xC::A
    yC::A
    zC::A
    # Range of grid coordinates at the faces of the cells. 
    # Note: there are Nx+1 faces in the x-dimension, Ny+1 in the y, and Nz+1 in the z.
    xF::A
    yF::A
    zF::A
end

"""
    RegularCartesianGrid(metadata::ModelMetadata, N, L)

Create a regular Cartesian grid with size \$N = (N_x, N_y, N_z)\$ and domain
size \$L = (L_x, L_y, L_z)\$ where fields are stored using floating-point values
of type T.

# Examples
```julia-repl
julia> g = RegularCartesianGrid((16, 16, 8), (2π, 2π, 2π))
```
"""
function RegularCartesianGrid(T, N, L)
    @assert length(N) == 3 && length(L) == 3 "N, L must have all three dimensions to specify which dimensions are used."
    @assert all(L .> 0) "Domain lengths must be nonzero and positive!"

    # Count the number of dimensions with 1 grid point, i.e. the number of flat
    # dimensions, and use it to determine the dimension of the model.
    num_flat_dims = count(i->(i==1), N)
    dim = 3 - num_flat_dims
    @assert 1 <= dim <= 3 "Only 1D, 2D, and 3D grids are supported right now."

    Nx, Ny, Nz = N
    Lx, Ly, Lz = L

    dim == 2 && @assert Nx == 1 || Ny == 1 || Nz == 1 "For 2D grids, Nx, Ny, or Nz must be 1."
    dim == 3 && @assert Nx != 1 && Ny != 1 && Nz != 1 "For 3D grid, cannot have dimensions of size 1."

    Δx = Lx / Nx
    Δy = Ly / Ny
    Δz = Lz / Nz

    Ax = Δy*Δz
    Ay = Δx*Δz
    Az = Δx*Δy

    V = Δx*Δy*Δz

    xC = Δx/2:Δx:Lx
    yC = Δy/2:Δy:Ly
    zC = -Δz/2:-Δz:-Lz

    xF = 0:Δx:Lx
    yF = 0:Δy:Ly
    zF = 0:-Δz:-Lz

    RegularCartesianGrid{T,typeof(xC)}(dim, Nx, Ny, Nz, Lx, Ly, Lz, Δx, Δy, Δz, Ax, Ay,
                                       Az, V, xC, yC, zC, xF, yF, zF)
end

RegularCartesianGrid(N, L) = RegularCartesianGrid(Float64, N, L)
RegularCartesianGrid(meta::ModelMetadata, N, L) = RegularCartesianGrid(meta.float_type, N, L)

size(g::RegularCartesianGrid) = (g.Nx, g.Ny, g.Nz)
Base.eltype(g::RegularCartesianGrid{T}) where T = T

show(io::IO, g::RegularCartesianGrid) =
    print(io, "$(g.dim)-dimensional ($(typeof(g.Lx))) regular Cartesian grid\n",
              "(Nx, Ny, Nz) = ", (g.Nx, g.Ny, g.Nz), '\n',
              "(Lx, Ly, Lz) = ", (g.Lx, g.Ly, g.Lz), '\n',
              "(Δx, Δy, Δz) = ", (g.Δx, g.Δy, g.Δz))
