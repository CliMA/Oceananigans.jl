import Base: size, show

"""
    RegularCartesianGrid{T<:AbstractFloat, R<:AbstractRange} <: Grid

A Cartesian grid with regularly spaces cells and faces so that \$Δx\$, \$Δy\$,
and \$Δz\$ are constants. Fields are stored using floating-point values of type
`T`. The cell-centered and face-centered coordinate ranges are stored as ranges
of type `A`.
"""
struct RegularCartesianGrid{T<:AbstractFloat, R<:AbstractRange} <: Grid
    dim::Int
    # Number of grid points in (x,y,z).
    Nx::Int
    Ny::Int
    Nz::Int
    # Halo size in (x,y,z).
    Hx::Int
    Hy::Int
    Hz::Int
    # Total number of grid points (including halo regions).
    Tx::Int
    Ty::Int
    Tz::Int
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
    xC::R
    yC::R
    zC::R
    # Range of grid coordinates at the faces of the cells.
    # Note: there are Nx+1 faces in the x-dimension, Ny+1 in the y, and Nz+1 in the z.
    xF::R
    yF::R
    zF::R
end

"""
    RegularCartesianGrid(T, N, L)

Create a regular Cartesian grid with size \$N = (N_x, N_y, N_z)\$ and domain
size \$L = (L_x, L_y, L_z)\$ where fields are stored using floating-point values
of type `T`.

# Examples
```julia-repl
julia> g = RegularCartesianGrid((16, 16, 8), (2π, 2π, 2π))
```
"""
function RegularCartesianGrid(T, N, L)
    !(length(N) == 3) && throw(ArgumentError("N=$N must be a tuple of length 3."))
    !(length(L) == 3) && throw(ArgumentError("L=$L must be a tuple of length 3."))

    !all(isa.(N, Integer)) && throw(ArgumentError("N=$N should contain integers."))
    !all(isa.(L, Number))  && throw(ArgumentError("L=$L should contain numbers."))

    !all(N .>= 1) && throw(ArgumentError("N=$N must be nonzero and positive!"))
    !all(L .> 0)  && throw(ArgumentError("L=$L must be nonzero and positive!"))

    !(T in [Float32, Float64]) && throw(ArgumentError("T=$T but only Float32 and Float64 grids are supported."))

    # Count the number of dimensions with 1 grid point, i.e. the number of flat
    # dimensions, and use it to determine the dimension of the model.
    num_flat_dims = count(i->(i==1), N)
    dim = 3 - num_flat_dims
    !(1 <= dim <= 3) && throw(ArgumentError("N=$N has dimension $dim. Only 1D, 2D, and 3D grids are supported."))

    Nx, Ny, Nz = N
    Lx, Ly, Lz = L

    dim == 2 && !(Nx == 1 || Ny == 1 || Nz == 1) &&
        throw(ArgumentError("For 2D grids, Nx, Ny, or Nz must be 1."))

    dim == 3 && !(Nx != 1 && Ny != 1 && Nz != 1) &&
        throw(ArgumentError("For 3D grids, cannot have dimensions of size 1."))

    # Right now we only support periodic horizontal boundary conditions and
    # usually use second-order advection schemes so halos of size Hx, Hy = 1 are
    # just what we need.
    Hx, Hy, Hz = 1, 1, 0

    Tx = Nx + 2*Hx
    Ty = Ny + 2*Hy
    Tz = Nz + 2*Hz

    Lx = convert(T, Lx)
    Ly = convert(T, Ly)
    Lz = convert(T, Lz)

    Δx = convert(T, Lx / Nx)
    Δy = convert(T, Ly / Ny)
    Δz = convert(T, Lz / Nz)

    Ax = convert(T, Δy*Δz)
    Ay = convert(T, Δx*Δz)
    Az = convert(T, Δx*Δy)

    V = convert(T, Δx*Δy*Δz)

    xC = Δx/2:Δx:Lx
    yC = Δy/2:Δy:Ly
    zC = -Δz/2:-Δz:-Lz

    xF = 0:Δx:Lx
    yF = 0:Δy:Ly
    zF = 0:-Δz:-Lz

    # Make sure all the coordinate ranges have the same type.
    !all(typeof.([xC, yC, zC, xF, yF, zF]) .== typeof(xC)) &&
        throw(ArgumentError("At least one coordinate range type did not match."))

    RegularCartesianGrid{T, typeof(xC)}(dim, Nx, Ny, Nz, Hx, Hy, Hz, Tx, Ty, Tz,
                                        Lx, Ly, Lz, Δx, Δy, Δz, Ax, Ay, Az, V,
                                        xC, yC, zC, xF, yF, zF)
end

# Constructor aliases.
RegularCartesianGrid(N, L) = RegularCartesianGrid(Float64, N, L)

size(g::RegularCartesianGrid) = (g.Nx, g.Ny, g.Nz)
Base.eltype(g::RegularCartesianGrid{T}) where T = T

show(io::IO, g::RegularCartesianGrid) =
    print(io, "$(g.dim)-dimensional ($(typeof(g.Lx))) regular Cartesian grid\n",
              "(Nx, Ny, Nz) = ", (g.Nx, g.Ny, g.Nz), '\n',
              "(Lx, Ly, Lz) = ", (g.Lx, g.Ly, g.Lz), '\n',
              "(Δx, Δy, Δz) = ", (g.Δx, g.Δy, g.Δz))
