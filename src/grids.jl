"""
    RegularCartesianGrid{T<:AbstractFloat, R<:AbstractRange} <: Grid

A Cartesian grid with regularly spaces cells and faces so that \$Δx\$, \$Δy\$,
and \$Δz\$ are constants. Fields are stored using floating-point values of type
`T`. The cell-centered and face-centered coordinate ranges are stored as ranges
of type `A`.
"""
struct RegularCartesianGrid{T<:AbstractFloat, R<:AbstractRange} <: AbstractGrid{T}
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
    length(N) == 3 || throw(ArgumentError("N=$N must be a tuple of length 3."))
    length(L) == 3 || throw(ArgumentError("L=$L must be a tuple of length 3."))

    all(isa.(N, Integer)) || throw(ArgumentError("N=$N should contain integers."))
    all(isa.(L, Number))  || throw(ArgumentError("L=$L should contain numbers."))

    all(N .>= 1) || throw(ArgumentError("N=$N must be nonzero and positive!"))
    all(L .> 0)  || throw(ArgumentError("L=$L must be nonzero and positive!"))

    Nx, Ny, Nz = N
    Lx, Ly, Lz = L

    # Right now we only support periodic horizontal boundary conditions and
    # usually use second-order advection schemes so halos of size Hx, Hy = 1 are
    # just what we need.
    Hx, Hy, Hz = 1, 1, 1

    Tx = Nx + 2*Hx
    Ty = Ny + 2*Hy
    Tz = Nz + 2*Hz

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

    RegularCartesianGrid{T, typeof(xC)}(Nx, Ny, Nz, Hx, Hy, Hz, Tx, Ty, Tz,
                                        Lx, Ly, Lz, Δx, Δy, Δz, Ax, Ay, Az, V,
                                        xC, yC, zC, xF, yF, zF)
end

# Constructor aliases.
RegularCartesianGrid(N, L) = RegularCartesianGrid(Float64, N, L)
RegularCartesianGrid(T=Float64; N, L) = RegularCartesianGrid(T, N, L)

size(g::RegularCartesianGrid) = (g.Nx, g.Ny, g.Nz)
eltype(g::RegularCartesianGrid{T}) where T = T

show(io::IO, g::RegularCartesianGrid) =
    print(io, "RegularCartesianGrid{$(eltype(g))}\n", 
              "  resolution (Nx, Ny, Nz) = ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz) = ", (g.Hx, g.Hy, g.Hz), '\n',
              "      domain (Lx, Ly, Lz) = ", (g.Lx, g.Ly, g.Lz), '\n',
              "grid spacing (Δx, Δy, Δz) = ", (g.Δx, g.Δy, g.Δz))
