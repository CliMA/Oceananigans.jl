"""
    RegularCartesianGrid{T<:AbstractFloat, R<:AbstractRange} <: AbstractGrid{T}

A Cartesian grid with with constant grid spacings `Δx`, `Δy`, and `Δz` between cell centers
and cell faces.
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
    RegularCartesianGrid([T=Float64]; N, L)

Creates a `RegularCartesianGrid` with `N = (Nx, Ny, Nz)` grid points and domain size
`L = (Lx, Ly, Lz)`, where constants are stored using floating point values of type `T`.

Additional properties
=====================
- `(xC, yC, zC)::AbstractRange`: (x, y, z) coordinates of cell centers
- `(xF, yF, zF)::AbstractRange`: (x, y, z) coordinates of cell faces
- `(Hx, Hy, Hz)::Int`: Halo size in the (x, y, z)-direction
- `(Tx, Ty, Tz)::Int`: "Total" grid size (interior + halo points) in the (x, y, z)-direction

Examples
========
julia> grid = RegularCartesianGrid(N=(32, 32, 32), L=(1, 1, 1))
RegularCartesianGrid{Float64}
  resolution (Nx, Ny, Nz) = (32, 32, 32)
   halo size (Hx, Hy, Hz) = (1, 1, 1)
      domain (Lx, Ly, Lz) = (1.0, 1.0, 1.0)
grid spacing (Δx, Δy, Δz) = (0.03125, 0.03125, 0.03125)

julia> grid = RegularCartesianGrid(Float32; N=(32, 32, 16), L=(8, 8, 2))
RegularCartesianGrid{Float32}
  resolution (Nx, Ny, Nz) = (32, 32, 16)
   halo size (Hx, Hy, Hz) = (1, 1, 1)
      domain (Lx, Ly, Lz) = (8.0f0, 8.0f0, 2.0f0)
grid spacing (Δx, Δy, Δz) = (0.25f0, 0.25f0, 0.125f0)
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

    xC = range(Δx/2, Lx-Δx/2; length=Nx)
    yC = range(Δy/2, Ly-Δy/2; length=Ny)
    zC = range(-Lz+Δz/2, -Δz/2; length=Nz)

    xF = range(0, Lx; length=Nx+1)
    yF = range(0, Ly; length=Ny+1)
    zF = range(-Lz, 0; length=Nz+1)

    RegularCartesianGrid{T, typeof(xC)}(Nx, Ny, Nz, Hx, Hy, Hz, Tx, Ty, Tz,
                                        Lx, Ly, Lz, Δx, Δy, Δz, Ax, Ay, Az, V,
                                        xC, yC, zC, xF, yF, zF)
end

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
