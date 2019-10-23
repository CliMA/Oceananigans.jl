"""
    RegularCartesianGrid{FT<:AbstractFloat, R<:AbstractRange} <: AbstractGrid{FT}

A Cartesian grid with with constant grid spacings `Δx`, `Δy`, and `Δz` between cell centers
and cell faces.
"""
struct RegularCartesianGrid{FT<:AbstractFloat, R<:AbstractRange} <: AbstractGrid{FT}
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
    Lx::FT
    Ly::FT
    Lz::FT
    # Grid spacing [m].
    Δx::FT
    Δy::FT
    Δz::FT
    # Cell face areas [m²].
    Ax::FT
    Ay::FT
    Az::FT
    # Volume of a cell [m³].
    V::FT
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
```
julia> grid = RegularCartesianGrid(N=(32, 32, 32), L=(1, 1, 1))
RegularCartesianGrid{Float64}
  resolution (Nx, Ny, Nz) = (32, 32, 32)
   halo size (Hx, Hy, Hz) = (1, 1, 1)
      domain (Lx, Ly, Lz) = (1.0, 1.0, 1.0)
grid spacing (Δx, Δy, Δz) = (0.03125, 0.03125, 0.03125)
```
```
julia> grid = RegularCartesianGrid(Float32; N=(32, 32, 16), L=(8, 8, 2))
RegularCartesianGrid{Float32}
  resolution (Nx, Ny, Nz) = (32, 32, 16)
   halo size (Hx, Hy, Hz) = (1, 1, 1)
      domain (Lx, Ly, Lz) = (8.0f0, 8.0f0, 2.0f0)
grid spacing (Δx, Δy, Δz) = (0.25f0, 0.25f0, 0.125f0)
```
"""
function RegularCartesianGrid(FT=Float64; size, length=nothing, x=nothing, y=nothing, z=nothing)
    # Hack that allows us to use `size` and `length` as keyword arguments but then also
    # use the `size` and `length` functions.
    sz, len = size, length
    length = Base.length

    length(sz) == 3        || throw(ArgumentError("size=$size must be a tuple of length 3."))
    all(isa.(sz, Integer)) || throw(ArgumentError("size=$size should contain integers."))
    all(sz .>= 1)          || throw(ArgumentError("size=$size must be nonzero and positive!"))

    if !isnothing(len)
        length(len) == 3       || throw(ArgumentError("length=$len must be a tuple of length 3."))
        all(isa.(len, Number)) || throw(ArgumentError("length=$len should contain numbers."))
        all(len .>= 0)         || throw(ArgumentError("length=$len must be nonzero and positive!"))

        Lx, Ly, Lz = len
        x = (0, Lx)
        y = (0, Ly)
        z = (-Lz, 0)
    end

    function coord2xyz(c)
        c == 1 && return "x"
        c == 2 && return "y"
        c == 3 && return "z"
    end

    for (i, c) in enumerate((x, y, z))
        name = coord2xyz(i)
        length(c) == 2       || throw(ArgumentError("$name=$c must be a tuple of length 2."))
        all(isa.(c, Number)) || throw(ArgumentError("$name=$c should contain numbers."))
        c[2] >= c[1]         || throw(ArgumentError("$name=$c should be an increasing interval."))
    end

    x₁, x₂ = x[1], x[2]
    y₁, y₂ = y[1], y[2]
    z₁, z₂ = z[1], z[2]
    Nx, Ny, Nz = sz
    Lx, Ly, Lz = x₂-x₁, y₂-y₁, z₂-z₁

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

    xC = range(x₁ + Δx/2, x₂ - Δx/2; length=Nx)
    yC = range(y₁ + Δy/2, y₂ - Δy/2; length=Ny)
    zC = range(z₁ + Δz/2, z₂ - Δz/2; length=Nz)

    xF = range(x₁, x₂; length=Nx+1)
    yF = range(y₁, y₂; length=Ny+1)
    zF = range(z₁, z₂; length=Nz+1)

    RegularCartesianGrid{FT, typeof(xC)}(Nx, Ny, Nz, Hx, Hy, Hz, Tx, Ty, Tz,
                                         Lx, Ly, Lz, Δx, Δy, Δz, Ax, Ay, Az, V,
                                         xC, yC, zC, xF, yF, zF)
end

size(grid::RegularCartesianGrid)   = (grid.Nx, grid.Ny, grid.Nz)
length(grid::RegularCartesianGrid) = (grid.Lx, grid.Ly, grid.Lz)
eltype(grid::RegularCartesianGrid{FT}) where FT = FT

show(io::IO, g::RegularCartesianGrid) =
    print(io, "RegularCartesianGrid{$(eltype(g))}\n",
              "domain: x ∈ [$(g.xF[1]), $(g.xF[end])], y ∈ [$(g.yF[1]), $(g.yF[end])], z ∈ [$(g.zF[end]), $(g.zF[1])]", '\n',
              "  resolution (Nx, Ny, Nz) = ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz) = ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δx, Δy, Δz) = ", (g.Δx, g.Δy, g.Δz))
