import Base: size, length, eltype, show

using Oceananigans: AbstractGrid

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
    RegularCartesianGrid([FT=Float64]; size, length, x, y, z)

Creates a `RegularCartesianGrid` with `size = (Nx, Ny, Nz)` grid points.

The physical length of the domain can be specified via `x`, `y`, and `z` keyword arguments
indicating the left and right endpoints of each dimensions, e.g. `x=(-π, π)` or via
the `length` argument, e.g. `length=(Lx, Ly, Lz)` which specifies the length of each dimension
in which case 0 ≤ x ≤ Lx, 0 ≤ y ≤ Ly, and -Lz ≤ z ≤ 0.

Constants are stored using floating point values of type `FT`.

Grid properties
===============
- `(xC, yC, zC)::AbstractRange`: (x, y, z) coordinates of cell centers
- `(xF, yF, zF)::AbstractRange`: (x, y, z) coordinates of cell faces
- `(Hx, Hy, Hz)::Int`: Halo size in the (x, y, z)-direction
- `(Tx, Ty, Tz)::Int`: "Total" grid size (interior + halo points) in the (x, y, z)-direction

Examples
========
```julia
julia> grid = RegularCartesianGrid(size=(32, 32, 32), length=(1, 2, 3))
RegularCartesianGrid{Float64}
domain: x ∈ [0.0, 1.0], y ∈ [0.0, 2.0], z ∈ [0.0, -3.0]
  resolution (Nx, Ny, Nz) = (32, 32, 32)
   halo size (Hx, Hy, Hz) = (1, 1, 1)
grid spacing (Δx, Δy, Δz) = (0.03125, 0.0625, 0.09375)
```
```julia
julia> grid = RegularCartesianGrid(Float32; size=(32, 32, 16), x=(0, 8), y=(-10, 10), z=(-π, π))
RegularCartesianGrid{Float32}
domain: x ∈ [0.0, 8.0], y ∈ [-10.0, 10.0], z ∈ [3.141592653589793, -3.141592653589793]
  resolution (Nx, Ny, Nz) = (32, 32, 16)
   halo size (Hx, Hy, Hz) = (1, 1, 1)
grid spacing (Δx, Δy, Δz) = (0.25f0, 0.625f0, 0.3926991f0)
```
"""
function RegularCartesianGrid(FT=Float64; size, halo=(1, 1, 1),
                              length=nothing, x=nothing, y=nothing, z=nothing)

    # Hack that allows us to use `size` and `length` as keyword arguments but then also
    # use the `length` function.
    sz, len = size, length
    length = Base.length

    Lx, Ly, Lz, x, y, z = validate_grid_size_and_length(sz, len, x, y, z)

    Nx, Ny, Nz = sz
    Hx, Hy, Hz = halo

    Tx = Nx + 2Hx
    Ty = Ny + 2Hy
    Tz = Nz + 2Hz

    Δx = Lx / Nx
    Δy = Ly / Ny
    Δz = Lz / Nz

    Ax = Δy*Δz
    Ay = Δx*Δz
    Az = Δx*Δy

    V = Δx*Δy*Δz

    x₁, x₂ = x[1], x[2]
    y₁, y₂ = y[1], y[2]
    z₁, z₂ = z[1], z[2]

    xC = range(x₁ + Δx/2, x₂ - Δx/2; length=Nx)
    yC = range(y₁ + Δy/2, y₂ - Δy/2; length=Ny)
    zC = range(z₁ + Δz/2, z₂ - Δz/2; length=Nz)

    xF = range(x₁, x₂; length=Nx+1)
    yF = range(y₁, y₂; length=Ny+1)
    zF = range(z₁, z₂; length=Nz+1)

    RegularCartesianGrid{FT, typeof(xC)}(Nx, Ny, Nz, Hx, Hy, Hz, Tx, Ty, Tz,
                                         Lx, Ly, Lz, Δx, Δy, Δz, xC, yC, zC, xF, yF, zF)
end

size(grid::RegularCartesianGrid)   = (grid.Nx, grid.Ny, grid.Nz)
length(grid::RegularCartesianGrid) = (grid.Lx, grid.Ly, grid.Lz)
eltype(grid::RegularCartesianGrid{FT}) where FT = FT

short_show(grid::RegularCartesianGrid{T}) where T = "RegularCartesianGrid{$T}"

show_domain(grid) = string("x ∈ [", grid.xF[1], ", ", grid.xF[end], "], ",
                           "y ∈ [", grid.yF[1], ", ", grid.yF[end], "], ",
                           "z ∈ [", grid.zF[1], ", ", grid.zF[end], "]")

show(io::IO, g::RegularCartesianGrid) =
    print(io, "RegularCartesianGrid{$(eltype(g))}\n",
              "domain: x ∈ [$(g.xF[1]), $(g.xF[end])], y ∈ [$(g.yF[1]), $(g.yF[end])], z ∈ [$(g.zF[end]), $(g.zF[1])]", '\n',
              "  resolution (Nx, Ny, Nz) = ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz) = ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δx, Δy, Δz) = ", (g.Δx, g.Δy, g.Δz))
