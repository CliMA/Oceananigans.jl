import Base: size, length, eltype, show

using Oceananigans: AbstractGrid

"""
    VerticallyStretchedCartesianGrid{FT<:AbstractFloat, R<:AbstractRange} <: AbstractGrid{FT}

A Cartesian grid with with constant horizontal grid spacings `Δx` and `Δy`, and
non-uniform or stretched vertical grid spacing `Δz` between cell centers and cell faces.
"""
struct VerticallyStretchedCartesianGrid{FT, R, A} <: AbstractGrid{FT}
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
    ΔzF::A
    ΔzC::A
    # Range of coordinates at the centers of the cells.
    xC::R
    yC::R
    zC::A
    # Range of grid coordinates at the faces of the cells.
    # Note: there are Nx+1 faces in the x-dimension, Ny+1 in the y, and Nz+1 in the z.
    xF::R
    yF::R
    zF::A
end

function VerticallyStretchedCartesianGrid(FT=Float64, arch=CPU();
        size, length=nothing, x=nothing, y=nothing, z=nothing,
        zF=nothing, ΔzF=nothing, zC=nothing, ΔzC=nothing)

    # Hack that allows us to use `size` and `length` as keyword arguments but then also
    # use the `size` and `length` functions.
    sz, len = size, length
    length = Base.length

    validate_grid_size_and_length(sz, len, x, y, z)

    if !isnothing(len)
        Lx, Ly, Lz = len
        x = (0, Lx)
        y = (0, Ly)
        z = (-Lz, 0)
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

    xF = range(x₁, x₂; length=Nx+1)
    yF = range(y₁, y₂; length=Ny+1)

    xC = range(x₁ + Δx/2, x₂ - Δx/2; length=Nx)
    yC = range(y₁ + Δy/2, y₂ - Δy/2; length=Ny)

    validate_variable_grid_spacing(zF, ΔzF, zC, ΔzC, z₁, z₂)
    zF, zC = generate_vertical_grid_spacing(zF, ΔzF, zC, ΔzC)

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
