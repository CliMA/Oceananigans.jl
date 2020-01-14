import Base: size, length, eltype, show

using Oceananigans: AbstractGrid

"""
    VerticallyStretchedCartesianGrid{FT<:AbstractFloat, R<:AbstractRange, A<:AbstractArray} <: AbstractGrid{FT}

A Cartesian grid with with constant horizontal grid spacings `Δx` and `Δy`, and
non-uniform or stretched vertical grid spacing `Δz` between cell centers and cell faces.
"""
struct VerticallyStretchedCartesianGrid{FT<:AbstractFloat, R<:AbstractRange, A<:AbstractArray} <: AbstractGrid{FT}
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
        size, halo=(1, 1, 1), length=nothing, x=nothing, y=nothing, z=nothing, zF=nothing)

    # Hack that allows us to use `size` and `length` as keyword arguments but then also
    # use the `size` and `length` functions.
    sz, len = size, length
    length = Base.length

    Lx, Ly, Lz, x, y, z = validate_grid_size_and_length(sz, len, halo, x, y, z)

    Nx, Ny, Nz = sz
    Hx, Hy, Hz = halo

    Tx = Nx + 2*Hx
    Ty = Ny + 2*Hy
    Tz = Nz + 2*Hz

    Δx = convert(FT, Lx / Nx)
    Δy = convert(FT, Ly / Ny)

    x₁, x₂ = convert.(FT, [x[1], x[2]])
    y₁, y₂ = convert.(FT, [y[1], y[2]])
    z₁, z₂ = convert.(FT, [z[1], z[2]])

    xF = range(x₁, x₂; length=Nx+1)
    yF = range(y₁, y₂; length=Ny+1)

    xC = range(x₁ + Δx/2, x₂ - Δx/2; length=Nx)
    yC = range(y₁ + Δy/2, y₂ - Δy/2; length=Ny)

    zF, zC, ΔzF, ΔzC = validate_and_generate_variable_grid_spacing(FT, zF, Nz, z₁, z₂)

    VerticallyStretchedCartesianGrid{FT, typeof(xF), typeof(zF)}(Nx, Ny, Nz, Hx, Hy, Hz, Tx, Ty, Tz, Lx, Ly, Lz,
                                                                 Δx, Δy, ΔzF, ΔzC, xC, yC, zC, xF, yF, zF)
end
