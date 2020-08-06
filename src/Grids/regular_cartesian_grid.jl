"""
    RegularCartesianGrid{FT, TX, TY, TZ, R} <: AbstractGrid{FT, TX, TY, TZ}

A Cartesian grid with with constant grid spacings `Δx`, `Δy`, and `Δz` between cell centers
and cell faces, elements of type `FT`, topology `{TX, TY, TZ}`, and coordinate ranges
of type `R`.
"""
struct RegularCartesianGrid{FT, TX, TY, TZ, R} <: AbstractGrid{FT, TX, TY, TZ}
    # Number of grid points in (x,y,z).
    Nx :: Int
    Ny :: Int
    Nz :: Int
    # Halo size in (x,y,z).
    Hx :: Int
    Hy :: Int
    Hz :: Int
    # Domain size [m].
    Lx :: FT
    Ly :: FT
    Lz :: FT
    # Grid spacing [m].
    Δx :: FT
    Δy :: FT
    Δz :: FT
    # Range of coordinates at the centers of the cells.
    xC :: R
    yC :: R
    zC :: R
    # Range of grid coordinates at the faces of the cells.
    xF :: R
    yF :: R
    zF :: R
end

"""
    RegularCartesianGrid([FT=Float64]; size,
                         extent = nothing, x = nothing, y = nothing, z = nothing,
                         topology = (Periodic, Periodic, Bounded), halo = (1, 1, 1))

Creates a `RegularCartesianGrid` with `size = (Nx, Ny, Nz)` grid points.

Keyword arguments
=================

- `size` (required): A 3-tuple `(Nx, Ny, Nz)` prescribing the number of grid points in `x, y, z`

- `extent`: A 3-tuple `(Lx, Ly, Lz)` prescribing the physical extent of the grid.
                     The origin is the oceanic default `(0, 0, -Lz)`.

- `x`, `y`, and `z`: Each of `x, y, z` are 2-tuples that specify the end points of the domain
                     in their respect directions.

*Note*: _Either_ `extent`, or all of `x`, `y`, and `z` must be specified.

- `topology`: A 3-tuple `(Tx, Ty, Tz)` specifying the topology of the domain.
              `Tx`, `Ty`, and `Tz` specify whether the `x`-, `y`-, and `z` directions are
              `Periodic`, `Bounded`, or `Flat`. In a `Flat` direction, derivatives are
              zero. The default is `(Periodic, Periodic, Bounded)`.

- `halo`: A 3-tuple of integers that specifies the size of the halo region of cells surrounding
          the physical interior in `x`, `y`, and `z`.

The physical extent of the domain can be specified via `x`, `y`, and `z` keyword arguments
indicating the left and right endpoints of each dimensions, e.g. `x=(-π, π)` or via
the `extent` argument, e.g. `extent=(Lx, Ly, Lz)` which specifies the extent of each dimension
in which case 0 ≤ x ≤ Lx, 0 ≤ y ≤ Ly, and -Lz ≤ z ≤ 0.

A grid topology may be specified via a tuple assigning one of `Periodic`, `Bounded, and `Flat`
to each dimension. By default, a horizontally periodic grid topology `(Periodic, Periodic, Flat)`
is assumed.

Constants are stored using floating point values of type `FT`. By default this is `Float64`.
Make sure to specify the desired `FT` if not using `Float64`.

Grid properties
===============

- `(Nx, Ny, Nz)::Int`: Number of physical points in the (x, y, z)-direction

- `(Hx, Hy, Hz)::Int`: Number of halo points in the (x, y, z)-direction

- `(Lx, Ly, Lz)::FT`: Physical extent of the grid in the (x, y, z)-direction

- `(Δx, Δy, Δz)::FT`: Cell width in the (x, y, z)-direction

- `(xC, yC, zC)`: (x, y, z) coordinates of cell centers.

- `(xF, yF, zF)`: (x, y, z) coordinates of cell faces.

Examples
========

```jldoctest
julia> using Oceananigans

julia> grid = RegularCartesianGrid(size=(32, 32, 32), extent=(1, 2, 3))
RegularCartesianGrid{Float64, Periodic, Periodic, Bounded}
                   domain: x ∈ [0.0, 1.0], y ∈ [0.0, 2.0], z ∈ [-3.0, 0.0]
                 topology: (Periodic, Periodic, Bounded)
  resolution (Nx, Ny, Nz): (32, 32, 32)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (0.03125, 0.0625, 0.09375)
```

```jldoctest
julia> using Oceananigans

julia> grid = RegularCartesianGrid(Float32; size=(32, 32, 16), x=(0, 8), y=(-10, 10), z=(-π, π))
RegularCartesianGrid{Float32, Periodic, Periodic, Bounded}
                   domain: x ∈ [0.0, 8.0], y ∈ [-10.0, 10.0], z ∈ [-3.1415927, 3.1415927]
                 topology: (Periodic, Periodic, Bounded)
  resolution (Nx, Ny, Nz): (32, 32, 16)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (0.25f0, 0.625f0, 0.3926991f0)
```
"""
function RegularCartesianGrid(FT=Float64;
                                  size,
                                     x = nothing, y = nothing, z = nothing,
                                extent = nothing,
                              topology = (Periodic, Periodic, Bounded),
                                  halo = (1, 1, 1),
                              )

    TX, TY, TZ = validate_topology(topology)
    Lx, Ly, Lz, x, y, z = validate_regular_grid_size_and_extent(FT, size, extent, halo, x, y, z)

    # Unpacking
    Nx, Ny, Nz = N = size
    Hx, Hy, Hz = H = halo
                 L = (Lx, Ly, Lz)
    Δx, Δy, Δz = Δ = L ./ N
                X₁ = (x[1], y[1], z[1])

    # Face-node limits in x, y, z
    xF₋, yF₋, zF₋ = XF₋ = @. X₁ - H * Δ
    xF₊, yF₊, zF₊ = XF₊ = @. XF₋ + total_extent(topology, halo, Δ, L)

    # Cell-node limits in x, y, z
    xC₋, yC₋, zC₋ = XC₋ = @. XF₋ + Δ / 2
    xC₊, yC₊, zC₊ = XC₊ = @. XC₋ + L + Δ * (2H - 1)

    TFx, TFy, TFz = total_length.(Face, topology, N, H)
    TCx, TCy, TCz = total_length.(Cell, topology, N, H)

    # Include halo points in coordinate arrays
    xF = range(xF₋, xF₊; length = TFx)
    yF = range(yF₋, yF₊; length = TFy)
    zF = range(zF₋, zF₊; length = TFz)

    xC = range(xC₋, xC₊; length = TCx)
    yC = range(yC₋, yC₊; length = TCy)
    zC = range(zC₋, zC₊; length = TCz)

    # Offset.
    xC = OffsetArray(xC, -Hx)
    yC = OffsetArray(yC, -Hy)
    zC = OffsetArray(zC, -Hz)

    xF = OffsetArray(xF, -Hx)
    yF = OffsetArray(yF, -Hy)
    zF = OffsetArray(zF, -Hz)

    return RegularCartesianGrid{FT, TX, TY, TZ, typeof(xC)}(
        Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, Δx, Δy, Δz, xC, yC, zC, xF, yF, zF)
end

short_show(grid::RegularCartesianGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    "RegularCartesianGrid{$FT, $TX, $TY, $TZ}(Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz))"

show_domain(grid) = string("x ∈ [", grid.xF[1], ", ", grid.xF[end], "], ",
                           "y ∈ [", grid.yF[1], ", ", grid.yF[end], "], ",
                           "z ∈ [", grid.zF[1], ", ", grid.zF[end], "]")

function show(io::IO, g::RegularCartesianGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    xₗ, xᵣ = x_domain(g)
    yₗ, yᵣ = y_domain(g)
    zₗ, zᵣ = z_domain(g)

    print(io, "RegularCartesianGrid{$FT, $TX, $TY, $TZ}\n",
              "                   domain: x ∈ [$xₗ, $xᵣ], y ∈ [$yₗ, $yᵣ], z ∈ [$zₗ, $zᵣ]\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "  resolution (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δx, Δy, Δz): ", (g.Δx, g.Δy, g.Δz))
end
