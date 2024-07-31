# Grids and architectures

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

Oceananigans simulates the dynamics of ocean-flavored fluids by solving differential equations that conserve momentum, mass, and energy on a mesh of finite volumes or "cells".
A "grid" encodes fundamental information about this mesh of finite volumes, including the domain geometry, the number of cells, and the machine architecture and number representation that is used to store discrete data on the finite volume mesh.

More specifically, grids specify 

* The domain geometry, which may be a line (one-dimensional), rectangle (two-dimensional), box (three-dimensional), or a sector of a thin spherical shells (two- or three-dimensional). Complex domains --- for example, domains with bathymetry or topography --- are represented by using a masking technique to "immerse" an irregular boundary within an "underlying grid". Where supported, dimensions may be indicated as
    - [`Periodic`](@ref), which means that the two ends of the dimension coincide, so that information leaving the left side of the domain re-enters on the right
    - [`Bounded`](@ref), which means that the boundaries are either impenetrable (solid walls), or "open" representing a specified external state.
    - [`Flat`](@ref), which means that the grid has 1 or 2 dimensions (3 - number of flat direction).
* The spatial resolution, which determines the distribution of sizes of the finite volume cells that divide the domain.
* The machine architecture, or whether data is stored on the CPU, GPU, or distributed across multiple devices or nodes.
* The representation of floating point numbers, which can be single-precision (`Float32`) or double precision (`Float64`).

## Supported grids

The underlying grids we currently support are:

1. [`RectilinearGrid`](@ref)s, which supports lines, rectangles and boxes, and
2. [`LatitudeLongitudeGrid`](@ref), which supports sectors of thin spherical shells whos cells are bounded by lines of constant latitude and longitude, and
3. [`OrthogonalSphericalShellGrid`](@ref), which supports sectors of thin spherical shells divided into orthogonal but otherwise arbitrary finite volumes.

Complex domains are represented with [`ImmersedBoundaryGrid`](@ref), which combines one of the above underlying grids with a type of immersed boundary. The immersed boundaries we support currently are

1. [`GridFittedBottom`](@ref), which fits a one- or two-dimensional bottom height to the underlying grid, so the active part of the domain is above the bottom height.
2. [`PartialCellBottom`](@ref), which is similar to [`GridFittedBottom`](@ref), except that the height of the bottommost cell is changed to conform to bottom height, limited to prevent the bottom cells from becoming too thin.
3. [`GridFittedBoundary`](@ref), which fits a three-dimensional mask to the grid.

### A first example with `RectilinearGrid`

One of the simplest grids we can create is a box (three-dimensional)
divided into cells of equal size. For this example we choose a topology that is horizontally-periodic
and vertically-bounded --- a common configuration for ocean simulations.

```jldoctest grids
architecture = CPU()

grid = RectilinearGrid(architecture,
                       topology = (Periodic, Periodic, Bounded),
                       size = (16, 8, 4),
                       x = (0, 64),
                       y = (0, 32),
                       z = (0, 8))

# output
16×8×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 64.0) regularly spaced with Δx=4.0
├── Periodic y ∈ [0.0, 32.0) regularly spaced with Δy=4.0
└── Bounded  z ∈ [0.0, 8.0]  regularly spaced with Δz=2.0
```

Let's walk through each of the arguments to `RectilinearGrid`, some of which are also shared with `LatitudeLongitudeGrid`.

#### The architecture

The first argument, `CPU()`, specifies the "architecture" of the simulation.
By writing `architecture = GPU()`, any fields constructed on `grid` will store their data on
an Nvidia [`GPU`](@ref), if one is available. By default, the grid will be constructed on
the [`CPU`](@ref) if this argument is omitted (as we do in the next example).
(TODO: also document [`Distributed`](@ref)).

#### The topology

The first keyword argument specifies the `topology` os the grid, which determines if the grid is
one-, two-, or three-dimensional (the current case), and additionally the nature of each dimension.
The `topology` of the grid is always a `Tuple` with three elements (a 3-`Tuple`).
For `RectilinearGrid`, the three elements correspond to ``(x, y, z)`` and indicate whether the respective direction is `Periodic`, `Bounded`, or `Flat`.
So `topology = (Periodic, Periodic, Bounded)` is periodic in ``x`` and ``y``, and bounded in ``z``.

#### The size

The `size` is a `Tuple` that specifes the number of grid points in each direction.
The number of tuple elements corresponds to the number of dimensions that are not `Flat`,
so for the first example `size` has three elements.

#### The dimensions `x`, `y`, and `z`

The last three keyword arguments specify the extent and location of the finite volume cells that divide up the
`x`, `y`, and `z` dimensions.
In the example, we used tuples, which specify equally-spaced cells.
For example, `x = (0, 64)` with `size = (16, 8, 4)` means that the `x`-dimension is divided into 16 cells,
where the first or leftmost cell interface is located at `x = 0` and the last or rightmost cell interface is
located at `x = 64`.
The width of each cell is `Δx=4.0`.

`RectilinearGrid` also supports dimensions that are "stretched", or which are divided into cells of varying width.
The next example illustrates how to specify cells of varying with using a vector of cell interfaces.

### A single column grid with stretched vertical interfaces

For our next example, we build a grid representing a "single column" in the z-direction with unevenly spaced cells,

```jldoctest grids
z_interfaces = [0, 4, 6, 7, 8]

grid = RectilinearGrid(size = 4,
                       z = z_interfaces,
                       topology = (Flat, Flat, Bounded))

# output
1×1×4 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── Flat x
├── Flat y
└── Bounded  z ∈ [0.0, 8.0] variably spaced with min(Δz)=1.0, max(Δz)=4.0
```

The `x` and `y` dimensions have been marked as `Flat`, which means that fields do not vary in those
directions. This also means that the kwargs which specify the `x` and `y` domains may be omitted, and that
the `size` is either a number (as in the example above) or a 1-`Tuple`.
Regarding the stretched cell interfaces specified by `z_interfaces`, notice that the number of
vertical cell interfaces is ``Nz + 1 = length(z_interfaces) = 5``, where ``Nz = 4`` is the number
of cells in the vertical.

#### A two-dimensional grid in ``x, y``

To build a two-dimensional, ``16 \times 8`` grid in ``x, y`` on the domain ``(0, 2π) \times (0, π)``,
we write

```jldoctest grids
grid = RectilinearGrid(size = (16, 8),
                              x = (0, 2π),
                              y = (0, π),
                              topology = (Periodic, Periodic, Flat))

# output
16×8×1 RectilinearGrid{Float64, Periodic, Periodic, Flat} on CPU with 3×3×0 halo
├── Periodic x ∈ [0.0, 6.28319)   regularly spaced with Δx=0.392699
├── Periodic y ∈ [0.0, 3.14159)   regularly spaced with Δy=0.392699
└── Flat z
```

Here we have omitted the `z` keyword argument, and `size` is a 2-`Tuple` rather than a
3-`Tuple` as in the previous examples.

#### Even more complicated example!

For a "channel" model, as the one we constructed above, one would probably like to have finer resolution near
the channel walls. We construct a grid that has non-regular spacing in the bounded dimensions, here ``y`` and ``z``
by prescribing functions for `y` and `z` keyword arguments.

For example, we can use the Chebychev nodes, which are more closely stacked near boundaries, to prescribe the
``y``- and ``z``-faces.

```jldoctest grids
Nx = Ny = 64
Nz = 32

Lx = Ly = 1e4
Lz = 1e3

chebychev_spaced_y_faces(j) = - Ly/2 * cos(π * (j - 1) / Ny);
chebychev_spaced_z_faces(k) = - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);

grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       topology = (Periodic, Bounded, Bounded),
                       x = (0, Lx),
                       y = chebychev_spaced_y_faces,
                       z = chebychev_spaced_z_faces)

# output
64×64×32 RectilinearGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 10000.0)    regularly spaced with Δx=156.25
├── Bounded  y ∈ [-5000.0, 5000.0] variably spaced with min(Δy)=6.02272, max(Δy)=245.338
└── Bounded  z ∈ [-1000.0, 0.0]    variably spaced with min(Δz)=2.40764, max(Δz)=49.0086
```

```@setup 1
using Oceananigans
using CairoMakie
CairoMakie.activate!(type = "svg")
Nx, Ny, Nz = 64, 64, 32
Lx, Ly, Lz = 1e4, 1e4, 1e3
chebychev_spaced_y_faces(j) = - Ly/2 * cos(π * (j - 1) / Ny);
chebychev_spaced_z_faces(k) = - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);
grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       topology = (Periodic, Bounded, Bounded),
                       x = (0, Lx),
                       y = chebychev_spaced_y_faces,
                       z = chebychev_spaced_z_faces)
```

We can easily visualize the spacings of ``y`` and ``z`` directions. We can use, e.g.,
[`ynodes`](@ref) and [`yspacings`](@ref) to extract the positions and spacings of the
nodes from the grid.

```@example 1
 yᶜ = ynodes(grid, Center())
Δyᶜ = yspacings(grid, Center())

 zᶜ = znodes(grid, Center())
Δzᶜ = zspacings(grid, Center())

using CairoMakie

fig = Figure(size=(800, 900))

ax1 = Axis(fig[1, 1]; xlabel = "y (m)", ylabel = "y-spacing (m)", limits = (nothing, (0, 250)))
lines!(ax1, yᶜ, Δyᶜ)
scatter!(ax1, yᶜ, Δyᶜ)

ax2 = Axis(fig[2, 1]; xlabel = "z-spacing (m)", ylabel = "z (m)", limits = ((0, 50), nothing))
lines!(ax2, zᶜ, Δzᶜ)
scatter!(ax2, zᶜ, Δzᶜ)

save("plot_stretched_grid.svg", fig); nothing #hide
```

![](plot_stretched_grid.svg)

## `LatitudeLongitudeGrid`

A simple latitude-longitude grid with `Float64` type can be constructed by

```jldoctest
julia> grid = LatitudeLongitudeGrid(size = (36, 34, 25),
                                    longitude = (-180, 180),
                                    latitude = (-85, 85),
                                    z = (-1000, 0))
36×34×25 LatitudeLongitudeGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo and with precomputed metrics
├── longitude: Periodic λ ∈ [-180.0, 180.0) regularly spaced with Δλ=10.0
├── latitude:  Bounded  φ ∈ [-85.0, 85.0]   regularly spaced with Δφ=5.0
└── z:         Bounded  z ∈ [-1000.0, 0.0]  regularly spaced with Δz=40.0
```

For more examples see [`RectilinearGrid`](@ref) and [`LatitudeLongitudeGrid`](@ref).

