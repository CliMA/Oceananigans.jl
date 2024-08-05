# Grids and architectures

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

Oceananigans simulates the dynamics of ocean-flavored fluids by solving differential equations that conserve momentum, mass, and energy on a mesh of finite volumes or "cells".
The first decision we make when setting up a simulation is: on what _grid_ are we going to run our simulation?
The "grid" encodes fundamental information: the shape of the domain we're simulating in, the way that domain is divided into a mesh of finite volumes, the machine architecture (CPU, GPU, lots of GPUs), and the precision of the numbers we would like to use to represent ocean-flavored fluid variables (double precision or single precision).

One of the simplest grids we can make divides a three-dimensional rectangular domain -- or "a box" --- into evenly-spaced cells.
To create such a grid on the CPU (the machine that we typically run julia on, basically), we write

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

This simple grid

* Uses the CPU. To make a grid on the GPU --- which means that computations on the grid will be conducted using a GPU connected to our CPU, if one is available --- we write `architecture = GPU()`.
* Has a domain that's "periodic" in ``x, y``, but bounded in ``z``. More on what that means, exactly, in a bit.
* Has `16` cells in `x`, `8` cells in `y`, and `4` cells in `z`. That means there are ``16 \times 8 \times 4 = 512`` cells in all.
* Has an `x` dimension that spans from `x=0`, to `x=64`. And `y` spans `y=0` to `y=32`, and `z` spans `z=0` to `z=8`.
* Has cells that are all the same size, dividing the ``16 \times 8 \times 4`` box into ``4 \times 4 \times 2`` cells. Note that length units are whatever is used to construct the grid, so it's up to the user to make sure that all inputs use consistent units.

Next we illustrate the construction of a grid on the _GPU_ (for this to work, a GPU has
to be available and `CUDA.jl` must be working --- for more about that, see the
[`CUDA.jl` documentation](https://cuda.juliagpu.org/stable/)).
The grid is two-dimensional in ``x, z`` (it's missing the `y`-dimension), and has variably-spaced
cell interfaces in the `z`-direction,

```@setup grids_gpu
using Oceananigans
```

```jldoctest grids_gpu
architecture = GPU()
z_faces = [0, 1, 3, 6, 10]

grid = RectilinearGrid(architecture,
                       topology = (Periodic, Flat, Bounded),
                       size = (10, 4),
                       x = (0, 20),
                       z = z_faces)

# output
10×1×4 RectilinearGrid{Float64, Periodic, Flat, Bounded} on GPU with 3×0×3 halo
├── Periodic x ∈ [0.0, 20.0)      regularly spaced with Δx=2.0
├── Flat y
└── Bounded  z ∈ [0.0, 10.0]      variably spaced with min(Δz)=1.0, max(Δz)=4.0
```

For the above grid, the ``y``-dimension is marked `Flat` by specifying `topology = (Periodic, Flat, Bounded)`.
This means that nothing varies in ``y``; derivatives in ``y`` are 0.
This also means that the keyword argument (or short, kwarg) that specifies the ``y``-domains may be omitted, and that
the `size` has only two elements rather than 3 as in the first example.
Notice that in the stretched cell interfaces specified by `z_interfaces`, the number of
vertical cell interfaces is `Nz + 1 = length(z_interfaces) = 5`, where `Nz = 4` is the number
of cells in the vertical.

## The nuts and bolts of building a grid

These examples illustrate how the definition of any grid requires

* The machine architecture, or whether data is stored on the CPU, GPU, or distributed across multiple devices or nodes.
* Information about the domain geometry. Domains can take a variety of shapes, including
    - lines (one-dimensional),
    - rectangles (two-dimensional),
    - boxes (three-dimensional),
    - sectors of a thin spherical shells (two- or three-dimensional).
    Irregular domains -- such as domains that include bathymetry or topography -- are represented by using a masking technique to "immerse" an irregular boundary within an "underlying" regular grid. Part of specifying the shape of the domain also requires specifying the nature of each dimension, which may be
    - [`Periodic`](@ref), which means that the dimension circles back onto itself: information leaving the left side of the domain re-enters on the right.
    - [`Bounded`](@ref), which means that the two sides of the dimension are either impenetrable (solid walls), or "open", representing a specified external state.
    - [`Flat`](@ref), which means nothing can vary in that dimension, reducing the overall dimensionality of the grid.
* Defining the number of cells that divide each dimension. The number of cells, with or without explicit specification of the cell interfaces, determines the spatial resolution of the grid.
* The representation of floating point numbers, which can be single-precision (`Float32`) or double precision (`Float64`).

Let's dive into each of these options in more detail.

### Specifying the machine architecture

The positional argument `CPU()` or `GPU()`, specifies the "architecture" of the simulation.
By writing `architecture = GPU()`, any fields constructed on `grid` will store their data on
an Nvidia [`GPU`](@ref), if one is available. By default, the grid will be constructed on
the [`CPU`](@ref) if this argument is omitted.
So, for example,

```jldoctest grids
grid     = RectilinearGrid(size=3, z=(0, 1), topology=(Flat, Flat, Bounded))
cpu_grid = RectilinearGrid(CPU(), size=3, z=(0, 1), topology=(Flat, Flat, Bounded))

grid == cpu_grid

# output
true
```

To use more than one CPU, we use the `Distributed` architecture,

```jldoctest grids
child_architecture = CPU()
architecture = Distributed(child_architecture)

# output
[ Info: MPI has not been initialized, so we are calling MPI.Init().
Distributed{CPU} across 1 rank:
├── local_rank: 0 of 0-0
└── local_index: [1, 1, 1]
```

which allows us to distributed computations across either CPUs or GPUs.
In this case, we didn't launch `julia` on multiple nodes using [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface),
so we're only "distributed" across 1 node.
For more, see [Distributed grids](@ref).

### Specifying the topology for each dimension

The keyword argument `topology` determines if the grid is
one-, two-, or three-dimensional (the current case), and additionally specifies the nature of each dimension.
`topology` is always a `Tuple` with three elements (a 3-`Tuple`).
For `RectilinearGrid`, the three elements correspond to ``(x, y, z)`` and indicate whether the respective direction is `Periodic`, `Bounded`, or `Flat`.
A few more examples are,

```julia
topology = (Periodic, Periodic, Periodic) # triply periodic
topology = (Periodic, Periodic, Bounded)  # periodic in x, y, bounded in z
topology = (Periodic, Bounded, Bounded)   # periodic in x, but bounded in y, z (a "channel")
topology = (Bounded, Bounded, Bounded)    # bounded in x, y, z (a closed box)
topology = (Periodic, Periodic, Flat)     # two-dimensional, doubly-periodic in x, y (a torus)
topology = (Periodic, Flat, Flat)         # one-dimensional, periodic in x (a line)
topology = (Flat, Flat, Bounded)          # one-dimensional and bounded in z (a single column)
```

### Specifying the size of the grid

The `size` is a `Tuple` that specifes the number of grid points in each direction.
The number of tuple elements corresponds to the number of dimensions that are not `Flat`,
so for the first example `size` has three elements.

#### The halo size

An additional keyword argument `halo` allows us to set the number of "halo cells" that surround the core "interior" grid.
In the first few examples, we did not provide `halo`, so that it took its default value of `halo = (3, 3, 3)`.
But we can change the halo size, for example,

```jldoctest grids
big_halo_grid = RectilinearGrid(topology = (Periodic, Periodic, Flat),
                                size = (32, 16),
                                halo = (7, 7),
                                x = (0, 2π),
                                y = (0, π))

# output
32×16×1 RectilinearGrid{Float64, Periodic, Periodic, Flat} on CPU with 7×7×0 halo
├── Periodic x ∈ [-6.90805e-17, 6.28319) regularly spaced with Δx=0.19635
├── Periodic y ∈ [-1.07194e-16, 3.14159) regularly spaced with Δy=0.19635
└── Flat z
```

Note that both `size` and `halo` are 2-`Tuple`s, rather than the 3-`Tuple` that would be required for a three-dimensional grid,
or the single number that would be used for a one-dimensional grid.

### The dimensions: `x, y, z` for `RectilinearGrid`, or `latitude, longitude, z` for `LatitudeLongitudeGrid`

The last three keyword arguments specify the extent and location of the finite volume cells that divide up the
three dimensions of the grid.
For `RectilinearGrid`, the dimensions are called `x`, `y`, and `z`, whereas for `LatitudeLongitudeGrid` the
dimensions are called `latitude`, `longitude`, and `z`.
The type of each keyword argument determines how the dimension is divided up:

* Tuples that specify only the end points indicate that the dimension should be divided into
  equally-spaced cells. For example, `x = (0, 64)` with `size = (16, 8, 4)` means that the
  `x`-dimension is divided into 16 cells, where the first or leftmost cell interface is located
  at `x = 0` and the last or rightmost cell interface is located at `x = 64`. The width of each cell is `Δx=4.0`.
* Vectors and functions alternatively give the location of each cell interface, and thereby may be used
  to build grids that are divided into cells of varying width.

## Types of grids

The types of grids that we currently support are:

1. [`RectilinearGrid`](@ref), which can be fashioned into lines, rectangles and boxes,
2. [`LatitudeLongitudeGrid`](@ref), which are sectors of thin spherical shells, with cells bounded by lines of constant latitude and longitude,
3. [`OrthogonalSphericalShellGrid`](@ref), which are sectors of thin spherical shells divided with mesh lines that intersect at right angles (thus, orthogonal) but otherwise arbitrary.

In general, `OrthogonalSphericalShellGrids` are constructed by a recipe or conformal map.
For example, a recipe that implements the ["tripolar" grid](https://www.sciencedirect.com/science/article/abs/pii/S0021999196901369)
is implemented in the package
[`OrthogonalSphericalShellGrids.jl`](https://github.com/CliMA/OrthogonalSphericalShellGrids.jl).

### Bathymetry, topography, and other irregularities

Irregular or "complex" domains are represented with [`ImmersedBoundaryGrid`](@ref), which combines one of the above underlying grids with a type of immersed boundary. The immersed boundaries we support currently are

1. [`GridFittedBottom`](@ref), which fits a one- or two-dimensional bottom height to the underlying grid, so the active part of the domain is above the bottom height.
2. [`PartialCellBottom`](@ref), which is similar to [`GridFittedBottom`](@ref), except that the height of the bottommost cell is changed to conform to bottom height, limited to prevent the bottom cells from becoming too thin.
3. [`GridFittedBoundary`](@ref), which fits a three-dimensional mask to the grid.

To build an `ImmersedBoundaryGrid`, we start by building one of the three underlying grids, and then embedding a boundary into that underlying grid.
We'll start start by walking through each of the arguments to `RectilinearGrid`, some of which are also shared with `LatitudeLongitudeGrid`.

## A complicated example: three-dimensional `RectilinearGrid` with variable spacing via functions

Next we build a grid that is both `Bounded` and stretched in both the `y` and `z` directions.
The purpose of the stretching is to increase grid resolution near the boundaries.
We'll do this by using functions to specify the keyword arguments `y` and `z`.

```jldoctest grids
Nx = Ny = 64
Nz = 32

Lx = Ly = 1e4
Lz = 1e3

# Note that j varies from 1 to Ny
chebychev_spaced_y_faces(j) = Ly * (1 - cos(π * (j - 1) / Ny)) / 2

# Note that k varies from 1 to Nz
chebychev_spaced_z_faces(k) = - Lz * (1 + cos(π * (k - 1) / Nz)) / 2

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

```@setup plot
using Oceananigans
using CairoMakie

set_theme!(Theme(fontsize=24))

Nx, Ny, Nz = 64, 64, 32
Lx, Ly, Lz = 1e4, 1e4, 1e3

chebychev_spaced_y_faces(j) = Ly * (1 - cos(π * (j - 1) / Ny)) / 2
chebychev_spaced_z_faces(k) = - Lz * (1 + cos(π * (k - 1) / Nz)) / 2

grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       topology = (Periodic, Bounded, Bounded),
                       x = (0, Lx),
                       y = chebychev_spaced_y_faces,
                       z = chebychev_spaced_z_faces)
```

We can easily visualize the spacings of ``y`` and ``z`` directions. We can use, e.g.,
[`ynodes`](@ref) and [`yspacings`](@ref) to extract the positions and spacings of the
nodes from the grid.

```@example plot
yc = ynodes(grid, Center())
zc = znodes(grid, Center())

yf = ynodes(grid, Face())
zf = znodes(grid, Face())

Δy = yspacings(grid, Center())
Δz = zspacings(grid, Center())

using CairoMakie
CairoMakie.activate!(type = "svg") # hide

fig = Figure(size=(1200, 1200))

axy = Axis(fig[1, 1], title="y-grid")
lines!(axy, [0, Ly], [0, 0], color=:gray)
scatter!(axy, yf, 0 * yf, marker=:vline, color=:gray, markersize=20)
scatter!(axy, yc, 0 * yc)
hidedecorations!(axy)
hidespines!(axy)

axΔy = Axis(fig[2, 1]; xlabel = "y (m)", ylabel = "y-spacing (m)")
scatter!(axΔy, yc, Δy)
hidespines!(axΔy, :t, :r) 

axz = Axis(fig[3, 1], title="z-grid")
lines!(axz, [-Lz, 0], [0, 0], color=:gray)
scatter!(axz, zf, 0 * zf, marker=:vline, color=:gray, markersize=20)
scatter!(axz, zc, 0 * zc)
hidedecorations!(axz)
hidespines!(axz)

axΔz = Axis(fig[4, 1]; xlabel = "z (m)", ylabel = "z-spacing (m)")
scatter!(axΔz, zc, Δz)
hidespines!(axΔz, :t, :r)

rowsize!(fig.layout, 1, Relative(0.1))
rowsize!(fig.layout, 3, Relative(0.1))

display(fig); save("plot_stretched_grid.svg", fig); nothing # hide
```

![](plot_stretched_grid.svg)

## `LatitudeLongitudeGrid` with constant spacing

To construct a simple latitude-longitude grid whose cells have a fixed width in latitude and longitude, we write

```jldoctest latlon
grid = LatitudeLongitudeGrid(size = (180, 10, 5),
                             longitude = (-180, 180),
                             latitude = (-60, 60),
                             z = (-1000, 0))

# output
36×34×25 LatitudeLongitudeGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo and with precomputed metrics
├── longitude: Periodic λ ∈ [-180.0, 180.0) regularly spaced with Δλ=10.0
├── latitude:  Bounded  φ ∈ [-85.0, 85.0]   regularly spaced with Δφ=5.0
└── z:         Bounded  z ∈ [-1000.0, 0.0]  regularly spaced with Δz=40.0
```

The only difference between `LatitudeLongitudeGrid` and `RectilinearGrid` are the names of the horizontal coordinates:
`LatitudeLongitudeGrid` has `longitude` and `latitude` where `RectilinearGrid` has `x` and `y`.
Note that if topology is not provided, then an attempt is made to infer it: if the `longitude` spans 360 degrees,
the default x-topology is `Periodic`, and `Bounded` if `longitude` spans less than 360 degrees.
For example,

```jldoctest latlon
grid = LatitudeLongitudeGrid(size = (60, 10, 5),
                             longitude = (0, 60),
                             latitude = (-60, 60),
                             z = (-1000, 0))

# output
36×34×25 LatitudeLongitudeGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo and with precomputed metrics
├── longitude: Periodic λ ∈ [-180.0, 180.0) regularly spaced with Δλ=10.0
├── latitude:  Bounded  φ ∈ [-85.0, 85.0]   regularly spaced with Δφ=5.0
└── z:         Bounded  z ∈ [-1000.0, 0.0]  regularly spaced with Δz=40.0
```

is `Bounded` by default.
Moreover, this can be overridden,

```jldoctest latlon
grid = LatitudeLongitudeGrid(size = (60, 10, 5),
                             topology = (Periodic, Bounded, Bounded),
                             longitude = (0, 60),
                             latitude = (-60, 60),
                             z = (-1000, 0))

# output
36×34×25 LatitudeLongitudeGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo and with precomputed metrics
├── longitude: Periodic λ ∈ [-180.0, 180.0) regularly spaced with Δλ=10.0
├── latitude:  Bounded  φ ∈ [-85.0, 85.0]   regularly spaced with Δφ=5.0
└── z:         Bounded  z ∈ [-1000.0, 0.0]  regularly spaced with Δz=40.0
```

Note that neither `latitude` nor `z` may be `Periodic` with `LatitudeLongitudeGrid`.

```@setup latlon_nodes
using Oceananigans
```

Unlike `RectilinearGrid`, which uses a Cartesian coordinate system,
the intrinsic coordinate system for `LatitudeLongitudeGrid` are the geographic coordinates
`(λ, φ, z)`, where `λ` is longitude, `φ` is latitude, and `z` is height.

Note: to type `λ` or `φ` at the REPL, write either `\lambda` (for `λ`) or `\varphi` (for `φ`) and then press `<TAB>`.

```@example latlon_nodes
grid = LatitudeLongitudeGrid(size = (1, 44),
                             longitude = (0, 1),   
                             latitude = (0, 88),
                             topology = (Bounded, Bounded, Flat))

φ = φnodes(grid, Center())
Δx = xspacings(grid, Center(), Center())

using CairoMakie
CairoMakie.activate!(type = "svg") # hide

fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1], xlabel="Zonal spacing on 2 degree grid (km)", ylabel="Latitude (degrees)")
scatter!(ax, Δx ./ 1e3, φ)

display(fig); save("plot_lat_lon_spacings.svg", fig); nothing # hide
```

![](plot_lat_lon_spacings.svg)

## `LatitudeLongitudeGrid` with variable spacing

The syntax for building a grid with variably spaced cells is the same as for `RectilinearGrid`.
In our next example, we use a function to build a Mercator grid with a spacing of 2 degrees at
the equator,

```jldoctest latlon_nodes
# Mercator scale factor
scale_factor(φ) = 1 / cosd(φ)

# Compute cell interfaces with Mercator spacing
m = 2 # spacing at the equator in degrees
function latitude_faces(j)
    if j == 1 # equator
        return 0
    else # crudely estimate the location of the jth face 
        φ₋ = latitude_faces(j-1)
        φ′ = φ₋ + m * scale_factor(φ₋) / 2
        return φ₋ + m * scale_factor(φ′)
    end
end    

Lx = 360
Nx = Int(Lx / m)
Ny = findfirst(latitude_faces.(1:Nx) .> 90) - 2

grid = LatitudeLongitudeGrid(size = (Nx, Ny),
                             longitude = (0, Lx),
                             latitude = latitude_faces,
                             topology = (Bounded, Bounded, Flat))

# output
180×28×1 LatitudeLongitudeGrid{Float64, Bounded, Bounded, Flat} on CPU with 3×3×0 halo and with precomputed metrics
├── longitude: Bounded  λ ∈ [0.0, 360.0]     regularly spaced with Δλ=2.0
├── latitude:  Bounded  φ ∈ [0.0, 77.2679]   variably spaced with min(Δφ)=2.0003, max(Δφ)=6.95319
└── z:         Flat z
```

We've also illustrated the construction of a grid that is `Flat` in the vertical direction.
Now let's plot the metrics for this grid,

```@setup plot
# Mercator scale factor
scale_factor(φ) = 1 / cosd(φ)

# Compute cell interfaces with Mercator spacing
m = 2 # spacing at the equator in degrees
function latitude_faces(j)
    if j == 1 # equator
        return 0
    else # crudely estimate the location of the jth face 
        φ₋ = latitude_faces(j-1)
        φ′ = φ₋ + m * scale_factor(φ₋) / 2
        return φ₋ + m * scale_factor(φ′)
    end
end    

Lx = 360
Nx = Int(Lx / m)
Ny = findfirst(latitude_faces.(1:Nx) .> 90) - 2

grid = LatitudeLongitudeGrid(size = (Nx, Ny),
                             longitude = (0, Lx),
                             latitude = latitude_faces,
                             topology = (Bounded, Bounded, Flat))
```

```@example plot
φ = φnodes(grid, Center())
Δx = xspacings(grid, Center(), Center(), with_halos=true)[1:Ny]
Δy = yspacings(grid, Center())[1:Ny]

using CairoMakie
CairoMakie.activate!(type = "svg") # hide

fig = Figure(size=(800, 400))
axx = Axis(fig[1, 1], xlabel="Zonal spacing (km)", ylabel="Latitude (degrees)")
scatter!(axx, Δx ./ 1e3, φ)

axy = Axis(fig[1, 2], xlabel="Meridional spacing (km)")
scatter!(axy, Δy ./ 1e3, φ)

hidespines!(axx, :t, :r)
hidespines!(axy, :t, :l, :r)
hideydecorations!(axy, grid=false)

display(fig); save("plot_lat_lon_mercator.svg", fig); nothing # hide
```

![](plot_lat_lon_mercator.svg)

## Single-precision `RectilinearGrid`

To build a grid whose fields are represented with single-precision floating point values,
we specify the `float_type`, in addition to the (optional) `architecture`,

```jldoctest grids
architecture = CPU()
float_type = Float32

grid = RectilinearGrid(architecture, float_type,
                       topology = (Periodic, Periodic, Bounded),
                       size = (16, 8, 4),
                       x = (0, 64),
                       y = (0, 32),
                       z = (0, 8))

# output
16×8×4 RectilinearGrid{Float32, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 64.0) regularly spaced with Δx=4.0
├── Periodic y ∈ [0.0, 32.0) regularly spaced with Δy=4.0
└── Bounded  z ∈ [0.0, 8.0]  regularly spaced with Δz=2.0
```

Single precision should be used with care.
Users interested in performing single-precision simulations should subject their work
to extensive testing and validation.

For more examples see [`RectilinearGrid`](@ref) and [`LatitudeLongitudeGrid`](@ref).

## Distributed grids

To run the next examples, we have to launch julia with MPI and things are a bit more involved.
For best results, create [a new environment](https://pkgdocs.julialang.org/v1/environments/)
in an empty folder, and then add both `Oceananigans.jl` and `MPI.jl` to the new environment.
Pasting this snipped into the terminal should do the trick:

```bash
mkdir DistributedExamples
cd DistributedExamples
touch Project.toml
julia --project -e 'using Pkg; Pkg.add("Oceananigans"); Pkg.add("MPI")'
```

Next, copy-paste the following code into a script called `distributed_example.jl`:

```julia
using Oceananigans
using MPI
MPI.Init()

architecture = Distributed(CPU())
on_rank_0 = MPI.Comm_rank(MPI.COMM_WORLD) == 0

if on_rank_0
    @show architecture
end
```

Next, run the script with

```julia
mpiexec -n 2 julia --project distributed_example.jl
```

This should produce

```
$ mpiexec -n 2 julia --project distributed_example.jl
architecture = Distributed{CPU} across 2 = 2×1×1 ranks:
├── local_rank: 0 of 0-1
├── local_index: [1, 1, 1]
└── connectivity: east=1 west=1
```

That's what it looks like to build a [`Distributed`](@ref) architecture.
Notice we chose to display only if we're on rank 0 --- because otherwise, all the ranks print
to the terminal at once, talking over each other, and things get messy. Also, we used the
"default communicator" `MPI.COMM_WORLD` to determine whether we were on rank 0. This works
because `Distributed` uses `communicator = MPI.COMM_WORLD` by default (and this should be
changed only with great intention). For more about `Distributed`, check out the docstring,

```@docs
Distributed()
```

Next, let's try to build a distributed grid. Copy-paste this code into a new file
called `distributed_grid_example.jl`,

```julia
using Oceananigans
using MPI
MPI.Init()

architecture = Distributed(CPU())
on_rank_0 = MPI.Comm_rank(MPI.COMM_WORLD) == 0

grid = RectilinearGrid(architecture,
                       size = (48, 48, 16),
                       x = (0, 64),
                       y = (0, 64),
                       z = (0, 16),
                       topology = (Periodic, Periodic, Bounded))

if on_rank_0
    @show grid
end
```

and then run

```bash
mpirun -n 2 julia --project distributed_grid_example.jl
```

If everything is set up correctly (🤞), then you should see

```
$ mpiexec -n 2 julia --project distributed_grid_example.jl
grid = 24×48×16 RectilinearGrid{Float64, FullyConnected, Periodic, Bounded} on Distributed{CPU} with 3×3×3 halo
├── FullyConnected x ∈ [0.0, 32.0) regularly spaced with Δx=1.33333
├── Periodic y ∈ [0.0, 64.0)       regularly spaced with Δy=1.33333
└── Bounded  z ∈ [0.0, 16.0]       regularly spaced with Δz=1.0
```

Now we're getting somewhere. Let's note a few things:

* We built the grid with `size = (48, 48, 16)`, but ended up with a `24×48×16` grid. Why's that?
  Well, `(48, 48, 16)` is the size of the _global_ grid, or in other words, the grid that we would get
  if we stitched together all the grids from each rank. Here we have two ranks, so the _local_ grid
  has half of the grid points of the total grid, which is by default partitioned in `x` --- yielding
  a local size of `(24, 48, 16)`.

* The global grid has topology `(Periodic, Periodic, Bounded)`, but the local grids have the
  topology `(FullyConnected, Periodic, Bounded)`. That means that each local grid, which represents
  half of the global grid and is partitioned in `x`, is not `Periodic` in `x`. Instead, the west
  and east sides of each local grid (left and right in the `x`-direction) are "connected" to another rank.

To drive these points home, let's run the same script, but using 3 processors instead of 2:

```
$ mpiexec -n 3 julia --project distributed_grid_example.jl
grid = 16×48×16 RectilinearGrid{Float64, FullyConnected, Periodic, Bounded} on Distributed{CPU} with 3×3×3 halo
├── FullyConnected x ∈ [0.0, 21.3333) regularly spaced with Δx=1.33333
├── Periodic y ∈ [0.0, 64.0)          regularly spaced with Δy=1.33333
└── Bounded  z ∈ [0.0, 16.0]          regularly spaced with Δz=1.0
```

Now we have three local grids, each with size `(16, 48, 16)`.

More topics not yet covered in this tutorial:
* Specifying an explicit partition for a distributed grid
* Distributing a grid across GPUs

