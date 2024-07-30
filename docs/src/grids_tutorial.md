# Grids and architectures

The grids currently supported are:
- `RectilinearGrid`s with either constant or variable grid spacings and
- `LatitudeLongitudeGrid` on the sphere.

## `RectilinearGrid`

A `RectilinearGrid` is constructed by specifying

1. the `size` of the grid (a `Tuple` specifying the number of grid points in each direction);

2. the `topology` of the grid as a 3-`Tuple` (a tuple of 3 elements) for ``(x, y, z)`` that indicates whether each direction is
    * `Periodic`, which means that stuff traveling off the left side of the grid enters on the right side,
    * `Bounded`, which may either be impenetrable or "open"
    * `Flat`, which means that the grid has 1 or 2 dimensions (3 - number of flat direction).

3. Keyword arguments `x`, `y`, `z` which define the extent and grid spacings in every non-`Flat` direction.
   These keyword arguments can be either 
   * A 2-`Tuple` that define the _end points_ of the given direction, or
   * A vector or function of the direction index that specifies the locations of cell interfaces. 

### Three-dimensional grid with regular spacing in all directions

For example, a regular rectilinear grid with ``N_x \times N_y \times N_z = 16 \times 8 \times 4`` grid points
in a domain spanning `(0, 0, 0)` to `(64, 32, 8)` is constructed by writing

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest grids
julia> grid = RectilinearGrid(size = (16, 8, 4),
                              x = (0, 64),
                              y = (0, 32),
                              z = (0, 8),
                              topology = (Periodic, Periodic, Bounded))
32×64×256 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 64.0)  regularly spaced with Δx=4.0
├── Periodic y ∈ [0.0, 32.0)  regularly spaced with Δy=4.0
└── Bounded  z ∈ [0.0, 8.0]   regularly spaced with Δz=2.0
```

### Three-dimensional grid with variable spacing in ``z``

Alternatively, to specify a stretched grid in the z-direction with a vector
of cell interfaces, we write

```jldoctest grids
julia> z_interfaces = [0, 4, 6, 7, 8]

julia> grid = RectilinearGrid(size = (16, 8, 4),
                              x = (0, 64),
                              y = (0, 32),
                              z = z_interfaces,
                              topology = (Periodic, Periodic, Bounded))
32×64×256 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 64.0)  regularly spaced with Δx=4.0
├── Periodic y ∈ [0.0, 32.0)  regularly spaced with Δy=4.0
└── Bounded  z ∈ [0.0, 8.0]   regularly spaced with Δz=2.0
```

Notice that the number of vertical cell interfaces is ``Nz + 1 = 5``, where ``Nz = 4`` is the number
of cells in the vertical.

### Building a grid on the GPU

The first positional argument in either `RectilinearGrid` or `LatitudeLongitudeGrid` is the grid's
architecture. By default `architecture = CPU()`. By providing `GPU()` as the `architecture` argument
we can construct the grid on GPU:

```julia
julia> grid = RectilinearGrid(GPU(), size = (32, 64, 256), extent = (128, 256, 512))
32×64×256 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on GPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 128.0)  regularly spaced with Δx=4.0
├── Periodic y ∈ [0.0, 256.0)  regularly spaced with Δy=4.0
└── Bounded  z ∈ [-512.0, 0.0] regularly spaced with Δz=2.0
```

## Building a two-dimensional grid in ``x, y``

To build a two-dimensional, ``16 \times 8`` grid in ``x, y`` on the domain ``(0, 2π) \times (0, π)``,
we write

```jldoctest grids
julia> grid = RectilinearGrid(size = (16, 8),
                              x = (0, 2π),
                              y = (0, π),
                              topology = (Periodic, Periodic, Flat))
32×64×256 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 64.0)  regularly spaced with Δx=4.0
├── Periodic y ∈ [0.0, 32.0)  regularly spaced with Δy=4.0
└── Flat z
```

Here we have omitted the `z` keyword argument, and `size` is a 2-`Tuple` rather than a
3-`Tuple` as in the previous examples.

### Even more complicated example!

For a "channel" model, as the one we constructed above, one would probably like to have finer resolution near
the channel walls. We construct a grid that has non-regular spacing in the bounded dimensions, here ``y`` and ``z``
by prescribing functions for `y` and `z` keyword arguments.

For example, we can use the Chebychev nodes, which are more closely stacked near boundaries, to prescribe the
``y``- and ``z``-faces.

```jldoctest grids
julia> Nx, Ny, Nz = 64, 64, 32;

julia> Lx, Ly, Lz = 1e4, 1e4, 1e3;

julia> chebychev_spaced_y_faces(j) = - Ly/2 * cos(π * (j - 1) / Ny);

julia> chebychev_spaced_z_faces(k) = - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);

julia> grid = RectilinearGrid(size = (Nx, Ny, Nz),
                              topology = (Periodic, Bounded, Bounded),
                              x = (0, Lx),
                              y = chebychev_spaced_y_faces,
                              z = chebychev_spaced_z_faces)
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
