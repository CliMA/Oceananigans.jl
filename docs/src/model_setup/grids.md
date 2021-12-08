# Grids

We currently support only `RectilinearGrid`s with either constant or variable grid spacings.
The spacings can be different for each dimension.

A `RectilinearGrid` is constructed by specifying the `size` of the grid (a `Tuple` specifying
the number of grid points in each direction) and either the `extent` (a `Tuple` specifying the
physical extent of the grid in each direction), or by prescribing `x`, `y`, and `z`. Keyword
arguments `x`, `y`, and `z` could be either *(i)* 2-`Tuple`s that define the the _end points_ in
each direction, or *(ii)* arrays or functions of the corresponding indices `i`, `j`, or `k` that
specify the locations of cell faces in the `x`-, `y`-, or `z`-direction, respectively.

A regular rectilinear grid with ``N_x \times N_y \times N_z = 32 \times 64 \times 256`` grid points
and an `extent` of ``L_x = 128`` meters, ``L_y = 256`` meters, and ``L_z = 512`` meters is constructed
by

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest
julia> grid = RectilinearGrid(size=(32, 64, 256), extent=(128, 256, 512))
RectilinearGrid{Float64, Periodic, Periodic, Bounded} 
             architecture: CPU()
                   domain: x ∈ [0.0, 128.0], y ∈ [0.0, 256.0], z ∈ [-512.0, 0.0]
                 topology: (Periodic, Periodic, Bounded)
        size (Nx, Ny, Nz): (32, 64, 256)
        halo (Hx, Hy, Hz): (1, 1, 1)
                grid in x: Regular, with spacing 4.0
                grid in y: Regular, with spacing 4.0
                grid in z: Regular, with spacing 2.0
```

!!! info "Default domain"
    When using the `extent` keyword, e.g., `extent = (Lx, Ly, Lz)`, then the ``x \in [0, L_x]``,
    ``y \in [0, L_y]``, and ``z \in [-L_z, 0]`` -- a sensible choice for oceanographic applications.

## Specifying the grid's topology

Another crucial keyword is a 3-`Tuple` that specifies the grid's `topology`.
In each direction the grid may be `Periodic`, `Bounded` or `Flat`.
By default, both the `RectilinearGrid` and the `RectilinearGrid` constructors 
assume the grid topology is horizontally-periodic
and bounded in the vertical, such that `topology = (Periodic, Periodic, Bounded)`.

A "channel" model that is periodic in the ``x``-direction and wall-bounded
in the ``y``- and ``z``-dimensions is build with,

```jldoctest
julia> grid = RectilinearGrid(topology=(Periodic, Bounded, Bounded), size=(64, 64, 32), extent=(1e4, 1e4, 1e3))
RectilinearGrid{Float64, Periodic, Bounded, Bounded} 
             architecture: CPU()
                   domain: x ∈ [0.0, 10000.0], y ∈ [0.0, 10000.0], z ∈ [-1000.0, 0.0]
                 topology: (Periodic, Bounded, Bounded)
        size (Nx, Ny, Nz): (64, 64, 32)
        halo (Hx, Hy, Hz): (1, 1, 1)
                grid in x: Regular, with spacing 156.25
                grid in y: Regular, with spacing 156.25
                grid in z: Regular, with spacing 31.25
```

The `Flat` topology is useful when running problems with fewer than 3 dimensions. As an example,
to use a two-dimensional horizontal, doubly periodic domain the topology is `(Periodic, Periodic, Flat)`.


## Specifying domain end points

To specify a domain with a different origin than the default, the `x`, `y`, and `z` keyword arguments must be used.
For example, a grid with ``x \in [-100, 100]`` meters, ``y \in [0, 12.5]`` meters, and ``z \in [-\pi, \pi]`` meters
is constructed via

```jldoctest
julia> grid = RectilinearGrid(size=(32, 16, 256), x=(-100, 100), y=(0, 12.5), z=(-π, π))
RectilinearGrid{Float64, Periodic, Periodic, Bounded}  
             architecture: CPU()
                   domain: x ∈ [-100.0, 100.0], y ∈ [0.0, 12.5], z ∈ [-3.141592653589793, 3.141592653589793]
                 topology: (Periodic, Periodic, Bounded)
        size (Nx, Ny, Nz): (32, 16, 256)
        halo (Hx, Hy, Hz): (1, 1, 1)
                grid in x: Regular, with spacing 6.25
                grid in y: Regular, with spacing 0.78125
                grid in z: Regular, with spacing 0.02454369260617026
```


## Grids with non-regular spacing in some of the directions

For a "channel" model, as the one we constructed above, one would probably like to have finer resolution near
the channel walls. We construct a grid that has non-regular spacing in the bounded dimensions, here ``y`` and ``z``
by by prescribing functions for `y` and `z` keyword arguments. 

For example, we can use the Chebychev nodes, which are more closely stacked near boundaries, to prescribe the
``y``- and ``z``-faces.

```jldoctest
julia> Nx, Ny, Nz = 64, 64, 32;

julia> Lx, Ly, Lz = 1e4, 1e4, 1e3;

julia> chebychev_spaced_y_faces(j) = - Ly/2 * cos(π * (j - 1) / Ny);

julia> chebychev_spaced_z_faces(k) = - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);

julia> grid = RectilinearGrid(size = (Nx, Ny, Nz),
                              topology=(Periodic, Bounded, Bounded),
                              x = (0, Lx),
                              y = chebychev_spaced_y_faces,
                              z = chebychev_spaced_z_faces)
RectilinearGrid{Float64, Periodic, Bounded, Bounded}  
             architecture: CPU()
                   domain: x ∈ [0.0, 10000.0], y ∈ [-5000.0, 5000.0], z ∈ [-1000.0, 0.0]
                 topology: (Periodic, Bounded, Bounded)
        size (Nx, Ny, Nz): (64, 64, 32)
        halo (Hx, Hy, Hz): (1, 1, 1)
                grid in x: Regular, with spacing 156.25
                grid in y: Stretched, with spacing min=6.022718974138115, max=245.33837163709035
                grid in z: Stretched, with spacing min=2.407636663901485, max=49.008570164780394
```

```@setup 1
using Oceananigans
using Plots
Plots.scalefontsizes(1.25)
Plots.default(lw=3)
Nx, Ny, Nz = 64, 64, 32
Lx, Ly, Lz = 1e4, 1e4, 1e3
chebychev_spaced_y_faces(j) = - Ly/2 * cos(π * (j - 1) / Ny);
chebychev_spaced_z_faces(k) = - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);
grid = RectilinearGrid(size = (Nx, Ny, Nz),
                              topology=(Periodic, Bounded, Bounded),
                              x = (0, Lx),
                              y = chebychev_spaced_y_faces,
                              z = chebychev_spaced_z_faces)
```

We can easily visualize the spacing of ``y`` and ``z`` directions.

```@example 1
using Plots

py = plot(grid.yᵃᶜᵃ[1:Ny],  grid.Δyᵃᶜᵃ[1:Ny],
           marker = :circle,
           ylabel = "y-spacing (m)",
           xlabel = "y (m)",
           legend = nothing,
           ylims = (0, 250))

pz = plot(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz],
          marker = :square,
          ylabel = "z (m)",
          xlabel = "z-spacing (m)",
          legend = nothing,
          xlims = (0, 50))

plot(py, pz, layout=(2, 1), size=(800, 900))

savefig("plot_stretched_grid.svg"); nothing # hide
```

![](plot_stretched_grid.svg)
