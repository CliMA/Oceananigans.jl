# Grids

We currently support only `RegularRectilinearOrthogonalGrid`s with constant grid spacings.
The spacing can be different for each dimension.

A `RegularRectilinearOrthogonalGrid` is constructed by specifying the `size` of the grid (a `Tuple` specifying the number of
grid points in each direction) and either the `extent` (a `Tuple` specifying the physical extent of the grid in
each direction), or 2-`Tuple`s `x`, `y`, and `z` (for a 3D grid) that defines the the _end points_ in each direction.

A regular rectilinear orthogonal grid with ``N_x \times N_y \times N_z = 32 \times 64 \times 256`` grid points and an `extent` of
``L_x = 128`` meters, ``L_y = 256`` meters, and ``L_z = 512`` meters is constructed using

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest
julia> grid = RegularRectilinearOrthogonalGrid(size=(32, 64, 256), extent=(128, 256, 512))
RegularRectilinearOrthogonalGrid{Float64, Periodic, Periodic, Bounded}
                   domain: x ∈ [0.0, 128.0], y ∈ [0.0, 256.0], z ∈ [-512.0, 0.0]
                 topology: (Periodic, Periodic, Bounded)
  resolution (Nx, Ny, Nz): (32, 64, 256)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (4.0, 4.0, 2.0)
```

!!! info "Default domain"
    When using the `extent` keyword, the domain is ``x \in [0, L_x]``, ``y \in [0, L_y]``, and ``z \in [-L_z, 0]``
    -- a sensible choice for oceanographic applications.

## Specifying the grid's topology

Another crucial keyword is a 3-`Tuple` that specifies the grid's `topology`.
In each direction the grid may be `Periodic` or `Bounded`.
By default, the `RegularRectilinearOrthogonalGrid` constructor assumes the grid topology is horizontally-periodic
and bounded in the vertical, such that `topology = (Periodic, Periodic, Bounded)`.

A "channel" model that is periodic in the ``x``-direction and wall-bounded
in the ``y``- and ``z``-dimensions is build with,

```jldoctest
julia> grid = RegularRectilinearOrthogonalGrid(topology=(Periodic, Bounded, Bounded), size=(64, 64, 32), extent=(1e4, 1e4, 1e3))
RegularRectilinearOrthogonalGrid{Float64, Periodic, Bounded, Bounded}
                   domain: x ∈ [0.0, 10000.0], y ∈ [0.0, 10000.0], z ∈ [-1000.0, 0.0]
                 topology: (Periodic, Bounded, Bounded)
  resolution (Nx, Ny, Nz): (64, 64, 32)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (156.25, 156.25, 31.25)
```

## Specifying domain end points

To specify a domain with a different origin than the default, the `x`, `y`, and `z` keyword arguments must be used.
For example, a grid with ``x \in [-100, 100]`` meters, ``y \in [0, 12.5]`` meters, and ``z \in [-π, π]`` meters
is constructed via

```jldoctest
julia> grid = RegularRectilinearOrthogonalGrid(size=(32, 16, 256), x=(-100, 100), y=(0, 12.5), z=(-π, π))
RegularRectilinearOrthogonalGrid{Float64, Periodic, Periodic, Bounded}
                   domain: x ∈ [-100.0, 100.0], y ∈ [0.0, 12.5], z ∈ [-3.141592653589793, 3.141592653589793]
                 topology: (Periodic, Periodic, Bounded)
  resolution (Nx, Ny, Nz): (32, 16, 256)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (6.25, 0.78125, 0.02454369260617026)
```
