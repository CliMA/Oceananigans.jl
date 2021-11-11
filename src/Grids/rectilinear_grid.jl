"""
    RectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ, Arch} <: AbstractRectilinearGrid{FT, TX, TY, TZ}

A rectilinear grid with with either constant or varying grid spacings between cell centers and cell faces
in all directions. Grid elements of type `FT`, topology `{TX, TY, TZ}`, grid spacings of type `{FX, FY, FZ}`
and coordinates in each direction of type `{VX, VY, VZ}`. 
"""
struct RectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ, Arch} <: AbstractRectilinearGrid{FT, TX, TY, TZ}
        architecture::Arch
        Nx :: Int
        Ny :: Int
        Nz :: Int
        Hx :: Int
        Hy :: Int
        Hz :: Int
        Lx :: FT
        Ly :: FT
        Lz :: FT
      # All directions can be either regular (FX, FY, FZ) <: Number or stretched (FX, FY, FZ) <: AbstractVector
      Δxᶠᵃᵃ :: FX
      Δxᶜᵃᵃ :: FX
      xᶠᵃᵃ  :: VX
      xᶜᵃᵃ  :: VX
      Δyᵃᶠᵃ :: FY
      Δyᵃᶜᵃ :: FY
      yᵃᶠᵃ  :: VY
      yᵃᶜᵃ  :: VY
      Δzᵃᵃᶠ :: FZ 
      Δzᵃᵃᶜ :: FZ
      zᵃᵃᶠ  :: VZ
      zᵃᵃᶜ  :: VZ
end

const XRegRectilinearGrid = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Number}
const YRegRectilinearGrid = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const ZRegRectilinearGrid = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const HRegRectilinearGrid = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Number, <:Number}
const  RegRectilinearGrid = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Number, <:Number, <:Number}

"""
    RectilinearGrid([FT = Float64];
                    architecture = CPU(),
                    size,
                    x = nothing,
                    y = nothing,
                    z = nothing,
                    halo = nothing,
                    extent = nothing,
                    topology = (Periodic, Periodic, Bounded))

Creates a `RectilinearGrid` with `size = (Nx, Ny, Nz)` grid points.

Keyword arguments
=================

- `architecture`: Specifies whether the array of coordinates, interfaces, and spacings are stored
                  on the CPU or GPU. Default: `architecture = CPU()`.

- `size` (required): A tuple prescribing the number of grid points in non-`Flat` directions.
                     `size` is a 3-tuple for 3D models, a 2-tuple for 2D models, and either a
                     scalar or 1-tuple for 1D models.

- `topology`: A 3-tuple `(TX, TY, TZ)` specifying the topology of the domain.
              `TX`, `TY`, and `TZ` specify whether the `x`-, `y`-, and `z` directions are
              `Periodic`, `Bounded`, or `Flat`. The topology `Flat` indicates that a model does
              not vary in those directions so that derivatives and interpolation are zero.
              The default is `topology = (Periodic, Periodic, Bounded)`.

- `extent`: A tuple prescribing the physical extent of the grid in non-`Flat` directions.
            The origin for three-dimensional domains is the oceanic default `(0, 0, -Lz)`.

- `x`, `y`, and `z`: Each of `x, y, z` are either i) 2-tuples that specify the end points of the domain
                     in their respect directions (in which case scalar values may be used in `Flat`
                     directions), or ii) arrays or functions of the corresponding index `i`, `j`, or `k`
                     that specifies the location of cell faces in the `x`-, `y`-, or `z`-direction respectively.
                     For example, to specify the cell faces in `z` we need to give a function that takes
                     `k` as an argument for indices `k=1` through `k = Nz+1`, where `Nz` is the `size` of the
                     stretched `z` dimension.

*Note*: _Either_ `extent`, or all of `x`, `y`, and `z` must be specified.

- `halo`: A tuple of integers that specifies the size of the halo region of cells surrounding
          the physical interior for each non-`Flat` direction.

The physical extent of the domain can be specified via `x`, `y`, and `z` keyword arguments
indicating the left and right endpoints of each dimensions, e.g. `x=(-π, π)` or via
the `extent` argument, e.g. `extent=(Lx, Ly, Lz)` which specifies the extent of each dimension
in which case 0 ≤ x ≤ Lx, 0 ≤ y ≤ Ly, and -Lz ≤ z ≤ 0.

A grid topology may be specified via a tuple assigning one of `Periodic`, `Bounded`, and `Flat`
to each dimension. By default, a horizontally periodic grid topology `(Periodic, Periodic, Bounded)`
is assumed.

Constants are stored using floating point values of type `FT`. By default this is `Float64`.
Make sure to specify the desired `FT` if not using `Float64`.

Grid properties
===============

- `(Nx, Ny, Nz) :: Int`: Number of physical points in the ``(x, y, z)``-direction.

- `(Hx, Hy, Hz) :: Int`: Number of halo points in the ``(x, y, z)``-direction.

- `(Lx, Ly, Lz) :: FT`: Physical extent of the grid in the ``(x, y, z)``-direction.

- `(Δxᶜᵃᵃ, Δyᵃᶜᵃ, Δzᵃᵃᶜ)`: Grid spacing in the ``(x, y, z)``-direction between cell centers.
                           Defined at cell centers in ``x``, ``y``, and ``z``.

- `(Δxᶠᵃᵃ, Δyᵃᶠᵃ, Δzᵃᵃᶠ)`: Grid spacing in the ``(x, y, z)``-direction between cell faces.
                           Defined at cell faces in ``x``, ``y``, and ``z``.

- `(xᶜᵃᵃ, yᵃᶜᵃ, zᵃᵃᶜ)`: ``(x, y, z)`` coordinates of cell centers.

- `(xᶠᵃᵃ, yᵃᶠᵃ, zᵃᵃᶠ)`: ``(x, y, z)`` coordinates of cell faces.

Examples
========

* A default grid with `Float64` type:

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(32, 32, 32), extent=(1, 2, 3))
RectilinearGrid{Float64, Periodic, Periodic, Bounded} on the CPU()
                   domain: x ∈ [0.0, 1.0], y ∈ [0.0, 2.0], z ∈ [-3.0, 0.0]
                 topology: (Periodic, Periodic, Bounded)
        size (Nx, Ny, Nz): (32, 32, 32)
        halo (Hx, Hy, Hz): (1, 1, 1)
grid in x: Regular, with spacing 0.03125
grid in y: Regular, with spacing 0.0625
grid in z: Regular, with spacing 0.09375
```

* A default grid with `Float32` type:

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(Float32; size=(32, 32, 16), x=(0, 8), y=(-10, 10), z=(-π, π))
RectilinearGrid{Float32, Periodic, Periodic, Bounded} on the CPU()
                   domain: x ∈ [0.0, 8.0], y ∈ [-10.0, 10.0], z ∈ [-3.1415927, 3.1415927]
                 topology: (Periodic, Periodic, Bounded)
        size (Nx, Ny, Nz): (32, 32, 16)
        halo (Hx, Hy, Hz): (1, 1, 1)
grid in x: Regular, with spacing 0.25
grid in y: Regular, with spacing 0.625
grid in z: Regular, with spacing 0.3926991
```

* A two-dimenisional, horizontally-periodic grid:

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(32, 32), extent=(2π, 4π), topology=(Periodic, Periodic, Flat))
RectilinearGrid{Float64, Periodic, Periodic, Flat} on the CPU()
                   domain: x ∈ [0.0, 6.283185307179586], y ∈ [0.0, 12.566370614359172], z ∈ [1.0, 1.0]
                 topology: (Periodic, Periodic, Flat)
        size (Nx, Ny, Nz): (32, 32, 1)
        halo (Hx, Hy, Hz): (1, 1, 0)
grid in x: Regular, with spacing 0.19634954084936207
grid in y: Regular, with spacing 0.39269908169872414
grid in z: Flattened
```

* A one-dimensional "column" grid:

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=256, z=(-128, 0), topology=(Flat, Flat, Bounded))
RectilinearGrid{Float64, Flat, Flat, Bounded} on the CPU()
                   domain: x ∈ [1.0, 1.0], y ∈ [1.0, 1.0], z ∈ [-128.0, 0.0]
                 topology: (Flat, Flat, Bounded)
        size (Nx, Ny, Nz): (1, 1, 256)
        halo (Hx, Hy, Hz): (0, 0, 1)
grid in x: Flattened
grid in y: Flattened
grid in z: Regular, with spacing 0.5
```

* A horizontally-periodic regular grid with cell interfaces stretched hyperbolically near the top:

```jldoctest
using Oceananigans

σ = 1.1 # stretching factor
Nz = 24 # vertical resolution
Lz = 32 # depth (m)

hyperbolically_spaced_faces(k) = - Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ))

grid = RectilinearGrid(size = (32, 32, Nz),
                       x = (0, 64),
                       y = (0, 64),
                       z = hyperbolically_spaced_faces)

# output
RectilinearGrid{Float64, Periodic, Periodic, Bounded} on the CPU()
                   domain: x ∈ [0.0, 64.0], y ∈ [0.0, 64.0], z ∈ [-32.0, -0.0]
                 topology: (Periodic, Periodic, Bounded)
        size (Nx, Ny, Nz): (32, 32, 24)
        halo (Hx, Hy, Hz): (1, 1, 1)
grid in x: Regular, with spacing 2.0
grid in y: Regular, with spacing 2.0
grid in z: Stretched, with spacing min=0.6826950100338962, max=1.8309085743885056
```

* A three-dimensional grid with regular spacing in x, cell interfaces that are closely spaced
  close to the boundaries in y (closely mimicing the Chebychev nodes) and cell interfaces
  stretched in z hyperbolically near the top:

```jldoctest
using Oceananigans

Nx, Ny, Nz = 32, 30, 24
Lx, Ly, Lz = 200, 100, 32 # (m)

chebychev_like_spaced_faces(j) = - Ly/2 * cos(π * (j-1) / Ny)

σ = 1.1 # stretching factor
hyperbolically_spaced_faces(j) = - Lz * (1 - tanh(σ * (j - 1) / Nz) / tanh(σ))

grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       topology=(Periodic, Bounded, Bounded),
                       x = (0, Lz),
                       y = chebychev_like_spaced_faces,
                       z = hyperbolically_spaced_faces)

# output
RectilinearGrid{Float64, Periodic, Bounded, Bounded} on the CPU()
                   domain: x ∈ [0.0, 32.0], y ∈ [-50.0, 50.0], z ∈ [-32.0, -0.0]
                 topology: (Periodic, Periodic, Bounded)
        size (Nx, Ny, Nz): (32, 30, 24)
        halo (Hx, Hy, Hz): (1, 1, 1)
grid in x: Regular, with spacing 1.0
grid in y: Stretched, with spacing min=0.2739052315863262, max=5.22642316338267
grid in z: Stretched, with spacing min=0.6826950100338962, max=1.8309085743885056
```
"""
function RectilinearGrid(FT = Float64;
                         architecture = CPU(),
                         size,
                         x = nothing,
                         y = nothing,
                         z = nothing,
                         halo = nothing,
                         extent = nothing,
                         topology = (Periodic, Periodic, Bounded))

    TX, TY, TZ = validate_topology(topology)
    size = validate_size(TX, TY, TZ, size)
    halo = validate_halo(TX, TY, TZ, halo)

    # Validate the rectilinear domain
    x, y, z = validate_rectilinear_domain(TX, TY, TZ, FT, extent, x, y, z)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶠᵃᵃ, Δxᶜᵃᵃ = generate_coordinate(FT, topology[1], Nx, Hx, x, architecture)
    Ly, yᵃᶠᵃ, yᵃᶜᵃ, Δyᵃᶠᵃ, Δyᵃᶜᵃ = generate_coordinate(FT, topology[2], Ny, Hy, y, architecture)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topology[3], Nz, Hz, z, architecture)
 
    FX   = typeof(Δxᶠᵃᵃ)
    FY   = typeof(Δyᵃᶠᵃ)
    FZ   = typeof(Δzᵃᵃᶠ)
    VX   = typeof(xᶠᵃᵃ)
    VY   = typeof(yᵃᶠᵃ)
    VZ   = typeof(zᵃᵃᶠ)
    Arch = typeof(architecture) 

    return RectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ, Arch}(architecture,
        Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ, Δyᵃᶜᵃ, Δyᵃᶠᵃ, yᵃᶠᵃ, yᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ)
end

@inline x_domain(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TX, grid.Nx, grid.xᶠᵃᵃ)
@inline y_domain(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TY, grid.Ny, grid.yᵃᶠᵃ)
@inline z_domain(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)

short_show(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    "RectilinearGrid{$FT, $TX, $TY, $TZ}(Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz))"

function domain_string(grid::RectilinearGrid)
    x₁, x₂ = domain(topology(grid, 1), grid.Nx, grid.xᶠᵃᵃ)
    y₁, y₂ = domain(topology(grid, 2), grid.Ny, grid.yᵃᶠᵃ)
    z₁, z₂ = domain(topology(grid, 3), grid.Nz, grid.zᵃᵃᶠ)
    return "x ∈ [$x₁, $x₂], y ∈ [$y₁, $y₂], z ∈ [$z₁, $z₂]"
end

function show(io::IO, g::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    print(io, "RectilinearGrid{$FT, $TX, $TY, $TZ} on the $(g.architecture)\n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "        size (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "        halo (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid in x: ", show_coordinate(g.Δxᶜᵃᵃ, TX), '\n',
              "grid in y: ", show_coordinate(g.Δyᵃᶜᵃ, TY), '\n',
              "grid in z: ", show_coordinate(g.Δzᵃᵃᶜ, TZ))
end


Adapt.adapt_structure(to, grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
             RectilinearGrid{FT, TX, TY, TZ, 
                          typeof(Adapt.adapt(to, grid.Δxᶜᵃᵃ)),
                          typeof(Adapt.adapt(to, grid.Δyᵃᶠᵃ)),
                          typeof(Adapt.adapt(to, grid.Δzᵃᵃᶠ)),
                          typeof(Adapt.adapt(to, grid.xᶠᵃᵃ)),
                          typeof(Adapt.adapt(to, grid.yᵃᶠᵃ)),
                          typeof(Adapt.adapt(to, grid.zᵃᵃᶠ)),
                          Nothing}(
        nothing,
        grid.Nx, grid.Ny, grid.Nz,
        grid.Hx, grid.Hy, grid.Hz,
        grid.Lx, grid.Ly, grid.Lz,
        Adapt.adapt(to, grid.Δxᶠᵃᵃ),
        Adapt.adapt(to, grid.Δxᶜᵃᵃ),
        Adapt.adapt(to, grid.xᶠᵃᵃ),
        Adapt.adapt(to, grid.xᶜᵃᵃ),
        Adapt.adapt(to, grid.Δyᵃᶠᵃ),
        Adapt.adapt(to, grid.Δyᵃᶜᵃ),
        Adapt.adapt(to, grid.yᵃᶠᵃ),
        Adapt.adapt(to, grid.yᵃᶜᵃ),
        Adapt.adapt(to, grid.Δzᵃᵃᶠ),
        Adapt.adapt(to, grid.Δzᵃᵃᶜ),
        Adapt.adapt(to, grid.zᵃᵃᶠ),
        Adapt.adapt(to, grid.zᵃᵃᶜ))

@inline xnode(::Center, i, grid::RectilinearGrid) = @inbounds grid.xᶜᵃᵃ[i]
@inline xnode(::Face  , i, grid::RectilinearGrid) = @inbounds grid.xᶠᵃᵃ[i]

@inline ynode(::Center, j, grid::RectilinearGrid) = @inbounds grid.yᵃᶜᵃ[j]
@inline ynode(::Face  , j, grid::RectilinearGrid) = @inbounds grid.yᵃᶠᵃ[j]

@inline znode(::Center, k, grid::RectilinearGrid) = @inbounds grid.zᵃᵃᶜ[k]
@inline znode(::Face  , k, grid::RectilinearGrid) = @inbounds grid.zᵃᵃᶠ[k]

all_x_nodes(::Type{Center}, grid::RectilinearGrid) = grid.xᶜᵃᵃ
all_x_nodes(::Type{Face}  , grid::RectilinearGrid) = grid.xᶠᵃᵃ
all_y_nodes(::Type{Center}, grid::RectilinearGrid) = grid.yᵃᶜᵃ
all_y_nodes(::Type{Face}  , grid::RectilinearGrid) = grid.yᵃᶠᵃ
all_z_nodes(::Type{Center}, grid::RectilinearGrid) = grid.zᵃᵃᶜ
all_z_nodes(::Type{Face}  , grid::RectilinearGrid) = grid.zᵃᵃᶠ

function with_halo(new_halo, old_grid::RectilinearGrid)

    size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)
    topo = topology(old_grid)

   
    if old_grid.Δxᶠᵃᵃ isa Number
        x = x_domain(old_grid)
    else
        x = old_grid.xᶠᵃᵃ
    end

    if old_grid.Δyᵃᶠᵃ isa Number
        y = y_domain(old_grid)
    else
        y = old_grid.yᵃᶠᵃ
    end

    if old_grid.Δzᵃᵃᶠ isa Number
        z = z_domain(old_grid)
    else
        z = old_grid.zᵃᵃᶠ
    end

    # Remove elements of size and new_halo in Flat directions as expected by grid
    # constructor
    size     = pop_flat_elements(size, topo)
    new_halo = pop_flat_elements(new_halo, topo)

    new_grid = RectilinearGrid(eltype(old_grid);
               architecture = old_grid.architecture,
               size = size,
               x = x, y = y,z = z,
               topology = topo,
               halo = new_halo)

    return new_grid
end

# Get minima of grid
#

function min_Δx(grid::RectilinearGrid)
    topo = topology(grid)
    if topo[1] == Flat
        return Inf
    else
        return min_number_or_array(grid.Δxᶜᵃᵃ)
    end
end

function min_Δy(grid::RectilinearGrid)
    topo = topology(grid)
    if topo[2] == Flat
        return Inf
    else
        return min_number_or_array(grid.Δyᵃᶜᵃ)
    end
end

function min_Δz(grid::RectilinearGrid)
    topo = topology(grid)
    if topo[3] == Flat
        return Inf
    else
        return min_number_or_array(grid.Δzᵃᵃᶜ)
    end
end

@inline min_number_or_array(var) = var
@inline min_number_or_array(var::AbstractVector) = minimum(parent(var))