using OrderedCollections: OrderedDict

struct RectilinearGrid{FT, TX, TY, TZ, CZ, FX, FY, VX, VY, Arch, Sz} <: AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, Arch}
    architecture :: Arch
    Nx :: Int
    Ny :: Int
    Nz :: Int
    Hx :: Int
    Hy :: Int
    Hz :: Int
    Lx :: FT
    Ly :: FT
    Lz :: FT
    # All directions can be either regular (FX, FY, FZ) <: Number
    # or stretched (FX, FY, FZ) <: AbstractVector
    Δxᶠᵃᵃ :: FX
    Δxᶜᵃᵃ :: FX
    xᶠᵃᵃ  :: VX
    xᶜᵃᵃ  :: VX
    Δyᵃᶠᵃ :: FY
    Δyᵃᶜᵃ :: FY
    yᵃᶠᵃ  :: VY
    yᵃᶜᵃ  :: VY
    z     :: CZ
end

function RectilinearGrid{TX, TY, TZ}(arch::Arch, Nx, Ny, Nz, Hx, Hy, Hz,
                                     Lx :: FT, Ly :: FT, Lz :: FT,
                                     Δxᶠᵃᵃ :: FX, Δxᶜᵃᵃ :: FX,
                                      xᶠᵃᵃ :: VX,  xᶜᵃᵃ :: VX,
                                     Δyᵃᶠᵃ :: FY, Δyᵃᶜᵃ :: FY,
                                      yᵃᶠᵃ :: VY,  yᵃᶜᵃ :: VY,
                                      z    :: CZ) where {Arch, FT, TX, TY, TZ,
                                                         FX, VX, FY, VY, CZ}

    size = GridSize(Nx, Ny, Nz, Hx, Hy, Hz)
    SZ   = typeof(size)

    return RectilinearGrid{FT, TX, TY, TZ,
                           CZ, FX, FY, VX, VY, Arch, SZ}(arch, Nx, Ny, Nz,
                                                         Hx, Hy, Hz, Lx, Ly, Lz,
                                                         Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ,
                                                         Δyᵃᶠᵃ, Δyᵃᶜᵃ, yᵃᶠᵃ, yᵃᶜᵃ, z)
end

# Read size and halo from the trailing `GridSize` type parameter so both are compile-time constants.
@generated function Base.size(grid::RectilinearGrid)
    sz = grid.parameters[end].parameters
    return :(($(sz[1]), $(sz[2]), $(sz[3])))
end

@generated function halo_size(grid::RectilinearGrid)
    sz = grid.parameters[end].parameters
    return :(($(sz[4]), $(sz[5]), $(sz[6])))
end

const RG = RectilinearGrid

const XRegularRG   = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const YRegularRG   = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,    <:Number}
const ZRegularRG   = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:RegularVerticalCoordinate}
const XYRegularRG  = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Number, <:Number}
const XZRegularRG  = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:RegularVerticalCoordinate, <:Number}
const YZRegularRG  = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:RegularVerticalCoordinate, <:Any, <:Number}
const XYZRegularRG = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:RegularVerticalCoordinate, <:Number, <:Number}

regular_dimensions(::XRegularRG)   = tuple(1)
regular_dimensions(::YRegularRG)   = tuple(2)
regular_dimensions(::ZRegularRG)   = tuple(3)
regular_dimensions(::XYRegularRG)  = (1, 2)
regular_dimensions(::XZRegularRG)  = (1, 3)
regular_dimensions(::YZRegularRG)  = (2, 3)
regular_dimensions(::XYZRegularRG) = (1, 2, 3)

stretched_dimensions(::YZRegularRG) = tuple(1)
stretched_dimensions(::XZRegularRG) = tuple(2)
stretched_dimensions(::XYRegularRG) = tuple(3)

"""
    RectilinearGrid([architecture = CPU(), FT = Float64];
                    size,
                    x = nothing,
                    y = nothing,
                    z = nothing,
                    halo = nothing,
                    extent = nothing,
                    topology = (Periodic, Periodic, Bounded))

Create a `RectilinearGrid` with `size = (Nx, Ny, Nz)` grid points.

Positional arguments
====================

- `architecture`: Specifies whether arrays of coordinates and spacings are stored
                  on the CPU or GPU. Default: `CPU()`.

- `FT`: Floating point data type. Default: `Float64`.

Keyword arguments
=================

- `size` (required): A tuple prescribing the number of grid points in non-`Flat` directions.
                     `size` is a 3-tuple for 3D models, a 2-tuple for 2D models, and either a
                     scalar or 1-tuple for 1D models.

- `topology`: A 3-tuple `(TX, TY, TZ)` specifying the topology of the domain.
              `TX`, `TY`, and `TZ` specify whether the `x`-, `y`-, and `z` directions are
              `Periodic`, `Bounded`, or `Flat`. The topology `Flat` indicates that a model does
              not vary in those directions so that derivatives and interpolation are zero.
              The default is `topology = (Periodic, Periodic, Bounded)`.

- `extent`: A tuple prescribing the physical extent of the grid in non-`Flat` directions, e.g.,
            `(Lx, Ly, Lz)`. All directions are constructed with regular grid spacing and the domain
            (in the case that no direction is `Flat`) is ``0 ≤ x ≤ L_x``, ``0 ≤ y ≤ L_y``, and
            ``-L_z ≤ z ≤ 0``, which is most appropriate for oceanic applications in which ``z = 0``
            usually is the ocean's surface.

- `x`, `y`, and `z`: Each of `x, y, z` are either (i) 2-tuples that specify the end points of the domain
                     in their respect directions (in which case scalar values may be used in `Flat`
                     directions), (ii) arrays that specify the locations of cell faces in the `x`-, `y`-,
                     or `z`-direction, or (iii) functions of the corresponding indices `i`, `j`, or `k`
                     that specify the locations of cell faces in the `x`-, `y`-, or `z`-direction, respectively.
                     For example, to prescribe the cell faces in `z` we need to provide a function that takes
                     `k` as argument and returns the location of the faces for indices `k = 1` through `k = Nz + 1`,
                     where `Nz` is the `size` of the stretched `z` dimension.

  !!! note "Physical extent of grid"
      _Either_ `extent`, or _all_ of `x`, `y`, and `z` must be specified.

- `halo`: A tuple of integers that specifies the size of the halo region, that is the number of cells surrounding
          the physical interior for each non-`Flat` direction. The default is 3 halo cells in every direction.

The physical extent of the domain can be specified either via `x`, `y`, and `z` keyword arguments
indicating the left and right endpoints of each dimensions, e.g., `x = (-π, π)` or via
the `extent` argument, e.g., `extent = (Lx, Ly, Lz)`, which specifies the extent of each dimension
in which case ``0 ≤ x ≤ L_x``, ``0 ≤ y ≤ L_y``, and ``-L_z ≤ z ≤ 0``.

A grid topology may be specified via a tuple assigning one of `Periodic`, `Bounded`, and, `Flat`
to each dimension. By default, a horizontally periodic grid topology `(Periodic, Periodic, Bounded)`
is assumed.

Constants are stored using floating point values of type `FT`. By default this is `Float64`.
Make sure to specify the desired `FT` if not using `Float64`.

Grid properties
===============

- `architecture`: The grid's architecture.

- `(Nx, Ny, Nz) :: Int`: Number of physical points in the ``(x, y, z)``-direction.

- `(Hx, Hy, Hz) :: Int`: Number of halo points in the ``(x, y, z)``-direction.

- `(Lx, Ly, Lz) :: FT`: Physical extent of the grid in the ``(x, y, z)``-direction.

- `(Δxᶜᵃᵃ, Δyᵃᶜᵃ, z.Δcᵃᵃᶜ)`: Spacings in the ``(x, y, z)``-directions between the cell faces.
                             These are the lengths in ``x``, ``y``, and ``z`` of `Center` cells and are
                             defined at `Center` locations.

- `(Δxᶠᵃᵃ, Δyᵃᶠᵃ, z.Δcᵃᵃᶠ)`: Spacings in the ``(x, y, z)``-directions between the cell centers.
                             These are the lengths in ``x``, ``y``, and ``z`` of `Face` cells and are
                             defined at `Face` locations.

- `(xᶜᵃᵃ, yᵃᶜᵃ, z.cᵃᵃᶜ)`: ``(x, y, z)`` coordinates of cell `Center`s.

- `(xᶠᵃᵃ, yᵃᶠᵃ, z.cᵃᵃᶠ)`: ``(x, y, z)`` coordinates of cell `Face`s.

Examples
========

* A grid with the default `Float64` type:

```jldoctest
using Oceananigans
grid = RectilinearGrid(size=(32, 32, 32), extent=(1, 2, 3))

# output

32×32×32 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 1.0)  regularly spaced with Δx=0.03125
├── Periodic y ∈ [0.0, 2.0)  regularly spaced with Δy=0.0625
└── Bounded  z ∈ [-3.0, 0.0] regularly spaced with Δz=0.09375
```

* A grid with `Float32` type:

```jldoctest
using Oceananigans
grid = RectilinearGrid(Float32; size=(32, 32, 16), x=(0, 8), y=(-10, 10), z=(-π, π))

# output

32×32×16 RectilinearGrid{Float32, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 8.0)          regularly spaced with Δx=0.25
├── Periodic y ∈ [-10.0, 10.0)       regularly spaced with Δy=0.625
└── Bounded  z ∈ [-3.14159, 3.14159] regularly spaced with Δz=0.392699
```

* A two-dimenisional, horizontally-periodic grid:

```jldoctest
using Oceananigans
grid = RectilinearGrid(size=(32, 32), extent=(2π, 4π), topology=(Periodic, Periodic, Flat))

# output

32×32×1 RectilinearGrid{Float64, Periodic, Periodic, Flat} on CPU with 3×3×0 halo
├── Periodic x ∈ [3.60072e-17, 6.28319) regularly spaced with Δx=0.19635
├── Periodic y ∈ [7.20145e-17, 12.5664) regularly spaced with Δy=0.392699
└── Flat z
```

* A one-dimensional "column" grid:

```jldoctest
using Oceananigans
grid = RectilinearGrid(size=256, z=(-128, 0), topology=(Flat, Flat, Bounded))

# output

1×1×256 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── Flat x
├── Flat y
└── Bounded  z ∈ [-128.0, 0.0] regularly spaced with Δz=0.5
```

* A horizontally-periodic regular grid with cell interfaces stretched hyperbolically near the top:

```jldoctest
using Oceananigans

σ = 1.1 # stretching factor
Nz = 24 # vertical resolution
Lz = 32 # depth (m)

hyperbolically_spaced_faces(k) = - Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ))

grid = RectilinearGrid(size = (32, 32, Nz),
                       x = (0, 64), y = (0, 64),
                       z = hyperbolically_spaced_faces)

# output

32×32×24 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 64.0)   regularly spaced with Δx=2.0
├── Periodic y ∈ [0.0, 64.0)   regularly spaced with Δy=2.0
└── Bounded  z ∈ [-32.0, -0.0] variably spaced with min(Δz)=0.682695, max(Δz)=1.83091
```

* A three-dimensional grid with regular spacing in ``x``, cell interfaces at Chebyshev nodes
  in ``y``, and cell interfaces hyperbolically stretched in ``z`` near the top:

```jldoctest
using Oceananigans

Nx, Ny, Nz = 32, 30, 24
Lx, Ly, Lz = 200, 100, 32 # (m)

chebychev_nodes(j) = - Ly/2 * cos(π * (j - 1) / Ny)

σ = 1.1 # stretching factor
hyperbolically_spaced_faces(k) = - Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ))

grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       topology = (Periodic, Bounded, Bounded),
                       x = (0, Lx),
                       y = chebychev_nodes,
                       z = hyperbolically_spaced_faces)

# output

32×30×24 RectilinearGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 200.0)  regularly spaced with Δx=6.25
├── Bounded  y ∈ [-50.0, 50.0] variably spaced with min(Δy)=0.273905, max(Δy)=5.22642
└── Bounded  z ∈ [-32.0, -0.0] variably spaced with min(Δz)=0.682695, max(Δz)=1.83091
```
"""
function RectilinearGrid(architecture::AbstractArchitecture = CPU(),
                         FT::DataType = Oceananigans.defaults.FloatType;
                         size,
                         x = nothing,
                         y = nothing,
                         z = nothing,
                         halo = nothing,
                         extent = nothing,
                         topology = (Periodic, Periodic, Bounded))

    topology, size, halo, x, y, z = validate_rectilinear_grid_args(topology, size, halo, FT, extent, x, y, z)

    TX, TY, TZ = topology
    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶠᵃᵃ, Δxᶜᵃᵃ = generate_coordinate(FT, topology, size, halo, x, :x, 1, architecture)
    Ly, yᵃᶠᵃ, yᵃᶜᵃ, Δyᵃᶠᵃ, Δyᵃᶜᵃ = generate_coordinate(FT, topology, size, halo, y, :y, 2, architecture)
    Lz, z                        = generate_coordinate(FT, topology, size, halo, z, :z, 3, architecture)

    return RectilinearGrid{TX, TY, TZ}(architecture,
                                       Nx, Ny, Nz,
                                       Hx, Hy, Hz,
                                       Lx, Ly, Lz,
                                       Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ,
                                       Δyᵃᶠᵃ, Δyᵃᶜᵃ, yᵃᶠᵃ, yᵃᶜᵃ,
                                       z)
end

""" Validate user input arguments to the `RectilinearGrid` constructor. """
function validate_rectilinear_grid_args(topology, size, halo, FT, extent, x, y, z)
    TX, TY, TZ = topology = validate_topology(topology)
    size = validate_size(TX, TY, TZ, size)
    halo = validate_halo(TX, TY, TZ, size, halo)

    # Validate the rectilinear domain
    x, y, z = validate_rectilinear_domain(TX, TY, TZ, FT, size, extent, x, y, z)

    return topology, size, halo, x, y, z
end

#####
##### Showing grids
#####

x_domain(grid::RectilinearGrid) = domain(topology(grid, 1)(), grid.Nx, grid.xᶠᵃᵃ)
y_domain(grid::RectilinearGrid) = domain(topology(grid, 2)(), grid.Ny, grid.yᵃᶠᵃ)

# architecture = CPU() default, assuming that a DataType positional arg
# is specifying the floating point type.
RectilinearGrid(FT::DataType; kwargs...) = RectilinearGrid(CPU(), FT; kwargs...)

function Base.summary(grid::RectilinearGrid)
    FT = eltype(grid)
    nTX, nTY, nTZ = map(T -> nameof(T), topology(grid))
    return string(size_summary(grid),
                  " RectilinearGrid{$FT, $nTX, $nTY, $nTZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo")
end

function Base.show(io::IO, grid::RectilinearGrid, withsummary=true)
    TX, TY, TZ = topology(grid)

    Ωx = domain(TX(), grid.Nx, grid.xᶠᵃᵃ)
    Ωy = domain(TY(), grid.Ny, grid.yᵃᶠᵃ)
    Ωz = domain(TZ(), grid.Nz, grid.z.cᵃᵃᶠ)

    x_summary = domain_summary(TX(), "x", Ωx)
    y_summary = domain_summary(TY(), "y", Ωy)
    z_summary = domain_summary(TZ(), "z", Ωz)

    longest = max(length(x_summary), length(y_summary), length(z_summary))

    x_summary = dimension_summary(TX(), "x", Ωx, grid.Δxᶜᵃᵃ, longest - length(x_summary))
    y_summary = dimension_summary(TY(), "y", Ωy, grid.Δyᵃᶜᵃ, longest - length(y_summary))
    z_summary = dimension_summary(TZ(), "z", Ωz, grid.z,     longest - length(z_summary))

    if withsummary
        print(io, summary(grid), "\n")
    end

    return print(io, "├── ", x_summary, "\n",
                     "├── ", y_summary, "\n",
                     "└── ", z_summary)
end

#####
##### For "column ensemble models"
#####

struct ColumnEnsembleSize{C<:Tuple{Int, Int}}
    ensemble :: C
    Nz :: Int
    Hz :: Int
end

ColumnEnsembleSize(; Nz, ensemble=(0, 0), Hz=1) = ColumnEnsembleSize(ensemble, Nz, Hz)
validate_size(TX, TY, TZ, e::ColumnEnsembleSize) = tuple(e.ensemble[1], e.ensemble[2], e.Nz)
validate_halo(TX, TY, TZ, size, e::ColumnEnsembleSize) = tuple(0, 0, e.Hz)

#####
##### Utilities
#####

function Adapt.adapt_structure(to, grid::RectilinearGrid)
    TX, TY, TZ = topology(grid)
    return RectilinearGrid{TX, TY, TZ}(nothing,
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
                                       Adapt.adapt(to, grid.z))
end

cpu_face_constructor_x(grid::XRegularRG) = x_domain(grid)
cpu_face_constructor_y(grid::YRegularRG) = y_domain(grid)

function constructor_arguments(grid::RectilinearGrid)
    arch = architecture(grid)

    # We use OrderedDict to preserve order of keys. Important for positional arguments since we wanna be able to splat them.
    args = OrderedDict(:architecture => arch, :number_type => eltype(grid))

    # Kwargs
    topo = topology(grid)

    if (topo[1] == Flat && grid.Nx > 1) ||
       (topo[2] == Flat && grid.Ny > 1)
        size = halo = ColumnEnsembleSize(Nz=grid.Nz, Hz=grid.Hz, ensemble=(grid.Nx, grid.Ny))
    else
        size = (grid.Nx, grid.Ny, grid.Nz)
        halo = (grid.Hx, grid.Hy, grid.Hz)
        size = pop_flat_elements(size, topo)
        halo = pop_flat_elements(halo, topo)
    end

    kwargs = Dict(:size => size,
                  :halo => halo,
                  :x => cpu_face_constructor_x(grid),
                  :y => cpu_face_constructor_y(grid),
                  :z => cpu_face_constructor_z(grid),
                  :topology => topo)

    return args, kwargs
end

function Base.similar(grid::RectilinearGrid)
    args, kwargs = constructor_arguments(grid)
    arch = args[:architecture]
    FT = args[:number_type]
    return RectilinearGrid(arch, FT; kwargs...)
end

"""
$(TYPEDSIGNATURES)

Return a `new_grid` that's identical to `grid` but with `number_type`.
"""
function with_number_type(FT, grid::RectilinearGrid)
    args, kwargs = constructor_arguments(grid)
    arch = args[:architecture]
    return RectilinearGrid(arch, FT; kwargs...)
end

"""
$(TYPEDSIGNATURES)

Return a `new_grid` that's identical to `grid` but with `halo`.
"""
function with_halo(halo, grid::RectilinearGrid)
    args, kwargs = constructor_arguments(grid)
    halo = pop_flat_elements(halo, topology(grid))
    kwargs[:halo] = halo
    arch = args[:architecture]
    FT = args[:number_type]
    return RectilinearGrid(arch, FT; kwargs...)
end

# See the `slice` docstring (defined in grid_utils.jl) for documentation. The `x`, `y`, `z`
# keywords set the constant coordinate of a collapsed dimension (default `:auto` → cell center).
function slice(grid::RectilinearGrid, i, j, k; x=:auto, y=:auto, z=:auto)
    arch = architecture(grid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)

    TX′, x′, Nx, Hx = slice_dimension(i, cpu_face_constructor_x(grid), grid.Nx, grid.Hx, TX; location=x)
    TY′, y′, Ny, Hy = slice_dimension(j, cpu_face_constructor_y(grid), grid.Ny, grid.Hy, TY; location=y)
    TZ′, z′, Nz, Hz = slice_dimension(k, cpu_face_constructor_z(grid), grid.Nz, grid.Hz, TZ; location=z)
    topo = (TX′, TY′, TZ′)

    sz   = pop_flat_elements((Nx, Ny, Nz), topo)
    halo = pop_flat_elements((Hx, Hy, Hz), topo)

    kwargs = Dict{Symbol, Any}(:size => sz, :halo => halo, :topology => topo,
                               :x => x′, :y => y′, :z => z′)

    return RectilinearGrid(arch, FT; kwargs...)
end

"""
$(TYPEDSIGNATURES)

Return a `new_grid` that's identical to `grid` but on `architecture`.
"""
function Architectures.on_architecture(arch::AbstractSerialArchitecture, grid::RectilinearGrid)
    if arch == architecture(grid)
        return grid
    end

    args, kwargs = constructor_arguments(grid)
    FT = args[:number_type]
    return RectilinearGrid(arch, FT; kwargs...)
end

#####
##### Definition of RectilinearGrid nodes
#####

ξname(::RG) = :x
ηname(::RG) = :y
rname(::RG) = :z

@inline xnode(i, grid::RG, ::Center) = getnode(grid.xᶜᵃᵃ, i)
@inline xnode(i, grid::RG, ::Face)   = getnode(grid.xᶠᵃᵃ, i)
@inline ynode(j, grid::RG, ::Center) = getnode(grid.yᵃᶜᵃ, j)
@inline ynode(j, grid::RG, ::Face)   = getnode(grid.yᵃᶠᵃ, j)

@inline ξnode(i, j, k, grid::RG, ℓx, ℓy, ℓz) = xnode(i, grid, ℓx)
@inline ηnode(i, j, k, grid::RG, ℓx, ℓy, ℓz) = ynode(j, grid, ℓy)

# Convenience definitions for x, y, znode
@inline xnode(i, j, k, grid::RG, ℓx, ℓy, ℓz) = xnode(i, grid, ℓx)
@inline ynode(i, j, k, grid::RG, ℓx, ℓy, ℓz) = ynode(j, grid, ℓy)

function nodes(grid::RectilinearGrid, ℓx, ℓy, ℓz; reshape=false, with_halos=false, indices=(Colon(), Colon(), Colon()))
    x = xnodes(grid, ℓx, ℓy, ℓz; with_halos, indices = indices[1])
    y = ynodes(grid, ℓx, ℓy, ℓz; with_halos, indices = indices[2])
    z = znodes(grid, ℓx, ℓy, ℓz; with_halos, indices = indices[3])

    if reshape
        # Here we have to deal with the fact that Flat directions may have
        # `nothing` nodes.
        #
        # A better solution (and more consistent with the rest of the API?)
        # might be to omit the `nothing` nodes in the `reshape`. In other words,
        # if `TX === Flat`, then we should return `(x, z)`. This is for future
        # consideration...
        #
        # See also `nodes` for `LatitudeLongitudeGrid`.

        Nx = isnothing(x) ? 1 : length(x)
        Ny = isnothing(y) ? 1 : length(y)
        Nz = isnothing(z) ? 1 : length(z)

        x = isnothing(x) ? zeros(1, 1, 1) : Base.reshape(x, Nx, 1, 1)
        y = isnothing(y) ? zeros(1, 1, 1) : Base.reshape(y, 1, Ny, 1)
        z = isnothing(z) ? zeros(1, 1, 1) : Base.reshape(z, 1, 1, Nz)
    end

    return (x, y, z)
end

@inline xnodes(grid::RG, ℓx::F; with_halos=false, indices=Colon()) = view(_property(grid.xᶠᵃᵃ, ℓx, topology(grid, 1), grid.Nx, grid.Hx, with_halos), indices)
@inline xnodes(grid::RG, ℓx::C; with_halos=false, indices=Colon()) = view(_property(grid.xᶜᵃᵃ, ℓx, topology(grid, 1), grid.Nx, grid.Hx, with_halos), indices)
@inline ynodes(grid::RG, ℓy::F; with_halos=false, indices=Colon()) = view(_property(grid.yᵃᶠᵃ, ℓy, topology(grid, 2), grid.Ny, grid.Hy, with_halos), indices)
@inline ynodes(grid::RG, ℓy::C; with_halos=false, indices=Colon()) = view(_property(grid.yᵃᶜᵃ, ℓy, topology(grid, 2), grid.Ny, grid.Hy, with_halos), indices)

# convenience
@inline xnodes(grid::RG, ℓx, ℓy, ℓz; with_halos=false, indices=Colon()) = xnodes(grid, ℓx; with_halos, indices)
@inline ynodes(grid::RG, ℓx, ℓy, ℓz; with_halos=false, indices=Colon()) = ynodes(grid, ℓy; with_halos, indices)

# Flat topologies
XFlatRG = RectilinearGrid{<:Any, Flat}
YFlatRG = RectilinearGrid{<:Any, <:Any, Flat}
ZFlatRG = RectilinearGrid{<:Any, <:Any, <:Any, Flat}
@inline xnodes(grid::XFlatRG, ℓx::F; with_halos=false, indices=Colon()) = _property(grid.xᶠᵃᵃ, ℓx, topology(grid, 1), grid.Nx, grid.Hx, with_halos)
@inline xnodes(grid::XFlatRG, ℓx::C; with_halos=false, indices=Colon()) = _property(grid.xᶜᵃᵃ, ℓx, topology(grid, 1), grid.Nx, grid.Hx, with_halos)
@inline ynodes(grid::YFlatRG, ℓy::F; with_halos=false, indices=Colon()) = _property(grid.yᵃᶠᵃ, ℓy, topology(grid, 2), grid.Ny, grid.Hy, with_halos)
@inline ynodes(grid::YFlatRG, ℓy::C; with_halos=false, indices=Colon()) = _property(grid.yᵃᶜᵃ, ℓy, topology(grid, 2), grid.Ny, grid.Hy, with_halos)

# Generalized coordinates
@inline ξnodes(grid::RG, ℓx; kwargs...) = xnodes(grid, ℓx; kwargs...)
@inline ηnodes(grid::RG, ℓy; kwargs...) = ynodes(grid, ℓy; kwargs...)

@inline ξnodes(grid::RG, ℓx, ℓy, ℓz; kwargs...) = xnodes(grid, ℓx; kwargs...)
@inline ηnodes(grid::RG, ℓx, ℓy, ℓz; kwargs...) = ynodes(grid, ℓy; kwargs...)

@inline isrectilinear(::RG) = true

#####
##### Grid-specific grid spacings
#####

@inline xspacings(grid::RG, ℓx) = xspacings(grid, ℓx, nothing, nothing)
@inline yspacings(grid::RG, ℓy) = yspacings(grid, nothing, ℓy, nothing)
