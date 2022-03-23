using KernelAbstractions: @kernel, @index

const R_Earth = 6371.0e3    # Mean radius of the Earth [m] https://en.wikipedia.org/wiki/Earth

struct LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, Arch} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Arch}
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
    Δλᶠᵃᵃ :: FX
    Δλᶜᵃᵃ :: FX
    λᶠᵃᵃ  :: VX
    λᶜᵃᵃ  :: VX
    Δφᵃᶠᵃ :: FY
    Δφᵃᶜᵃ :: FY
    φᵃᶠᵃ  :: VY
    φᵃᶜᵃ  :: VY
    Δzᵃᵃᶠ :: FZ 
    Δzᵃᵃᶜ :: FZ
    zᵃᵃᶠ  :: VZ
    zᵃᵃᶜ  :: VZ
    # Precomputed metrics M <: Nothing means metrics will be computed on the fly
    Δxᶠᶜᵃ :: M
    Δxᶜᶠᵃ :: M
    Δxᶠᶠᵃ :: M
    Δxᶜᶜᵃ :: M
    Δyᶠᶜᵃ :: MY
    Δyᶜᶠᵃ :: MY
    Azᶠᶜᵃ :: M
    Azᶜᶠᵃ :: M
    Azᶠᶠᵃ :: M
    Azᶜᶜᵃ :: M
    # Spherical radius
    radius :: FT

    function LatitudeLongitudeGrid{TX, TY, TZ}(architecture::Arch,
                                               Nλ, Nφ, Nz,
                                               Hλ, Hφ, Hz,
                                               Lλ::FT, Lφ::FT, Lz::FT,
                                               Δλᶠᵃᵃ :: FX, Δλᶜᵃᵃ :: FX,
                                                λᶠᵃᵃ :: VX,  λᶜᵃᵃ :: VX,
                                               Δφᵃᶠᵃ :: FY, Δφᵃᶜᵃ :: FY,
                                                φᵃᶠᵃ :: VY,  φᵃᶜᵃ :: VY,
                                               Δzᵃᵃᶠ :: FZ, Δzᵃᵃᶜ :: FZ,
                                                zᵃᵃᶠ :: VZ,  zᵃᵃᶜ :: VZ,
                                               Δxᶠᶜ::M, Δxᶜᶠ::M,
                                               Δxᶠᶠ::M, Δxᶜᶜ::M,
                                               Δyᶠᶜ::MY, Δyᶜᶠ::MY,
                                               Azᶠᶜ::M, Azᶜᶠ::M, Azᶠᶠ::M, Azᶜᶜ::M,
                                               radius::FT) where {Arch, FT,
                                                                  TX, TY, TZ,
                                                                  FX, FY, FZ,
                                                                  VX, VY, VZ,
                                                                  M, MY}

        return new{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, Arch}(architecture,
                                                                        Nλ, Nφ, Nz,
                                                                        Hλ, Hφ, Hz,
                                                                        Lλ, Lφ, Lz,
                                                                        Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                                        Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                                        Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
                                                                        Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ,
                                                                        Δyᶠᶜ, Δyᶜᶠ,
                                                                        Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ, radius)
    end
end

const XRegLatLonGrid = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const YRegLatLonGrid = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,    <:Number}
const ZRegLatLonGrid = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,    <:Any, <:Number}
const HRegLatLonGrid = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number, <:Number}

regular_dimensions(::ZRegLatLonGrid) = tuple(3)

"""
    LatitudeLongitudeGrid([architecture = CPU(), FT = Float64];
                          size,
                          longitude,
                          latitude,
                          z,
                          radius = R_Earth,
                          topology = nothing,
                          precompute_metrics = true,
                          halo = (1, 1, 1))

Creates a `LatitudeLongitudeGrid` with coordinates `(λ, φ, z)` denoting longitude, latitude,
and vertical coordinate respectively.

Positional arguments
====================

- `architecture`: Specifies whether arrays of coordinates and spacings are stored
                  on the CPU or GPU. Default: `CPU()`.

- `FT` : Floating point data type. Default: `Float64`.

Keyword arguments
=================

- `size` (required): A 3-tuple prescribing the number of grid points each direction.

- `longitude`, `latitude`, `z` (required): Each is either a
                                           (i) 2-tuple that specify the end points of the domain,
                                           (ii) one-dimensional array specifying the cell interface locations or
                                           (iii) a single-argument function that takes an index and returns
                                                 cell interface location.
  **Note**: the latitude and longitude coordinates extents are expected in degrees.

- `radius`: The radius of the sphere the grid lives on. By default is equal to the radius of Earth.

- `topology`: Tuple of topologies (`Flat`, `Bounded`, `Periodic`) for each direction. The vertical 
              `topology[3]` must be `Bounded`, while the latitude-longitude topologies can be
              `Bounded`, `Periodic`, or `Flat`.

- `precompute_metrics`: Boolean specifying whether to precompute horizontal spacings and areas.
                        Default: `true`. When `false`, horizontal spacings and areas are computed
                        on-the-fly during a simulation.

- `halo`: A 3-tuple of integers specifying the size of the halo region of cells surrounding
          the physical interior.
"""
function LatitudeLongitudeGrid(architecture::AbstractArchitecture = CPU(),
                               FT::DataType = Float64;
                               size,
                               longitude,
                               latitude,
                               z,
                               radius = R_Earth,
                               topology = nothing,
                               precompute_metrics = true,
                               halo = (1, 1, 1))

    Nλ, Nφ, Nz, Hλ, Hφ, Hz, latitude, longitude, topology =
        validate_lat_lon_grid_args(latitude, longitude, size, halo, topology)
    
    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real}, 
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    TX, TY, TZ = topology
    
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, TX, Nλ, Hλ, longitude, architecture)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, TY, Nφ, Hφ, latitude,  architecture)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, TZ, Nz, Hz, z,         architecture)

    preliminary_grid = LatitudeLongitudeGrid{TX, TY, TZ}(architecture,
                                                         Nλ, Nφ, Nz,
                                                         Hλ, Hφ, Hz,
                                                         Lλ, Lφ, Lz,
                                                         Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                         Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                         Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
                                                         (nothing for i=1:10)..., FT(radius))

    return !precompute_metrics ? preliminary_grid : with_precomputed_metrics(preliminary_grid)
end

# architecture = CPU() default, assuming that a DataType positional arg
# is specifying the floating point type.
LatitudeLongitudeGrid(FT::DataType; kwargs...) = LatitudeLongitudeGrid(CPU(), FT; kwargs...)

""" Return a reproduction of `grid` with precomputed metric terms. """
function with_precomputed_metrics(grid)
    arch = architecture(grid)
    Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Δyᶠᶜ, Δyᶜᶠ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ = allocate_metrics(grid)
    wait(device_event(arch))

    precompute_curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ )
    wait(device_event(arch))

    Δyᶠᶜ, Δyᶜᶠ = precompute_Δy_metrics(grid, Δyᶠᶜ, Δyᶜᶠ)

    Nλ, Nφ, Nz = size(grid)
    Hλ, Hφ, Hz = halo_size(grid)
    TX, TY, TZ = topology(grid)

    return LatitudeLongitudeGrid{TX, TY, TZ}(architecture(grid),
                                             Nλ, Nφ, Nz,
                                             Hλ, Hφ, Hz,
                                             grid.Lx, grid.Ly, grid.Lz,
                                             grid.Δλᶠᵃᵃ, grid.Δλᶜᵃᵃ, grid.λᶠᵃᵃ, grid.λᶜᵃᵃ,
                                             grid.Δφᵃᶠᵃ, grid.Δφᵃᶜᵃ, grid.φᵃᶠᵃ, grid.φᵃᶜᵃ,
                                             grid.Δzᵃᵃᶠ, grid.Δzᵃᵃᶜ, grid.zᵃᵃᶠ, grid.zᵃᵃᶜ,
                                             Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ,
                                             Δyᶠᶜ, Δyᶜᶠ,
                                             Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ, grid.radius)
end

function validate_lat_lon_grid_args(latitude, longitude, size, halo, topology)

    λ₁, λ₂ = get_domain_extent(longitude, size[1])
    @assert λ₁ < λ₂ && λ₂ - λ₁ ≤ 360

    φ₁, φ₂ = get_domain_extent(latitude, size[2])
    @assert -90 <= φ₁ < φ₂ <= 90

    (φ₁ == -90 || φ₂ == 90) &&
        @warn "Are you sure you want to use a latitude-longitude grid with a grid point at the pole?"

    Lλ = λ₂ - λ₁

    if !isnothing(topology)
        TX, TY, TZ = topology
        TZ === Bounded || throw(ArgumentError("z topology must be Bounded"))
    else
        TX = Lλ == 360 ? Periodic : Bounded
        TY = Bounded
        TZ = Bounded
    end

    Nλ, Nφ, Nz = N = validate_size(TX, TY, TZ, size)
    Hλ, Hφ, Hz = H = validate_halo(TX, TY, TZ, halo)

    return Nλ, Nφ, Nz, Hλ, Hφ, Hz, latitude, longitude, (TX, TY, TZ)
end

function Base.summary(grid::LatitudeLongitudeGrid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)
    metric_computation = isnothing(grid.Δxᶠᶜᵃ) ? "without precomputed metrics" : "with precomputed metrics"

    return string(size_summary(size(grid)),
                  " LatitudeLongitudeGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo",
                  " and ", metric_computation)
end

function Base.show(io::IO, grid::LatitudeLongitudeGrid)
    TX, TY, TZ = topology(grid)

    λ₁, λ₂ = domain(topology(grid, 1), grid.Nx, grid.λᶠᵃᵃ)
    φ₁, φ₂ = domain(topology(grid, 2), grid.Ny, grid.φᵃᶠᵃ)
    z₁, z₂ = domain(topology(grid, 3), grid.Nz, grid.zᵃᵃᶠ)

    x_summary = domain_summary(TX(), "λ", λ₁, λ₂)
    y_summary = domain_summary(TY(), "φ", φ₁, φ₂)
    z_summary = domain_summary(TZ(), "z", z₁, z₂)

    longest = max(length(x_summary), length(y_summary), length(z_summary)) 

    x_summary = "longitude: " * dimension_summary(TX(), "λ", λ₁, λ₂, grid.Δλᶜᵃᵃ, longest - length(x_summary))
    y_summary = "latitude:  " * dimension_summary(TY(), "φ", φ₁, φ₂, grid.Δφᵃᶜᵃ, longest - length(y_summary))
    z_summary = "z:         " * dimension_summary(TZ(), "z", z₁, z₂, grid.Δzᵃᵃᶜ, longest - length(z_summary))

    print(io, summary(grid), '\n',
          "├── ", x_summary, '\n',
          "├── ", y_summary, '\n',
          "└── ", z_summary)
end

# Node by node
@inline xnode(::Face,   i, grid::LatitudeLongitudeGrid) = @inbounds grid.λᶠᵃᵃ[i]
@inline xnode(::Center, i, grid::LatitudeLongitudeGrid) = @inbounds grid.λᶜᵃᵃ[i]
@inline ynode(::Face,   j, grid::LatitudeLongitudeGrid) = @inbounds grid.φᵃᶠᵃ[j]
@inline ynode(::Center, j, grid::LatitudeLongitudeGrid) = @inbounds grid.φᵃᶜᵃ[j]
@inline znode(::Face,   k, grid::LatitudeLongitudeGrid) = @inbounds grid.zᵃᵃᶠ[k]
@inline znode(::Center, k, grid::LatitudeLongitudeGrid) = @inbounds grid.zᵃᵃᶜ[k]

all_x_nodes(::Type{Face},   grid::LatitudeLongitudeGrid) = grid.λᶠᵃᵃ
all_x_nodes(::Type{Center}, grid::LatitudeLongitudeGrid) = grid.λᶜᵃᵃ
all_y_nodes(::Type{Face},   grid::LatitudeLongitudeGrid) = grid.φᵃᶠᵃ
all_y_nodes(::Type{Center}, grid::LatitudeLongitudeGrid) = grid.φᵃᶜᵃ
all_z_nodes(::Type{Face},   grid::LatitudeLongitudeGrid) = grid.zᵃᵃᶠ
all_z_nodes(::Type{Center}, grid::LatitudeLongitudeGrid) = grid.zᵃᵃᶜ

@inline x_domain(grid::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TX, grid.Nx, grid.λᶠᵃᵃ)
@inline y_domain(grid::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TY, grid.Ny, grid.φᵃᶠᵃ)
@inline z_domain(grid::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)

@inline cpu_face_constructor_x(grid::XRegLatLonGrid) = x_domain(grid)
@inline cpu_face_constructor_y(grid::YRegLatLonGrid) = y_domain(grid)
@inline cpu_face_constructor_z(grid::ZRegLatLonGrid) = z_domain(grid)

function with_halo(new_halo, old_grid::LatitudeLongitudeGrid)

    size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)
    topo = topology(old_grid)

    x = cpu_face_constructor_x(old_grid)
    y = cpu_face_constructor_y(old_grid)
    z = cpu_face_constructor_z(old_grid)

    # Remove elements of size and new_halo in Flat directions as expected by grid
    # constructor
    size     = pop_flat_elements(size, topo)
    new_halo = pop_flat_elements(new_halo, topo)

    new_grid = LatitudeLongitudeGrid(architecture(old_grid), eltype(old_grid);
                                     size = size, halo = new_halo,
                                     longitude = x, latitude = y, z = z,
                                     precompute_metrics = metrics_precomputed(old_grid))

    return new_grid
end

function on_architecture(new_arch::AbstractArchitecture, old_grid::LatitudeLongitudeGrid)
    old_properties = (old_grid.Δλᶠᵃᵃ, old_grid.Δλᶜᵃᵃ, old_grid.λᶠᵃᵃ,  old_grid.λᶜᵃᵃ,
                      old_grid.Δφᵃᶠᵃ, old_grid.Δφᵃᶜᵃ, old_grid.φᵃᶠᵃ,  old_grid.φᵃᶜᵃ,
                      old_grid.Δzᵃᵃᶠ, old_grid.Δzᵃᵃᶜ, old_grid.zᵃᵃᶠ,  old_grid.zᵃᵃᶜ,
                      old_grid.Δxᶠᶜᵃ, old_grid.Δxᶜᶠᵃ, old_grid.Δxᶠᶠᵃ, old_grid.Δxᶜᶜᵃ,
                      old_grid.Δyᶠᶜᵃ, old_grid.Δyᶜᶠᵃ,
                      old_grid.Azᶠᶜᵃ, old_grid.Azᶜᶠᵃ, old_grid.Azᶠᶠᵃ, old_grid.Azᶜᶜᵃ)

    new_properties = Tuple(arch_array(new_arch, p) for p in old_properties)

    TX, TY, TZ = topology(old_grid)

    return LatitudeLongitudeGrid{TX, TY, TZ}(new_arch,
                                             old_grid.Nx, old_grid.Ny, old_grid.Nz,
                                             old_grid.Hx, old_grid.Hy, old_grid.Hz,
                                             old_grid.Lx, old_grid.Ly, old_grid.Lz,
                                             new_properties...,
                                             old_grid.radius)
end

function Adapt.adapt_structure(to, grid::LatitudeLongitudeGrid)
    TX, TY, TZ = topology(grid)
    return LatitudeLongitudeGrid{TX, TY, TZ}(nothing,
                                             grid.Nx, grid.Ny, grid.Nz,
                                             grid.Hx, grid.Hy, grid.Hz,
                                             grid.Lx, grid.Ly, grid.Lz,
                                             Adapt.adapt(to, grid.Δλᶠᵃᵃ),
                                             Adapt.adapt(to, grid.Δλᶜᵃᵃ),
                                             Adapt.adapt(to, grid.λᶠᵃᵃ),
                                             Adapt.adapt(to, grid.λᶜᵃᵃ),
                                             Adapt.adapt(to, grid.Δφᵃᶠᵃ),
                                             Adapt.adapt(to, grid.Δφᵃᶜᵃ),
                                             Adapt.adapt(to, grid.φᵃᶠᵃ),
                                             Adapt.adapt(to, grid.φᵃᶜᵃ),
                                             Adapt.adapt(to, grid.Δzᵃᵃᶠ),
                                             Adapt.adapt(to, grid.Δzᵃᵃᶜ),
                                             Adapt.adapt(to, grid.zᵃᵃᶠ),
                                             Adapt.adapt(to, grid.zᵃᵃᶜ),
                                             Adapt.adapt(to, grid.Δxᶠᶜᵃ),
                                             Adapt.adapt(to, grid.Δxᶜᶠᵃ),
                                             Adapt.adapt(to, grid.Δxᶠᶠᵃ),
                                             Adapt.adapt(to, grid.Δxᶜᶜᵃ),
                                             Adapt.adapt(to, grid.Δyᶠᶜᵃ),
                                             Adapt.adapt(to, grid.Δyᶜᶠᵃ),
                                             Adapt.adapt(to, grid.Azᶠᶜᵃ),
                                             Adapt.adapt(to, grid.Azᶜᶠᵃ),
                                             Adapt.adapt(to, grid.Azᶠᶠᵃ),
                                             Adapt.adapt(to, grid.Azᶜᶜᵃ),
                                             grid.radius)
end

#####
##### On-the-fly computation of LatitudeLongitudeGrid metrics
#####

@inline hack_cosd(φ) = cos(π * φ / 180)
@inline hack_sind(φ) = sin(π * φ / 180)

@inline Δxᶠᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ[i])
@inline Δxᶜᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ[i])
@inline Δxᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ[i])
@inline Δxᶜᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ[i])
@inline Δyᶜᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
@inline Δyᶠᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])
@inline Azᶠᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))

@inline Δxᶠᶜᵃ(i, j, k, grid::XRegLatLonGrid) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ)
@inline Δxᶜᶠᵃ(i, j, k, grid::XRegLatLonGrid) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ)   
@inline Δxᶠᶠᵃ(i, j, k, grid::XRegLatLonGrid) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ)
@inline Δxᶜᶜᵃ(i, j, k, grid::XRegLatLonGrid) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ)   
@inline Δyᶜᶠᵃ(i, j, k, grid::YRegLatLonGrid) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ)
@inline Δyᶠᶜᵃ(i, j, k, grid::YRegLatLonGrid) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ)
@inline Azᶠᶜᵃ(i, j, k, grid::XRegLatLonGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::XRegLatLonGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::XRegLatLonGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::XRegLatLonGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))

#####
##### Utilities to precompute metrics 
#####

@inline metrics_precomputed(::LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, Nothing}) = false 
@inline metrics_precomputed(::LatitudeLongitudeGrid) = true

#####
##### Kernels that precompute the z- and x-metric
#####

@inline metric_worksize(grid::LatitudeLongitudeGrid)  = (length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ) - 1) 
@inline metric_workgroup(grid::LatitudeLongitudeGrid) = (16, 16) 

@inline metric_worksize(grid::XRegLatLonGrid)  =  length(grid.φᵃᶜᵃ) - 1 
@inline metric_workgroup(grid::XRegLatLonGrid) =  16

function precompute_curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    
    arch = grid.architecture

    workgroup, worksize  = metric_workgroup(grid), metric_worksize(grid)
    curvilinear_metrics! = precompute_metrics_kernel!(Architectures.device(arch), workgroup, worksize)

    event = curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ; dependencies=device_event(arch))
    wait(event)

    return nothing
end

@kernel function precompute_metrics_kernel!(grid::LatitudeLongitudeGrid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    i, j = @index(Global, NTuple)

    # Manually offset x- and y-index
    i′ = i + grid.Δλᶜᵃᵃ.offsets[1] 
    j′ = j + grid.φᵃᶜᵃ.offsets[1] + 1

    @inbounds begin
        Δxᶠᶜ[i′, j′] = Δxᶠᶜᵃ(i′, j′, 1, grid)
        Δxᶜᶠ[i′, j′] = Δxᶜᶠᵃ(i′, j′, 1, grid)
        Δxᶠᶠ[i′, j′] = Δxᶠᶠᵃ(i′, j′, 1, grid)
        Δxᶜᶜ[i′, j′] = Δxᶜᶜᵃ(i′, j′, 1, grid)
        Azᶠᶜ[i′, j′] = Azᶠᶜᵃ(i′, j′, 1, grid)
        Azᶜᶠ[i′, j′] = Azᶜᶠᵃ(i′, j′, 1, grid)
        Azᶠᶠ[i′, j′] = Azᶠᶠᵃ(i′, j′, 1, grid)
        Azᶜᶜ[i′, j′] = Azᶜᶜᵃ(i′, j′, 1, grid)
    end
end

@kernel function precompute_metrics_kernel!(grid::XRegLatLonGrid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    j = @index(Global, Linear)

    # Manually offset y-index
    j′ = j + grid.φᵃᶜᵃ.offsets[1] + 1

    @inbounds begin
        Δxᶠᶜ[j′] = Δxᶠᶜᵃ(1, j′, 1, grid)
        Δxᶜᶠ[j′] = Δxᶜᶠᵃ(1, j′, 1, grid)
        Δxᶠᶠ[j′] = Δxᶠᶠᵃ(1, j′, 1, grid)
        Δxᶜᶜ[j′] = Δxᶜᶜᵃ(1, j′, 1, grid)
        Azᶠᶜ[j′] = Azᶠᶜᵃ(1, j′, 1, grid)
        Azᶜᶠ[j′] = Azᶜᶠᵃ(1, j′, 1, grid)
        Azᶠᶠ[j′] = Azᶠᶠᵃ(1, j′, 1, grid)
        Azᶜᶜ[j′] = Azᶜᶜᵃ(1, j′, 1, grid)
    end
end

#####
##### Kernels that precompute the y-metric
#####

function precompute_Δy_metrics(grid::LatitudeLongitudeGrid, Δyᶠᶜ, Δyᶜᶠ)
    arch = grid.architecture
    precompute_Δy! = precompute_Δy_kernel!(Architectures.device(arch), 16, length(grid.Δφᵃᶜᵃ) - 1)
    event = precompute_Δy!(grid, Δyᶠᶜ, Δyᶜᶠ; dependencies=device_event(arch))
    
    wait(event)
    return Δyᶠᶜ, Δyᶜᶠ
end

function  precompute_Δy_metrics(grid::YRegLatLonGrid, Δyᶠᶜ, Δyᶜᶠ)
    Δyᶜᶠ =  Δyᶜᶠᵃ(1, 1, 1, grid)
    Δyᶠᶜ =  Δyᶠᶜᵃ(1, 1, 1, grid)
    return Δyᶠᶜ, Δyᶜᶠ
end

@kernel function precompute_Δy_kernel!(grid, Δyᶠᶜ, Δyᶜᶠ)
    j  = @index(Global, Linear)

    # Manually offset y-index
    j′ = j + grid.Δφᵃᶜᵃ.offsets[1] + 1

    @inbounds begin
        Δyᶜᶠ[j′] = Δyᶜᶠᵃ(1, j′, 1, grid)
        Δyᶠᶜ[j′] = Δyᶜᶠᵃ(1, j′, 1, grid)
    end
end

#####
##### Metric memory allocation
#####

function allocate_metrics(grid::LatitudeLongitudeGrid)
    FT = eltype(grid)
    
    # preallocate quantities to ensure correct type and size
    grid_metrics = (:Δxᶠᶜ,
                    :Δxᶜᶠ,
                    :Δxᶠᶠ,
                    :Δxᶜᶜ,
                    :Azᶠᶜ,
                    :Azᶜᶠ,
                    :Azᶠᶠ,
                    :Azᶜᶜ)

    arch = grid.architecture
    
    if typeof(grid) <: XRegLatLonGrid
        offsets     = grid.φᵃᶜᵃ.offsets[1]
        metric_size = length(grid.φᵃᶜᵃ)
    else    
        offsets     = (grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
        metric_size = (length(grid.Δλᶜᵃᵃ)   , length(grid.φᵃᶜᵃ))
    end

    for metric in grid_metrics
        parentM        = Symbol(metric, :_parent)
        @eval $parentM = zeros($FT, $metric_size...)
        @eval $metric  = OffsetArray(arch_array($arch, $parentM), $offsets...)
    end

    if typeof(grid) <: YRegLatLonGrid
        Δyᶠᶜ = FT(0.0)
        Δyᶜᶠ = FT(0.0)
    else    
        parentC = zeros(FT, length(grid.Δφᵃᶜᵃ))
        parentF = zeros(FT, length(grid.Δφᵃᶜᵃ))
        Δyᶠᶜ    = OffsetArray(arch_array(arch, parentC), grid.Δφᵃᶜᵃ.offsets[1])
        Δyᶜᶠ    = OffsetArray(arch_array(arch, parentF), grid.Δφᵃᶜᵃ.offsets[1])
    end
    
    return Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Δyᶠᶜ, Δyᶜᶠ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ
end
