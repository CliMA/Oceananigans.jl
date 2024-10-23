using KernelAbstractions: @kernel, @index

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

    LatitudeLongitudeGrid{TX, TY, TZ}(architecture::Arch,
                                      Nλ, Nφ, Nz,
                                      Hλ, Hφ, Hz,
                                      Lλ :: FT, Lφ :: FT, Lz :: FT,
                                      Δλᶠᵃᵃ :: FX, Δλᶜᵃᵃ :: FX,
                                       λᶠᵃᵃ :: VX,  λᶜᵃᵃ :: VX,
                                      Δφᵃᶠᵃ :: FY, Δφᵃᶜᵃ :: FY,
                                       φᵃᶠᵃ :: VY,  φᵃᶜᵃ :: VY,
                                      Δzᵃᵃᶠ :: FZ, Δzᵃᵃᶜ :: FZ,
                                       zᵃᵃᶠ :: VZ,  zᵃᵃᶜ :: VZ,
                                      Δxᶠᶜᵃ :: M,  Δxᶜᶠᵃ :: M,
                                      Δxᶠᶠᵃ :: M,  Δxᶜᶜᵃ :: M,
                                      Δyᶠᶜᵃ :: MY, Δyᶜᶠᵃ :: MY,
                                      Azᶠᶜᵃ :: M,  Azᶜᶠᵃ :: M,
                                      Azᶠᶠᵃ :: M,  Azᶜᶜᵃ :: M,
                                      radius :: FT) where {Arch, FT, TX, TY, TZ,
                                                           FX, FY, FZ, VX, VY, VZ,
                                                           M, MY} =
    new{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, Arch}(architecture,
                                                             Nλ, Nφ, Nz,
                                                             Hλ, Hφ, Hz,
                                                             Lλ, Lφ, Lz,
                                                             Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                             Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                             Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
                                                             Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ,
                                                             Δyᶠᶜᵃ, Δyᶜᶠᵃ,
                                                             Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ, radius)
end

const LLG = LatitudeLongitudeGrid
const XRegularLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const YRegularLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,    <:Number}
const ZRegularLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,    <:Any,    <:Number}
const HRegularLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number, <:Number}
const HNonRegularLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray, <:AbstractArray}
const YNonRegularLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number, <:AbstractArray}

regular_dimensions(::ZRegularLLG) = tuple(3)

"""
    LatitudeLongitudeGrid([architecture = CPU(), FT = Float64];
                          size,
                          longitude,
                          latitude,
                          z = nothing,
                          radius = R_Earth,
                          topology = nothing,
                          precompute_metrics = true,
                          halo = nothing)

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

- `longitude` (required), `latitude` (required), `z` (default: `nothing`):
  Each is either a:
  1. 2-tuple that specify the end points of the domain,
  2. one-dimensional array specifying the cell interface locations, or
  3. a single-argument function that takes an index and returns cell interface location.

  **Note**: the latitude and longitude coordinates extents are expected in degrees.

- `radius`: The radius of the sphere the grid lives on. By default is equal to the radius of Earth.

- `topology`: Tuple of topologies (`Flat`, `Bounded`, `Periodic`) for each direction. The vertical
              `topology[3]` must be `Bounded`, while the latitude-longitude topologies can be
              `Bounded`, `Periodic`, or `Flat`. If no topology is provided then, by default, the
              topology is (`Periodic`, `Bounded`, `Bounded`) if the latitudinal extent is 360 degrees
              or (`Bounded`, `Bounded`, `Bounded`) otherwise.

- `precompute_metrics`: Boolean specifying whether to precompute horizontal spacings and areas.
                        Default: `true`. When `false`, horizontal spacings and areas are computed
                        on-the-fly during a simulation.

- `halo`: A 3-tuple of integers specifying the size of the halo region of cells surrounding
          the physical interior. The default is 3 halo cells in every direction.

Examples
========

* A default grid with `Float64` type:

```jldoctest
julia> using Oceananigans

julia> grid = LatitudeLongitudeGrid(size=(36, 34, 25),
                                    longitude = (-180, 180),
                                    latitude = (-85, 85),
                                    z = (-1000, 0))
36×34×25 LatitudeLongitudeGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo and with precomputed metrics
├── longitude: Periodic λ ∈ [-180.0, 180.0) regularly spaced with Δλ=10.0
├── latitude:  Bounded  φ ∈ [-85.0, 85.0]   regularly spaced with Δφ=5.0
└── z:         Bounded  z ∈ [-1000.0, 0.0]  regularly spaced with Δz=40.0
```

* A bounded spherical sector with cell interfaces stretched hyperbolically near the top:

```jldoctest
julia> using Oceananigans

julia> σ = 1.1; # stretching factor

julia> Nz = 24; # vertical resolution

julia> Lz = 1000; # depth (m)

julia> hyperbolically_spaced_faces(k) = - Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));

julia> grid = LatitudeLongitudeGrid(size=(36, 34, Nz),
                                    longitude = (-180, 180),
                                    latitude = (-20, 20),
                                    z = hyperbolically_spaced_faces,
                                    topology = (Bounded, Bounded, Bounded))
36×34×24 LatitudeLongitudeGrid{Float64, Bounded, Bounded, Bounded} on CPU with 3×3×3 halo and with precomputed metrics
├── longitude: Bounded  λ ∈ [-180.0, 180.0] regularly spaced with Δλ=10.0
├── latitude:  Bounded  φ ∈ [-20.0, 20.0]   regularly spaced with Δφ=1.17647
└── z:         Bounded  z ∈ [-1000.0, -0.0] variably spaced with min(Δz)=21.3342, max(Δz)=57.2159
```
"""
function LatitudeLongitudeGrid(architecture::AbstractArchitecture = CPU(),
                               FT::DataType = Float64;
                               size,
                               longitude = nothing,
                               latitude = nothing,
                               z = nothing,
                               radius = R_Earth,
                               topology = nothing,
                               precompute_metrics = true,
                               halo = nothing)

    if architecture == GPU() && !has_cuda()
        throw(ArgumentError("Cannot create a GPU grid. No CUDA-enabled GPU was detected!"))
    end

    topology, size, halo, latitude, longitude, z, precompute_metrics =
        validate_lat_lon_grid_args(topology, size, halo, FT, latitude, longitude, z, precompute_metrics)

    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo

    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real},
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    TX, TY, TZ = topology

    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, TX(), Nλ, Hλ, longitude, :longitude, architecture)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, TY(), Nφ, Hφ, latitude,  :latitude,  architecture)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, TZ(), Nz, Hz, z,         :z,         architecture)

    preliminary_grid = LatitudeLongitudeGrid{TX, TY, TZ}(architecture,
                                                         Nλ, Nφ, Nz,
                                                         Hλ, Hφ, Hz,
                                                         Lλ, Lφ, Lz,
                                                         Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                         Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                         Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
                                                         (nothing for i=1:10)..., FT(radius))

    if !precompute_metrics
        return preliminary_grid
    else
        return with_precomputed_metrics(preliminary_grid)
    end
end

# architecture = CPU() default, assuming that a DataType positional arg
# is specifying the floating point type.
LatitudeLongitudeGrid(FT::DataType; kwargs...) = LatitudeLongitudeGrid(CPU(), FT; kwargs...)

""" Return a reproduction of `grid` with precomputed metric terms. """
function with_precomputed_metrics(grid)
    Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ = allocate_metrics(grid)

    # Compute Δx's and areas
    arch = grid.architecture
    dev = Architectures.device(arch)
    workgroup, worksize  = metric_workgroup(grid), metric_worksize(grid)
    loop! = compute_Δx_Az!(dev, workgroup, worksize)
    loop!(grid, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ)

    # Compute Δy's if needed
    if !(grid isa YRegularLLG)
        loop! = compute_Δy!(dev, 16, length(grid.Δφᵃᶜᵃ) - 1)
        loop!(grid, Δyᶠᶜᵃ, Δyᶜᶠᵃ)
    end

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
                                             Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ,
                                             Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ, grid.radius)
end

function validate_lat_lon_grid_args(topology, size, halo, FT, latitude, longitude, z, precompute_metrics)
    if !isnothing(topology)
        TX, TY, TZ = validate_topology(topology)
        Nλ, Nφ, Nz = size = validate_size(TX, TY, TZ, size)
    else # Set default topology according to longitude
        Nλ, Nφ, Nz = size # using default topology, does not support Flat
        λ₁, λ₂ = get_domain_extent(longitude, Nλ)

        Lλ = λ₂ - λ₁
        TX = Lλ == 360 ? Periodic : Bounded
        TY = Bounded
        TZ = Bounded
    end

    if TY() isa Periodic
        throw(ArgumentError("LatitudeLongitudeGrid cannot be Periodic in latitude!"))
    end

    # Validate longitude and latitude
    λ₁, λ₂ = get_domain_extent(longitude, Nλ)
    λ₂ - λ₁ ≤ 360 || throw(ArgumentError("Longitudinal extent cannot be greater than 360 degrees."))
    λ₁ <= λ₂      || throw(ArgumentError("Longitudes must increase west to east."))

    φ₁, φ₂ = get_domain_extent(latitude, Nφ)
    -90 <= φ₁ || throw(ArgumentError("The southernmost latitude cannot be less than -90 degrees."))
    φ₂ <= 90  || throw(ArgumentError("The northern latitude cannot be less than -90 degrees."))
    φ₁ <= φ₂  || throw(ArgumentError("Latitudes must increase south to north."))

    if TX == Flat || TY == Flat
        precompute_metrics = false
    end

    longitude = validate_dimension_specification(TX, longitude, :longitude, Nλ, FT)
    latitude  = validate_dimension_specification(TY, latitude,  :latitude,  Nφ, FT)
    z         = validate_dimension_specification(TZ, z,         :z,         Nz, FT)

    halo = validate_halo(TX, TY, TZ, size, halo)
    topology = (TX, TY, TZ)

    return topology, size, halo, latitude, longitude, z, precompute_metrics
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

function Base.show(io::IO, grid::LatitudeLongitudeGrid, withsummary=true)
    TX, TY, TZ = topology(grid)

    Ωλ = domain(TX(), size(grid, 1), grid.λᶠᵃᵃ)
    Ωφ = domain(TY(), size(grid, 2), grid.φᵃᶠᵃ)
    Ωz = domain(TZ(), size(grid, 3), grid.zᵃᵃᶠ)

    x_summary = domain_summary(TX(), "λ", Ωλ)
    y_summary = domain_summary(TY(), "φ", Ωφ)
    z_summary = domain_summary(TZ(), "z", Ωz)

    longest = max(length(x_summary), length(y_summary), length(z_summary))

    x_summary = "longitude: " * dimension_summary(TX(), "λ", Ωλ, grid.Δλᶜᵃᵃ, longest - length(x_summary))
    y_summary = "latitude:  " * dimension_summary(TY(), "φ", Ωφ, grid.Δφᵃᶜᵃ, longest - length(y_summary))
    z_summary = "z:         " * dimension_summary(TZ(), "z", Ωz, grid.Δzᵃᵃᶜ, longest - length(z_summary))

    if withsummary
        print(io, summary(grid), "\n")
    end

    return print(io, "├── ", x_summary, "\n",
                     "├── ", y_summary, "\n",
                     "└── ", z_summary)
end

@inline x_domain(grid::LLG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TX, grid.Nx, grid.λᶠᵃᵃ)
@inline y_domain(grid::LLG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TY, grid.Ny, grid.φᵃᶠᵃ)
@inline z_domain(grid::LLG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)

@inline cpu_face_constructor_x(grid::XRegularLLG) = x_domain(grid)
@inline cpu_face_constructor_y(grid::YRegularLLG) = y_domain(grid)
@inline cpu_face_constructor_z(grid::ZRegularLLG) = z_domain(grid)

function constructor_arguments(grid::LatitudeLongitudeGrid)
    arch = architecture(grid)
    FT = eltype(grid)
    args = Dict(:architecture => arch, :number_type => eltype(grid))

    # Kwargs
    topo = topology(grid)
    size = (grid.Nx, grid.Ny, grid.Nz)
    halo = (grid.Hx, grid.Hy, grid.Hz)
    size = pop_flat_elements(size, topo)
    halo = pop_flat_elements(halo, topo)

    kwargs = Dict(:size => size,
                  :halo => halo,
                  :longitude => cpu_face_constructor_x(grid),
                  :latitude => cpu_face_constructor_y(grid),
                  :z => cpu_face_constructor_z(grid),
                  :topology => topo,
                  :radius => grid.radius,
                  :precompute_metrics => metrics_precomputed(grid))

    return args, kwargs
end

function Base.similar(grid::LatitudeLongitudeGrid)
    args, kwargs = constructor_arguments(grid)
    arch = args[:architecture]
    FT = args[:number_type]
    return LatitudeLongitudeGrid(arch, FT; kwargs...)
end

function with_number_type(FT, grid::LatitudeLongitudeGrid)
    args, kwargs = constructor_arguments(grid)
    arch = args[:architecture]
    return LatitudeLongitudeGrid(arch, FT; kwargs...)
end

function with_halo(halo, grid::LatitudeLongitudeGrid)
    args, kwargs = constructor_arguments(grid)
    halo = pop_flat_elements(halo, topology(grid))
    kwargs[:halo] = halo
    arch = args[:architecture]
    FT = args[:number_type]
    return LatitudeLongitudeGrid(arch, FT; kwargs...)
end

function on_architecture(arch::AbstractSerialArchitecture, grid::LatitudeLongitudeGrid)
    if arch == architecture(grid)
        return grid
    end

    args, kwargs = constructor_arguments(grid)
    FT = args[:number_type]
    return LatitudeLongitudeGrid(arch, FT; kwargs...)
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

@inline Δxᶠᶜᵃ(i, j, k, grid::XRegularLLG) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ)
@inline Δxᶜᶠᵃ(i, j, k, grid::XRegularLLG) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ)
@inline Δxᶠᶠᵃ(i, j, k, grid::XRegularLLG) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ)
@inline Δxᶜᶜᵃ(i, j, k, grid::XRegularLLG) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ)
@inline Δyᶜᶠᵃ(i, j, k, grid::YRegularLLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ)
@inline Δyᶠᶜᵃ(i, j, k, grid::YRegularLLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ)
@inline Azᶠᶜᵃ(i, j, k, grid::XRegularLLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::XRegularLLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::XRegularLLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::XRegularLLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))

#####
##### Utilities to precompute metrics
#####

@inline metrics_precomputed(::LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, Nothing}) = false
@inline metrics_precomputed(::LatitudeLongitudeGrid) = true

#####
##### Kernels that precompute the z- and x-metric
#####

@inline metric_worksize(grid::LatitudeLongitudeGrid)  = (length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶠᵃ) - 2)
@inline metric_workgroup(grid::LatitudeLongitudeGrid) = (16, 16)

@inline metric_worksize(grid::XRegularLLG)  = length(grid.φᵃᶠᵃ) - 2
@inline metric_workgroup(grid::XRegularLLG) = 16

@kernel function compute_Δx_Az!(grid::LatitudeLongitudeGrid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)
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

@kernel function compute_Δx_Az!(grid::XRegularLLG, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)
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

@kernel function compute_Δy!(grid, Δyᶠᶜ, Δyᶜᶠ)
    j = @index(Global, Linear)

    # Manually offset y-index
    j′ = j + grid.Δφᵃᶜᵃ.offsets[1] + 1

    @inbounds begin
        Δyᶜᶠ[j′] = Δyᶜᶠᵃ(1, j′, 1, grid)
        Δyᶠᶜ[j′] = Δyᶠᶜᵃ(1, j′, 1, grid)
    end
end

#####
##### Metric memory allocation
#####

function allocate_metrics(grid::LatitudeLongitudeGrid)
    FT = eltype(grid)

    arch = grid.architecture

    if grid isa XRegularLLG
        offsets     = grid.φᵃᶜᵃ.offsets[1]
        metric_size = length(grid.φᵃᶜᵃ)
    else
        offsets     = (grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
        metric_size = (length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    end

    Δxᶠᶜ = OffsetArray(zeros(FT, arch, metric_size...), offsets...)
    Δxᶜᶠ = OffsetArray(zeros(FT, arch, metric_size...), offsets...)
    Δxᶠᶠ = OffsetArray(zeros(FT, arch, metric_size...), offsets...)
    Δxᶜᶜ = OffsetArray(zeros(FT, arch, metric_size...), offsets...)
    Azᶠᶜ = OffsetArray(zeros(FT, arch, metric_size...), offsets...)
    Azᶜᶠ = OffsetArray(zeros(FT, arch, metric_size...), offsets...)
    Azᶠᶠ = OffsetArray(zeros(FT, arch, metric_size...), offsets...)
    Azᶜᶜ = OffsetArray(zeros(FT, arch, metric_size...), offsets...)

    if grid isa YRegularLLG
        Δyᶠᶜ = Δyᶠᶜᵃ(1, 1, 1, grid)
        Δyᶜᶠ = Δyᶜᶠᵃ(1, 1, 1, grid)
    else
        parentC = zeros(FT, length(grid.Δφᵃᶜᵃ))
        parentF = zeros(FT, length(grid.Δφᵃᶜᵃ))
        Δyᶠᶜ    = OffsetArray(on_architecture(arch, parentC), grid.Δφᵃᶜᵃ.offsets[1])
        Δyᶜᶠ    = OffsetArray(on_architecture(arch, parentF), grid.Δφᵃᶜᵃ.offsets[1])
    end

    return Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Δyᶠᶜ, Δyᶜᶠ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ
end

#####
##### Utilities
#####

coordinates(::LatitudeLongitudeGrid) = (:λᶠᵃᵃ, :λᶜᵃᵃ, :φᵃᶠᵃ, :φᵃᶜᵃ, :zᵃᵃᶠ, :zᵃᵃᶜ)

#####
##### Grid nodes
#####

ξname(::LLG) = :λ
ηname(::LLG) = :φ
rname(::LLG) = :z

@inline λnode(i, grid::LLG, ::Center) = getnode(grid.λᶜᵃᵃ, i)
@inline λnode(i, grid::LLG, ::Face)   = getnode(grid.λᶠᵃᵃ, i)
@inline φnode(j, grid::LLG, ::Center) = getnode(grid.φᵃᶜᵃ, j)
@inline φnode(j, grid::LLG, ::Face)   = getnode(grid.φᵃᶠᵃ, j)
@inline znode(k, grid::LLG, ::Center) = getnode(grid.zᵃᵃᶜ, k)
@inline znode(k, grid::LLG, ::Face)   = getnode(grid.zᵃᵃᶠ, k)

# Definitions for node
@inline ξnode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = λnode(i, grid, ℓx)
@inline ηnode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = φnode(j, grid, ℓy)
@inline rnode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = znode(k, grid, ℓz)

@inline xnode(i, j, grid::LLG, ℓx, ℓy) = grid.radius * deg2rad(λnode(i, grid, ℓx)) * hack_cosd((φnode(j, grid, ℓy)))
@inline ynode(j, grid::LLG, ℓy)        = grid.radius * deg2rad(φnode(j, grid, ℓy))

# Convenience definitions
@inline λnode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = λnode(i, grid, ℓx)
@inline φnode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = φnode(j, grid, ℓy)
@inline xnode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = xnode(i, j, grid, ℓx, ℓy)
@inline ynode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = ynode(j, grid, ℓy)
@inline znode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = znode(k, grid, ℓz)

function nodes(grid::LLG, ℓx, ℓy, ℓz; reshape=false, with_halos=false)
    λ = λnodes(grid, ℓx, ℓy, ℓz; with_halos)
    φ = φnodes(grid, ℓx, ℓy, ℓz; with_halos)
    z = znodes(grid, ℓx, ℓy, ℓz; with_halos)

    if reshape
        # Here we have to deal with the fact that Flat directions may have
        # `nothing` nodes.
        #
        # A better solution (and more consistent with the rest of the API?)
        # might be to omit the `nothing` nodes in the `reshape`. In other words,
        # if `TX === Flat`, then we should return `(x, z)`. This is for future
        # consideration...
        #
        # See also `nodes` for `RectilinearGrid`.

        Nλ = isnothing(λ) ? 1 : length(λ)
        Nφ = isnothing(φ) ? 1 : length(φ)
        Nz = isnothing(z) ? 1 : length(z)

        λ = isnothing(λ) ? zeros(1, 1, 1) : Base.reshape(λ, Nλ, 1, 1)
        φ = isnothing(φ) ? zeros(1, 1, 1) : Base.reshape(φ, 1, Nφ, 1)
        z = isnothing(z) ? zeros(1, 1, 1) : Base.reshape(z, 1, 1, Nz)
    end

    return (λ, φ, z)
end

const F = Face
const C = Center

@inline function xnodes(grid::LLG, ℓx, ℓy; with_halos=false)
    λ = λnodes(grid, ℓx; with_halos=with_halos)'
    φ = φnodes(grid, ℓy; with_halos=with_halos)
    R = grid.radius
    return @. R * deg2rad(λ) * hack_cosd(φ)
end

@inline function ynodes(grid::LLG, ℓy; with_halos=false)
    φ = φnodes(grid, ℓy; with_halos=with_halos)
    R = grid.radius
    return @. R * deg2rad(φ)
end

@inline znodes(grid::LLG, ℓz::F; with_halos=false) = _property(grid.zᵃᵃᶠ, ℓz, topology(grid, 3), size(grid, 3), with_halos)
@inline znodes(grid::LLG, ℓz::C; with_halos=false) = _property(grid.zᵃᵃᶜ, ℓz, topology(grid, 3), size(grid, 3), with_halos)

# Convenience
@inline λnodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = λnodes(grid, ℓx; with_halos)
@inline φnodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = φnodes(grid, ℓy; with_halos)
@inline znodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = znodes(grid, ℓz; with_halos)
@inline xnodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = xnodes(grid, ℓx, ℓy; with_halos)
@inline ynodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = ynodes(grid, ℓy; with_halos)

# Generalized coordinates
@inline ξnodes(grid::LLG, ℓx; kwargs...) = λnodes(grid, ℓx; kwargs...)
@inline ηnodes(grid::LLG, ℓy; kwargs...) = φnodes(grid, ℓy; kwargs...)
@inline rnodes(grid::LLG, ℓz; kwargs...) = znodes(grid, ℓz; kwargs...)

@inline ξnodes(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = λnodes(grid, ℓx; kwargs...)
@inline ηnodes(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = φnodes(grid, ℓy; kwargs...)
@inline rnodes(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = znodes(grid, ℓz; kwargs...)

#####
##### Grid spacings in x, y, z (in meters)
#####

@inline xspacings(grid::LLG, ℓx::C, ℓy::C; with_halos=false) = _property(grid.Δxᶜᶜᵃ, ℓx, ℓy,
                                                                         topology(grid, 1), topology(grid, 2),
                                                                         size(grid, 1), size(grid, 2), with_halos)

@inline xspacings(grid::LLG, ℓx::C, ℓy::F; with_halos=false) = _property(grid.Δxᶜᶠᵃ, ℓx, ℓy,
                                                                         topology(grid, 1), topology(grid, 2),
                                                                         size(grid, 1), size(grid, 2), with_halos)

@inline xspacings(grid::LLG, ℓx::F, ℓy::C; with_halos=false) = _property(grid.Δxᶠᶜᵃ, ℓx, ℓy,
                                                                         topology(grid, 1), topology(grid, 2),
                                                                         size(grid, 1), size(grid, 2), with_halos)

@inline xspacings(grid::LLG, ℓx::F, ℓy::F; with_halos=false) = _property(grid.Δxᶠᶠᵃ, ℓx, ℓy,
                                                                         topology(grid, 1), topology(grid, 2),
                                                                         size(grid, 1), size(grid, 2), with_halos)

@inline xspacings(grid::HRegularLLG, ℓx::C, ℓy::C; with_halos=false) = _property(grid.Δxᶜᶜᵃ, ℓy, topology(grid, 2),
                                                                                 size(grid, 2), with_halos)

@inline xspacings(grid::HRegularLLG, ℓx::C, ℓy::F; with_halos=false) = _property(grid.Δxᶜᶠᵃ, ℓy, topology(grid, 2),
                                                                                 size(grid, 2), with_halos)

@inline xspacings(grid::HRegularLLG, ℓx::F, ℓy::C; with_halos=false) = _property(grid.Δxᶠᶜᵃ, ℓy, topology(grid, 2),
                                                                                 size(grid, 2), with_halos)

@inline xspacings(grid::HRegularLLG, ℓx::F, ℓy::F; with_halos=false) = _property(grid.Δxᶠᶠᵃ, ℓy, topology(grid, 2),
                                                                                 size(grid, 2), with_halos)


@inline yspacings(grid::YNonRegularLLG, ℓx::C, ℓy::F; with_halos=false) = _property(grid.Δyᶜᶠᵃ, ℓy, topoloy(grid, 2),
                                                                                    size(grid, 2), with_halos)

@inline yspacings(grid::YNonRegularLLG, ℓx::F, ℓy::C; with_halos=false) = _property(grid.Δyᶠᶜᵃ, ℓy, topoloy(grid, 2),
                                                                                    size(grid, 2), with_halos)

@inline yspacings(grid::YRegularLLG, ℓx, ℓy; with_halos=false) = yspacings(grid, ℓy; with_halos)
@inline yspacings(grid, ℓy::C; kwargs...) = grid.Δyᶠᶜᵃ
@inline yspacings(grid, ℓy::F; kwargs...) = grid.Δyᶜᶠᵃ

@inline xspacings(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = xspacings(grid, ℓx, ℓy; kwargs...)
@inline yspacings(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = yspacings(grid, ℓx, ℓy; kwargs...)
@inline zspacings(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = zspacings(grid, ℓz; kwargs...)

#####
##### Grid spacings in λ, φ (in degrees)
#####

@inline λnodes(grid::LLG, ℓx::F; with_halos=false) = _property(grid.λᶠᵃᵃ, ℓx, topology(grid, 1), size(grid, 1), with_halos)
@inline λnodes(grid::LLG, ℓx::C; with_halos=false) = _property(grid.λᶜᵃᵃ, ℓx, topology(grid, 1), size(grid, 1), with_halos)
@inline φnodes(grid::LLG, ℓy::F; with_halos=false) = _property(grid.φᵃᶠᵃ, ℓy, topology(grid, 2), size(grid, 2), with_halos)
@inline φnodes(grid::LLG, ℓy::C; with_halos=false) = _property(grid.φᵃᶜᵃ, ℓy, topology(grid, 2), size(grid, 2), with_halos)

#####
##### Grid spacings
#####

@inline λspacings(grid::LLG, ℓx::C; with_halos=false) = _property(grid.Δλᶜᵃᵃ, ℓx, topology(grid, 1), size(grid, 1), with_halos)
@inline λspacings(grid::LLG, ℓx::F; with_halos=false) = _property(grid.Δλᶠᵃᵃ, ℓx, topology(grid, 1), size(grid, 1), with_halos)
@inline φspacings(grid::LLG, ℓy::C; with_halos=false) = _property(grid.Δφᵃᶜᵃ, ℓy, topology(grid, 2), size(grid, 2), with_halos)
@inline φspacings(grid::LLG, ℓy::F; with_halos=false) = _property(grid.Δφᵃᶠᵃ, ℓy, topology(grid, 2), size(grid, 2), with_halos)
@inline zspacings(grid::LLG, ℓz::C; with_halos=false) = _property(grid.Δzᵃᵃᶜ, ℓz, topology(grid, 3), size(grid, 3), with_halos)
@inline zspacings(grid::LLG, ℓz::F; with_halos=false) = _property(grid.Δzᵃᵃᶠ, ℓz, topology(grid, 3), size(grid, 3), with_halos)

@inline λspacings(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = λspacings(grid, ℓx; kwargs...)
@inline φspacings(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = φspacings(grid, ℓy; kwargs...)

@inline λspacing(i, grid::LLG, ::C) = @inbounds grid.Δλᶜᵃᵃ[i]
@inline λspacing(i, grid::LLG, ::F) = @inbounds grid.Δλᶠᵃᵃ[i]
@inline λspacing(i, grid::XRegularLLG, ::C) = grid.Δλᶜᵃᵃ
@inline λspacing(i, grid::XRegularLLG, ::F) = grid.Δλᶠᵃᵃ

@inline φspacing(j, grid::LLG, ::C) = @inbounds grid.Δφᵃᶜᵃ[j]
@inline φspacing(j, grid::LLG, ::F) = @inbounds grid.Δφᵃᶠᵃ[j]
@inline φspacing(j, grid::YRegularLLG, ::C) = grid.Δφᵃᶜᵃ
@inline φspacing(j, grid::YRegularLLG, ::F) = grid.Δφᵃᶠᵃ

@inline λspacing(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = λspacing(i, grid, ℓx)
@inline φspacing(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = φspacing(j, grid, ℓy)
