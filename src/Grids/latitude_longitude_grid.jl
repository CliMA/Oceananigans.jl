using KernelAbstractions: @kernel, @index

""" 
    LatitudeLongitude

the mapping Type for a LatitudeLongitudeGrid. 
It holds the degree-spacings which inform the regularity of the grid.

If Δλᶠᵃᵃ is a `Number` the λ direction has a constant spacing, otherwise it is stretched.
If Δφᵃᶠᵃ is a `Number` the φ direction has a constant spacing, otherwise it is stretched.
"""
struct LatitudeLongitudeMapping{LF, PF, LC, PC} <: AbstractOrthogonalMapping
    Δλᶠᵃᵃ :: LF
    Δφᵃᶠᵃ :: PF
    Δλᶜᵃᵃ :: LC
    Δφᵃᶜᵃ :: PC
end

Adapt.adapt_structure(to, m::LatitudeLongitudeMapping) = 
    LatitudeLongitudeMapping(Adapt.adapt(to, m.Δλᶠᵃᵃ),
                             Adapt.adapt(to, m.Δφᵃᶠᵃ),
                             Adapt.adapt(to, m.Δλᶜᵃᵃ),
                             Adapt.adapt(to, m.Δφᵃᶜᵃ))

on_architecture(arch, m::LatitudeLongitudeMapping) =
    LatitudeLongitudeMapping(arch_array(arch, m.Δλᶠᵃᵃ), 
                             arch_array(arch, m.Δφᵃᶠᵃ), 
                             arch_array(arch, m.Δλᶜᵃᵃ), 
                             arch_array(arch, m.Δφᵃᶜᵃ))

const LatitudeLongitudeGrid{FT, TX, TY, TZ, FX, FY, FZ, Arch} = 
    OrthogonalSphericalShellGrid{FT, <:LatitudeLongitudeMapping, TX, TY, TZ, FX, FY, FZ, Arch} where {FT, TX, TY, TZ, FX, FY, FZ, Arch}
            
const LLG = LatitudeLongitudeGrid
#  LatitudeLongitudeGrid{FT, TX, TY, TZ, FX (stretching in x), FY (stretching in y), FZ (stretching in z)}
const XRegularLLG = OrthogonalSphericalShellGrid{<:Any, <:LatitudeLongitudeMapping{<:Number}}
const YRegularLLG = OrthogonalSphericalShellGrid{<:Any, <:LatitudeLongitudeMapping{<:Any, <:Number}}
const ZRegularLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const HRegularLLG = OrthogonalSphericalShellGrid{<:Any, <:LatitudeLongitudeMapping{<:Number, <:Number}}
const YNonRegularLLG = OrthogonalSphericalShellGrid{<:Any, <:LatitudeLongitudeMapping{<:Any, <:AbstractArray}}

const LLGNoMetric = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Nothing}
const XRegularLLGNoMetric = OrthogonalSphericalShellGrid{<:Any, <:LatitudeLongitudeMapping{<:Number}, <:Any, <:Any, <:Any, <:Nothing, <:Nothing}
const YRegularLLGNoMetric = OrthogonalSphericalShellGrid{<:Any, <:LatitudeLongitudeMapping{<:Any, <:Number}, <:Any, <:Any, <:Any, <:Nothing, <:Nothing}

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

    preliminary_grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture,
                                                                LatitudeLongitudeMapping(Δλᶠᵃᵃ, Δφᵃᶠᵃ, Δλᶜᵃᵃ, Δφᵃᶜᵃ),
                                                                Nλ, Nφ, Nz,
                                                                Hλ, Hφ, Hz,
                                                                Lλ, Lφ, Lz,
                                                                λᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, λᶠᵃᵃ, 
                                                                φᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ, φᵃᶠᵃ, 
                                                                zᵃᵃᶜ, zᵃᵃᶠ,
                                                                Δzᵃᵃᶜ, Δzᵃᵃᶠ, 
                                                                (nothing for i=1:12)..., FT(radius))

    return !precompute_metrics ? preliminary_grid : with_precomputed_metrics(preliminary_grid)
end

# architecture = CPU() default, assuming that a DataType positional arg
# is specifying the floating point type.
LatitudeLongitudeGrid(FT::DataType; kwargs...) = LatitudeLongitudeGrid(CPU(), FT; kwargs...)

""" Return a reproduction of `grid` with precomputed metric terms. """
function with_precomputed_metrics(grid)
    Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ = allocate_metrics(grid)

    precompute_curvilinear_metrics!(grid, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ)

    Δyᶠᶜᵃ, Δyᶜᶠᵃ = precompute_Δy_metrics(grid, Δyᶠᶜᵃ, Δyᶜᶠᵃ)

    Nλ, Nφ, Nz = size(grid)
    Hλ, Hφ, Hz = halo_size(grid)
    TX, TY, TZ = topology(grid)

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture(grid),
                                                    grid.mapping,
                                                    Nλ, Nφ, Nz,
                                                    Hλ, Hφ, Hz,
                                                    grid.Lx, grid.Ly, grid.Lz,
                                                    grid.λᶜᶜᵃ, grid.λᶠᶜᵃ, grid.λᶜᶠᵃ, grid.λᶠᶠᵃ,
                                                    grid.φᶜᶜᵃ, grid.φᶠᶜᵃ, grid.φᶜᶠᵃ, grid.φᶠᶠᵃ,
                                                    grid.zᵃᵃᶜ, grid.zᵃᵃᶠ,
                                                    grid.Δzᵃᵃᶜ, grid.Δzᵃᵃᶠ,
                                                    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,  
                                                    Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ,
                                                    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, grid.radius)
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

    halo = validate_halo(TX, TY, TZ, halo)
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

    λ₁, λ₂ = domain(TX(), size(grid, 1), grid.λᶠᶠᵃ)
    φ₁, φ₂ = domain(TY(), size(grid, 2), grid.φᶠᶠᵃ)
    z₁, z₂ = domain(TZ(), size(grid, 3), grid.zᵃᵃᶠ)

    x_summary = domain_summary(TX(), "λ", λ₁, λ₂)
    y_summary = domain_summary(TY(), "φ", φ₁, φ₂)
    z_summary = domain_summary(TZ(), "z", z₁, z₂)

    longest = max(length(x_summary), length(y_summary), length(z_summary))

    x_summary = "longitude: " * dimension_summary(TX(), "λ", λ₁, λ₂, grid.mapping.Δλᶜᵃᵃ, longest - length(x_summary))
    y_summary = "latitude:  " * dimension_summary(TY(), "φ", φ₁, φ₂, grid.mapping.Δφᵃᶜᵃ, longest - length(y_summary))
    z_summary = "z:         " * dimension_summary(TZ(), "z", z₁, z₂, grid.Δzᵃᵃᶜ,                longest - length(z_summary))

    if withsummary
        print(io, summary(grid), "\n")
    end

    return print(io, "├── ", x_summary, "\n",
                     "├── ", y_summary, "\n",
                     "└── ", z_summary)
end

@inline x_domain(grid::LLG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TX, grid.Nx, grid.λᶠᶠᵃ)
@inline y_domain(grid::LLG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TY, grid.Ny, grid.φᶠᶠᵃ)
@inline z_domain(grid::LLG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)

@inline cpu_face_constructor_x(grid::XRegularLLG) = x_domain(grid)
@inline cpu_face_constructor_y(grid::YRegularLLG) = y_domain(grid)
@inline cpu_face_constructor_z(grid::ZRegularLLG) = z_domain(grid)

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
                                     longitude = x, latitude = y, z = z, topology = topo,
                                     precompute_metrics = metrics_precomputed(old_grid),
                                     radius = old_grid.radius)

    return new_grid
end

#####
##### On-the-fly computation of LatitudeLongitudeGrid metrics
#####

@inline hack_cosd(φ) = cos(π * φ / 180)
@inline hack_sind(φ) = sin(π * φ / 180)

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGNoMetric) = @inbounds grid.radius * hack_cosd(grid.φᶜᶜᵃ[j]) * deg2rad(grid.mapping.Δλᶠᵃᵃ[i])
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGNoMetric) = @inbounds grid.radius * hack_cosd(grid.φᶠᶠᵃ[j]) * deg2rad(grid.mapping.Δλᶜᵃᵃ[i])
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGNoMetric) = @inbounds grid.radius * hack_cosd(grid.φᶠᶠᵃ[j]) * deg2rad(grid.mapping.Δλᶠᵃᵃ[i])
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGNoMetric) = @inbounds grid.radius * hack_cosd(grid.φᶜᶜᵃ[j]) * deg2rad(grid.mapping.Δλᶜᵃᵃ[i])
@inline Δyᶜᶠᵃ(i, j, k, grid::LLGNoMetric) = @inbounds grid.radius * deg2rad(grid.mapping.Δφᵃᶠᵃ[j])
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGNoMetric) = @inbounds grid.radius * deg2rad(grid.mapping.Δφᵃᶜᵃ[j])
@inline Azᶠᶜᵃ(i, j, k, grid::LLGNoMetric) = @inbounds grid.radius^2 * deg2rad(grid.mapping.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᶠᶠᵃ[j+1]) - hack_sind(grid.φᶠᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLGNoMetric) = @inbounds grid.radius^2 * deg2rad(grid.mapping.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᶜᶜᵃ[j])   - hack_sind(grid.φᶜᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGNoMetric) = @inbounds grid.radius^2 * deg2rad(grid.mapping.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᶜᶜᵃ[j])   - hack_sind(grid.φᶜᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGNoMetric) = @inbounds grid.radius^2 * deg2rad(grid.mapping.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᶠᶠᵃ[j+1]) - hack_sind(grid.φᶠᶠᵃ[j]))

@inline Δxᶠᶜᵃ(i, j, k, grid::XRegularLLGNoMetric) = @inbounds grid.radius * hack_cosd(grid.φᶜᶜᵃ[j]) * deg2rad(grid.mapping.Δλᶠᵃᵃ)
@inline Δxᶜᶠᵃ(i, j, k, grid::XRegularLLGNoMetric) = @inbounds grid.radius * hack_cosd(grid.φᶠᶠᵃ[j]) * deg2rad(grid.mapping.Δλᶜᵃᵃ)
@inline Δxᶠᶠᵃ(i, j, k, grid::XRegularLLGNoMetric) = @inbounds grid.radius * hack_cosd(grid.φᶠᶠᵃ[j]) * deg2rad(grid.mapping.Δλᶠᵃᵃ)
@inline Δxᶜᶜᵃ(i, j, k, grid::XRegularLLGNoMetric) = @inbounds grid.radius * hack_cosd(grid.φᶜᶜᵃ[j]) * deg2rad(grid.mapping.Δλᶜᵃᵃ)
@inline Δyᶜᶠᵃ(i, j, k, grid::YRegularLLGNoMetric) = @inbounds grid.radius * deg2rad(grid.mapping.Δφᵃᶠᵃ)
@inline Δyᶠᶜᵃ(i, j, k, grid::YRegularLLGNoMetric) = @inbounds grid.radius * deg2rad(grid.mapping.Δφᵃᶜᵃ)
@inline Azᶠᶜᵃ(i, j, k, grid::XRegularLLGNoMetric) = @inbounds grid.radius^2 * deg2rad(grid.mapping.Δλᶠᵃᵃ) * (hack_sind(grid.φᶠᶠᵃ[j+1]) - hack_sind(grid.φᶠᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::XRegularLLGNoMetric) = @inbounds grid.radius^2 * deg2rad(grid.mapping.Δλᶜᵃᵃ) * (hack_sind(grid.φᶜᶜᵃ[j])   - hack_sind(grid.φᶜᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::XRegularLLGNoMetric) = @inbounds grid.radius^2 * deg2rad(grid.mapping.Δλᶠᵃᵃ) * (hack_sind(grid.φᶜᶜᵃ[j])   - hack_sind(grid.φᶜᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::XRegularLLGNoMetric) = @inbounds grid.radius^2 * deg2rad(grid.mapping.Δλᶜᵃᵃ) * (hack_sind(grid.φᶠᶠᵃ[j+1]) - hack_sind(grid.φᶠᶠᵃ[j]))

#####
##### Utilities to precompute metrics 
#####

@inline metrics_precomputed(::LLGNoMetric)     = false 
@inline metrics_precomputed(::LatitudeLongitudeGrid) = true

#####
##### Kernels that precompute the z- and x-metric
#####

@inline metric_worksize(grid::LatitudeLongitudeGrid)  = (length(grid.mapping.Δλᶜᵃᵃ), length(grid.φᶜᶜᵃ) - 1) 
@inline metric_workgroup(grid::LatitudeLongitudeGrid) = (16, 16) 

@inline metric_worksize(grid::XRegularLLG)  = length(grid.φᶜᶜᵃ) - 1 
@inline metric_workgroup(grid::XRegularLLG) = 16

function precompute_curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    
    arch = grid.architecture

    workgroup, worksize  = metric_workgroup(grid), metric_worksize(grid)
    curvilinear_metrics! = precompute_metrics_kernel!(Architectures.device(arch), workgroup, worksize)

    curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)

    return nothing
end

@kernel function precompute_metrics_kernel!(grid::LatitudeLongitudeGrid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    i, j = @index(Global, NTuple)

    # Manually offset x- and y-index
    i′ = i + grid.mapping.Δλᶜᵃᵃ.offsets[1]
    j′ = j + grid.φᶜᶜᵃ.offsets[1] + 1

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

@kernel function precompute_metrics_kernel!(grid::XRegularLLG, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    j = @index(Global, Linear)

    # Manually offset y-index
    j′ = j + grid.φᶜᶜᵃ.offsets[1] + 1

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
    precompute_Δy! = precompute_Δy_kernel!(Architectures.device(arch), 16, length(grid.mapping.Δφᵃᶜᵃ) - 1)
    precompute_Δy!(grid, Δyᶠᶜ, Δyᶜᶠ)
    
    return Δyᶠᶜ, Δyᶜᶠ
end

function  precompute_Δy_metrics(grid::YRegularLLG, Δyᶠᶜ, Δyᶜᶠ)
    Δyᶜᶠ =  Δyᶜᶠᵃ(1, 1, 1, grid)
    Δyᶠᶜ =  Δyᶠᶜᵃ(1, 1, 1, grid)
    return Δyᶠᶜ, Δyᶜᶠ
end

@kernel function precompute_Δy_kernel!(grid, Δyᶠᶜ, Δyᶜᶠ)
    j = @index(Global, Linear)

    # Manually offset y-index
    j′ = j + grid.mapping.Δφᵃᶜᵃ.offsets[1] + 1

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
    
    # preallocate quantities to ensure correct type and size
    grid_metrics = (:Δxᶠᶜ,
                    :Δxᶜᶠ,
                    :Δxᶠᶠ,
                    :Δxᶜᶜ,
                    :Azᶠᶜ,
                    :Azᶜᶠ,
                    :Azᶠᶠ,
                    :Azᶜᶜ)

    arch = architecture(grid)
    
    if grid isa XRegularLLG
        offsets     = grid.φᶜᶜᵃ.offsets[1]
        metric_size = length(grid.φᶜᶜᵃ)
    else
        offsets     = (grid.mapping.Δλᶜᵃᵃ.offsets[1], grid.φᶜᶜᵃ.offsets[1])
        metric_size = (length(grid.mapping.Δλᶜᵃᵃ)   , length(grid.φᶜᶜᵃ))
    end

    for metric in grid_metrics
        parentM        = Symbol(metric, :_parent)
        @eval $parentM = zeros($FT, $arch, $metric_size...)
        @eval $metric  = OffsetArray($parentM, $offsets...)
    end

    if grid isa YRegularLLG
        Δyᶠᶜ = FT(0.0)
        Δyᶜᶠ = FT(0.0)
    else
        parentC = zeros(FT, length(grid.mapping.Δφᵃᶜᵃ))
        parentF = zeros(FT, length(grid.mapping.Δφᵃᶜᵃ))
        Δyᶠᶜ    = OffsetArray(arch_array(arch, parentC), grid.mapping.Δφᵃᶜᵃ.offsets[1])
        Δyᶜᶠ    = OffsetArray(arch_array(arch, parentF), grid.mapping.Δφᵃᶜᵃ.offsets[1])
    end
    
    return Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Δyᶠᶜ, Δyᶜᶠ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ
end

#####
##### Utilities
#####

coordinates(::LatitudeLongitudeGrid) = (:λᶠᶠᵃ, :λᶜᶜᵃ, :φᶠᶠᵃ, :φᶜᶜᵃ, :zᵃᵃᶠ, :zᵃᵃᶜ)

#####
##### Grid nodes
#####

ξname(::LLG) = :λ
ηname(::LLG) = :φ
rname(::LLG) = :z

@inline λnode(i, grid::LLG, ::Center) = @inbounds grid.λᶜᶜᵃ[i]
@inline λnode(i, grid::LLG, ::Face)   = @inbounds grid.λᶠᶠᵃ[i]
@inline φnode(j, grid::LLG, ::Center) = @inbounds grid.φᶜᶜᵃ[j]
@inline φnode(j, grid::LLG, ::Face)   = @inbounds grid.φᶠᶠᵃ[j]
@inline znode(k, grid::LLG, ::Center) = @inbounds grid.zᵃᵃᶜ[k]
@inline znode(k, grid::LLG, ::Face)   = @inbounds grid.zᵃᵃᶠ[k]

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
        N = (length(λ), length(φ), length(z))
        λ = Base.reshape(λ, N[1], 1, 1)
        φ = Base.reshape(φ, 1, N[2], 1)
        z = Base.reshape(z, 1, 1, N[3])
    end

    return (λ, φ, z)
end

const F = Face
const C = Center

@inline λnodes(grid::LLG, ℓx::F; with_halos=false) = with_halos ? grid.λᶠᶠᵃ :
    view(grid.λᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), size(grid, 1)))
@inline λnodes(grid::LLG, ℓx::C; with_halos=false) = with_halos ? grid.λᶜᶜᵃ :
    view(grid.λᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), size(grid, 1)))

@inline φnodes(grid::LLG, ℓy::F; with_halos=false) = with_halos ? grid.φᶠᶠᵃ :
    view(grid.φᶠᶠᵃ, interior_indices(ℓy, topology(grid, 2)(), size(grid, 2)))
@inline φnodes(grid::LLG, ℓy::C; with_halos=false) = with_halos ? grid.φᶜᶜᵃ :
    view(grid.φᶜᶜᵃ, interior_indices(ℓy, topology(grid, 2)(), size(grid, 2)))

@inline xnodes(grid::LLG, ℓx, ℓy; with_halos=false) =
    grid.radius * deg2rad.(λnodes(grid, ℓx; with_halos=with_halos))' .* hack_cosd.(φnodes(grid, ℓy; with_halos=with_halos))
@inline ynodes(grid::LLG, ℓy; with_halos=false)     =
    grid.radius * deg2rad.(φnodes(grid, ℓy; with_halos=with_halos))

@inline znodes(grid::LLG, ℓz::F; with_halos=false) = with_halos ? grid.zᵃᵃᶠ :
    view(grid.zᵃᵃᶠ, interior_indices(ℓz, topology(grid, 3)(), size(grid, 3)))
@inline znodes(grid::LLG, ℓz::C; with_halos=false) = with_halos ? grid.zᵃᵃᶜ :
    view(grid.zᵃᵃᶜ, interior_indices(ℓz, topology(grid, 3)(), size(grid, 3)))

# Convenience
@inline λnodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = λnodes(grid, ℓx; with_halos)
@inline φnodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = φnodes(grid, ℓy; with_halos)
@inline znodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = znodes(grid, ℓz; with_halos)
@inline xnodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = xnodes(grid, ℓx, ℓy; with_halos)
@inline ynodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = ynodes(grid, ℓy; with_halos)

#####
##### Grid spacings in x, y, z (in meters)
#####

@inline xspacings(grid::LLG, ℓx::C, ℓy::C; with_halos=false) = with_halos ? grid.Δxᶜᶜᵃ :
    view(grid.Δxᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), size(grid, 2)))
@inline xspacings(grid::LLG, ℓx::C, ℓy::F;   with_halos=false) = with_halos ? grid.Δxᶜᶠᵃ :
    view(grid.Δxᶜᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), size(grid, 2)))
@inline xspacings(grid::LLG, ℓx::F, ℓy::C;   with_halos=false) = with_halos ? grid.Δxᶠᶜᵃ :
    view(grid.Δxᶠᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), size(grid, 2)))
@inline xspacings(grid::LLG, ℓx::F, ℓy::F;     with_halos=false) = with_halos ? grid.Δxᶠᶠᵃ :
    view(grid.Δxᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), size(grid, 2)))

@inline xspacings(grid::HRegularLLG, ℓx::C, ℓy::C; with_halos=false) = with_halos ? grid.Δxᶜᶜᵃ :
    view(grid.Δxᶜᶜᵃ, interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline xspacings(grid::HRegularLLG, ℓx::C, ℓy::F;   with_halos=false) = with_halos ? grid.Δxᶜᶠᵃ :
    view(grid.Δxᶜᶠᵃ, interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline xspacings(grid::HRegularLLG, ℓx::F, ℓy::C;   with_halos=false) = with_halos ? grid.Δxᶠᶜᵃ :
    view(grid.Δxᶠᶜᵃ, interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline xspacings(grid::HRegularLLG, ℓx::F, ℓy::F;     with_halos=false) = with_halos ? grid.Δxᶠᶠᵃ :
    view(grid.Δxᶠᶠᵃ, interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline yspacings(grid::LLG, ℓx::C, ℓy::F;   with_halos=false) = with_halos ? grid.Δyᶜᶠᵃ :
    view(grid.Δyᶜᶠᵃ, interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline yspacings(grid::LLG, ℓx::F,   ℓy::C; with_halos=false) = with_halos ? grid.Δyᶠᶜᵃ :
    view(grid.Δyᶠᶜᵃ, interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline yspacings(grid::YRegularLLG, ℓx::C, ℓy::F; with_halos=false) = yspacings(grid, ℓy; with_halos)
@inline yspacings(grid::YRegularLLG, ℓx::F, ℓy::C; with_halos=false) = yspacings(grid, ℓy; with_halos)
@inline yspacings(grid, ℓy::C; kwargs...) = grid.Δyᶠᶜᵃ
@inline yspacings(grid, ℓy::F; kwargs...) = grid.Δyᶜᶠᵃ

@inline zspacings(grid::LLG, ℓz::C; with_halos=false) = with_halos ? grid.Δzᵃᵃᶜ : view(grid.Δzᵃᵃᶜ, interior_indices(ℓz, topology(grid, 3)(), size(grid, 3)))
@inline zspacings(grid::LLG, ℓz::F; with_halos=false) = with_halos ? grid.Δzᵃᵃᶠ : view(grid.Δzᵃᵃᶠ, interior_indices(ℓz, topology(grid, 3)(), size(grid, 3)))
@inline zspacings(grid::ZRegularLLG, ℓz::C; with_halos=false) = grid.Δzᵃᵃᶜ
@inline zspacings(grid::ZRegularLLG, ℓz::F; with_halos=false) = grid.Δzᵃᵃᶠ

@inline xspacings(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = xspacings(grid, ℓx, ℓy; kwargs...)
@inline yspacings(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = yspacings(grid, ℓx, ℓy; kwargs...)
@inline zspacings(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = zspacings(grid, ℓz; kwargs...)

#####
##### Grid spacings in λ, φ (in degrees)
#####

@inline λspacings(grid::LLG, ℓx::C; with_halos=false) = with_halos ? grid.mapping.Δλᶜᵃᵃ : view(grid.mapping.Δλᶜᵃᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx))
@inline λspacings(grid::LLG, ℓx::F; with_halos=false) = with_halos ? grid.mapping.Δλᶠᵃᵃ : view(grid.mapping.Δλᶠᵃᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx))
@inline λspacings(grid::XRegularLLG, ℓx::C; with_halos=false) = grid.λᶠᶠᵃ[2] - grid.λᶠᶠᵃ[1]
@inline λspacings(grid::XRegularLLG, ℓx::F; with_halos=false) = grid.λᶜᶜᵃ[2] - grid.λᶜᶜᵃ[1]

@inline φspacings(grid::LLG, ℓy::C; with_halos=false) = with_halos ? grid.mapping.Δφᵃᶜᵃ : view(grid.mapping.Δφᵃᶜᵃ, interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φspacings(grid::LLG, ℓy::F; with_halos=false) = with_halos ? grid.mapping.Δφᵃᶠᵃ : view(grid.mapping.Δφᵃᶠᵃ, interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φspacings(grid::YRegularLLG, ℓy::C; with_halos=false) = grid.φᶠᶠᵃ[2] - grid.φᶠᶠᵃ[1]
@inline φspacings(grid::YRegularLLG, ℓy::F; with_halos=false) = grid.φᶜᶜᵃ[2] - grid.φᶜᶜᵃ[1]

@inline λspacings(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = λspacings(grid, ℓx; with_halos)
@inline φspacings(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = φspacings(grid, ℓy; with_halos)

@inline λspacing(i, grid::LLG, ::C) = @inbounds grid.λᶠᶠᵃ[i+1] - grid.λᶠᶠᵃ[i]
@inline λspacing(i, grid::LLG, ::F) = @inbounds grid.λᶜᶜᵃ[i]   - grid.λᶜᶜᵃ[i-1]
@inline φspacing(j, grid::LLG, ::C) = @inbounds grid.φᶠᶠᵃ[j+1] - grid.φᶠᶠᵃ[j]
@inline φspacing(j, grid::LLG, ::F) = @inbounds grid.φᶜᶜᵃ[j]   - grid.φᶜᶜᵃ[j-1]

@inline λspacing(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = λspacing(i, grid, ℓx)
@inline φspacing(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = φspacing(j, grid, ℓy)
