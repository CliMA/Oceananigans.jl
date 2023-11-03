using KernelAbstractions: @kernel, @index

""" 
    LatitudeLongitude

the classification Type for a LatitudeLongitudeGrid. 
It holds the degree-spacings which inform the regularity of the grid.

If Δλᶠᵃᵃ is a `Number` the λ direction has a constant spacing, otherwise the λ direction is stretched.
If Δφᵃᶠᵃ is a `Number` the λ direction has a constant spacing, otherwise the λ direction is stretched.
"""
struct LatitudeLongitude{LF, PF, LC, PC}
    Δλᶠᵃᵃ :: LF
    Δφᵃᶠᵃ :: PF
    Δλᶜᵃᵃ :: LC
    Δφᵃᶜᵃ :: PC
end

const LatitudeLongitudeGrid{FT, TX, TY, TZ, FX, FY, FZ, Arch} = 
    OrthogonalSphericalShellGrid{FT, <:LatitudeLongitude, TX, TY, TZ, FX, FY, FZ, Arch} where {FT, TX, TY, TZ, FX, FY, FZ, Arch}
            
const LLG = LatitudeLongitudeGrid
#                   LatitudeLongitudeGrid{FT, TX, TY, TZ, FX (stretching in x), FY (stretching in y), FZ (stretching in z)}
const XRegularLLG = OrthogonalSphericalShellGrid{<:Any, <:LatitudeLongitude{<:Number}}
const YRegularLLG = OrthogonalSphericalShellGrid{<:Any, <:LatitudeLongitude{<:Any, <:Number}}
const ZRegularLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const HRegularLLG = OrthogonalSphericalShellGrid{<:Any, <:LatitudeLongitude{<:Number, <:Number}}
const YNonRegularLLG = OrthogonalSphericalShellGrid{<:Any, <:LatitudeLongitude{<:Any, <:AbstractArray}}

const LLGWithoutMetrics = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Nothing}

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

    Nλ, Nφ, Nz, Hλ, Hφ, Hz, latitude, longitude, z, topology, precompute_metrics =
        validate_lat_lon_grid_args(FT, latitude, longitude, z, size, halo, topology, precompute_metrics)

    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real}, 
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    TX, TY, TZ = topology

    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, TX(), Nλ, Hλ, longitude, :longitude, architecture)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, TY(), Nφ, Hφ, latitude,  :latitude,  architecture)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, TZ(), Nz, Hz, z,         :z,         architecture)

    preliminary_grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture,
                                                                LatitudeLongitude(Δλᶠᵃᵃ, Δφᵃᶠᵃ, Δλᶜᵃᵃ, Δφᵃᶜᵃ),
                                                                Nλ, Nφ, Nz,
                                                                Hλ, Hφ, Hz,
                                                                Lλ, Lφ, Lz,
                                                                λᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, λᶠᵃᵃ, 
                                                                φᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ, φᵃᶠᵃ, 
                                                                zᵃᵃᶜ, zᵃᵃᶠ,
                                                                Δzᵃᵃᶠ, Δzᵃᵃᶜ,
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
                                                    grid.classification,
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

function validate_lat_lon_grid_args(FT, latitude, longitude, z, size, halo, topology, precompute_metrics)
    Nλ, Nφ, Nz = N = size
    
    λ₁, λ₂ = get_domain_extent(longitude, Nλ)
    @assert λ₁ <= λ₂ && λ₂ - λ₁ ≤ 360

    φ₁, φ₂ = get_domain_extent(latitude, Nφ)
    @assert -90 <= φ₁ <= φ₂ <= 90

    (φ₁ == -90 || φ₂ == 90) &&
        @warn "Are you sure you want to use a latitude-longitude grid with a grid point at the pole?"
    
    if !isnothing(topology)
        TX, TY, TZ = topology
        Nλ, Nφ, Nz = N = validate_size(TX, TY, TZ, size)
        Hλ, Hφ, Hz = H = validate_halo(TX, TY, TZ, halo)
    else
        Lλ = λ₂ - λ₁

        TX = Lλ == 360 ? Periodic : Bounded
        TY = Bounded
        TZ = Bounded
    end

    if TX == Flat || TY == Flat 
        precompute_metrics = false
    end

    Hλ, Hφ, Hz = H = validate_halo(TX, TY, TZ, halo)

    longitude = validate_dimension_specification(TX, longitude, :x, Nλ, FT)
    latitude  = validate_dimension_specification(TY, latitude,  :y, Nφ, FT)
    z         = validate_dimension_specification(TZ, z,         :z, Nz, FT)

    return Nλ, Nφ, Nz, Hλ, Hφ, Hz, latitude, longitude, z, (TX, TY, TZ), precompute_metrics
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

    x_summary = "longitude: " * dimension_summary(TX(), "λ", λ₁, λ₂, grid.classification.Δλᶜᵃᵃ, longest - length(x_summary))
    y_summary = "latitude:  " * dimension_summary(TY(), "φ", φ₁, φ₂, grid.classification.Δφᵃᶜᵃ, longest - length(y_summary))
    z_summary = "z:         " * dimension_summary(TZ(), "z", z₁, z₂, grid.Δzᵃᵃᶜ,                longest - length(z_summary))

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

function on_architecture(new_arch::AbstractArchitecture, old_grid::LatitudeLongitudeGrid)
    old_properties = (old_grid.λᶜᶜᵃ,  old_grid.λᶠᶜᵃ,  old_grid.λᶜᶠᵃ,  old_grid.λᶠᶠᵃ, 
                      old_grid.φᶜᶜᵃ,  old_grid.φᶠᶜᵃ,  old_grid.φᶜᶠᵃ,  old_grid.φᶠᶠᵃ, 
                      old_grid.zᵃᵃᶜ,  old_grid.zᵃᵃᶠ,  old_grid.Δzᵃᵃᶜ, old_grid.Δzᵃᵃᶠ, 
                      old_grid.Δxᶜᶜᵃ, old_grid.Δxᶠᶜᵃ, old_grid.Δxᶜᶠᵃ, old_grid.Δxᶠᶠᵃ, 
                      old_grid.Δyᶜᶜᵃ, old_grid.Δyᶠᶜᵃ, old_grid.Δyᶜᶠᵃ, old_grid.Δyᶠᶠᵃ,                      
                      old_grid.Azᶜᶜᵃ, old_grid.Azᶠᶜᵃ, old_grid.Azᶜᶠᵃ, old_grid.Azᶠᶠᵃ)

    new_properties = Tuple(arch_array(new_arch, p) for p in old_properties)

    TX, TY, TZ = topology(old_grid)

    return LatitudeLongitudeGrid{TX, TY, TZ}(new_arch,
                                             old_grid.classification,
                                             old_grid.Nx, old_grid.Ny, old_grid.Nz,
                                             old_grid.Hx, old_grid.Hy, old_grid.Hz,
                                             old_grid.Lx, old_grid.Ly, old_grid.Lz,
                                             new_properties...,
                                             old_grid.radius)
end

#####
##### On-the-fly computation of LatitudeLongitudeGrid metrics
#####

@inline hack_cosd(φ) = cos(π * φ / 180)
@inline hack_sind(φ) = sin(π * φ / 180)

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGWithoutMetrics) = @inbounds grid.radius * hack_cosd(grid.φᶜᶜᵃ[j]) * deg2rad(grid.λᶜᶜᵃ[i] - grid.λᶜᶜᵃ[i-1])
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGWithoutMetrics) = @inbounds grid.radius * hack_cosd(grid.φᶠᶠᵃ[j]) * deg2rad(grid.λᶠᶠᵃ[i+1] - grid.λᶠᶠᵃ[i])
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGWithoutMetrics) = @inbounds grid.radius * hack_cosd(grid.φᶠᶠᵃ[j]) * deg2rad(grid.λᶜᶜᵃ[i] - grid.λᶜᶜᵃ[i-1])
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGWithoutMetrics) = @inbounds grid.radius * hack_cosd(grid.φᶜᶜᵃ[j]) * deg2rad(grid.λᶠᶠᵃ[i+1] - grid.λᶠᶠᵃ[i])
@inline Δyᶜᶠᵃ(i, j, k, grid::LLGWithoutMetrics) = @inbounds grid.radius * deg2rad(grid.φᶜᶜᵃ[j] - grid.φᶜᶜᵃ[j-1])
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGWithoutMetrics) = @inbounds grid.radius * deg2rad(grid.φᶠᶠᵃ[i+1] - grid.φᶠᶠᵃ[i])
@inline Azᶠᶜᵃ(i, j, k, grid::LLGWithoutMetrics) = @inbounds grid.radius^2 * deg2rad(grid.λᶜᶜᵃ[i] - grid.λᶜᶜᵃ[i-1]) * (hack_sind(grid.φᶠᶠᵃ[j+1]) - hack_sind(grid.φᶠᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLGWithoutMetrics) = @inbounds grid.radius^2 * deg2rad(grid.λᶠᶠᵃ[i+1] - grid.λᶠᶠᵃ[i]) * (hack_sind(grid.φᶜᶜᵃ[j])   - hack_sind(grid.φᶜᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGWithoutMetrics) = @inbounds grid.radius^2 * deg2rad(grid.λᶜᶜᵃ[i] - grid.λᶜᶜᵃ[i-1]) * (hack_sind(grid.φᶜᶜᵃ[j])   - hack_sind(grid.φᶜᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGWithoutMetrics) = @inbounds grid.radius^2 * deg2rad(grid.λᶠᶠᵃ[i+1] - grid.λᶠᶠᵃ[i]) * (hack_sind(grid.φᶠᶠᵃ[j+1]) - hack_sind(grid.φᶠᶠᵃ[j]))

#####
##### Utilities to precompute metrics 
#####

@inline metrics_precomputed(::LLGWithoutMetrics)     = false 
@inline metrics_precomputed(::LatitudeLongitudeGrid) = true

#####
##### Kernels that precompute the z- and x-metric
#####

@inline metric_worksize(grid::LatitudeLongitudeGrid)  = (length(grid.λᶜᶜᵃ) - 1, length(grid.φᶜᶜᵃ) - 1) 
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
    i′ = i + grid.λᶜᶜᵃ.offsets[1] + 1
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
    precompute_Δy! = precompute_Δy_kernel!(Architectures.device(arch), 16, length(grid.classification.Δφᵃᶜᵃ) - 1)
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
    j′ = j + grid.classification.Δφᵃᶜᵃ.offsets[1] + 1

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

    arch = grid.architecture
    
    if grid isa XRegularLLG
        offsets     = grid.φᶜᶜᵃ.offsets[1]
        metric_size = length(grid.φᶜᶜᵃ)
    else
        offsets     = (grid.λᶜᶜᵃ.offsets[1], grid.φᶜᶜᵃ.offsets[1])
        metric_size = (length(grid.λᶜᶜᵃ)   , length(grid.φᶜᶜᵃ))
    end

    for metric in grid_metrics
        parentM        = Symbol(metric, :_parent)
        @eval $parentM = zeros($FT, $metric_size...)
        @eval $metric  = OffsetArray(arch_array($arch, $parentM), $offsets...)
    end

    if grid isa YRegularLLG
        Δyᶠᶜ = FT(0.0)
        Δyᶜᶠ = FT(0.0)
    else
        parentC = zeros(FT, length(grid.classification.Δφᵃᶜᵃ))
        parentF = zeros(FT, length(grid.classification.Δφᵃᶜᵃ))
        Δyᶠᶜ    = OffsetArray(arch_array(arch, parentC), grid.classification.Δφᵃᶜᵃ.offsets[1])
        Δyᶜᶠ    = OffsetArray(arch_array(arch, parentF), grid.classification.Δφᵃᶜᵃ.offsets[1])
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

@inline λnode(i, grid::LLG, ::Center) = @inbounds grid.λᶜᵃᵃ[i]
@inline λnode(i, grid::LLG, ::Face)   = @inbounds grid.λᶠᵃᵃ[i]
@inline φnode(j, grid::LLG, ::Center) = @inbounds grid.φᵃᶜᵃ[j]
@inline φnode(j, grid::LLG, ::Face)   = @inbounds grid.φᵃᶠᵃ[j]
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

@inline λnodes(grid::LLG, ℓx::F; with_halos=false) = with_halos ? grid.λᶠᵃᵃ :
    view(grid.λᶠᵃᵃ, interior_indices(ℓx, topology(grid, 1)(), size(grid, 1)))
@inline λnodes(grid::LLG, ℓx::C; with_halos=false) = with_halos ? grid.λᶜᵃᵃ :
    view(grid.λᶜᵃᵃ, interior_indices(ℓx, topology(grid, 1)(), size(grid, 1)))

@inline φnodes(grid::LLG, ℓy::F; with_halos=false) = with_halos ? grid.φᵃᶠᵃ :
    view(grid.φᵃᶠᵃ, interior_indices(ℓy, topology(grid, 2)(), size(grid, 2)))
@inline φnodes(grid::LLG, ℓy::C; with_halos=false) = with_halos ? grid.φᵃᶜᵃ :
    view(grid.φᵃᶜᵃ, interior_indices(ℓy, topology(grid, 2)(), size(grid, 2)))

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

@inline yspacings(grid::YRegularLLG, ℓx, ℓy; with_halos=false) = yspacings(grid, ℓy; with_halos)
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

@inline λspacings(grid::LLG, ℓx::C; with_halos=false) = with_halos ? grid.classification.Δλᶜᵃᵃ : view(grid.classification.Δλᶜᵃᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx))
@inline λspacings(grid::LLG, ℓx::F; with_halos=false) = with_halos ? grid.classification.Δλᶠᵃᵃ : view(grid.classification.Δλᶠᵃᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx))
@inline λspacings(grid::XRegularLLG, ℓx::C; with_halos=false) = grid.λᶠᶠᵃ[2] - grid.λᶠᶠᵃ[1]
@inline λspacings(grid::XRegularLLG, ℓx::F; with_halos=false) = grid.λᶜᶜᵃ[2] - grid.λᶜᶜᵃ[1]

@inline φspacings(grid::LLG, ℓy::C; with_halos=false) = with_halos ? grid.classification.Δφᵃᶜᵃ : view(grid.classification.Δφᵃᶜᵃ, interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φspacings(grid::LLG, ℓy::F; with_halos=false) = with_halos ? grid.classification.Δφᵃᶠᵃ : view(grid.classification.Δφᵃᶠᵃ, interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φspacings(grid::YRegularLLG, ℓy::C; with_halos=false) = grid.φᶠᶠᵃ[2] - grid.φᶠᶠᵃ[1]
@inline φspacings(grid::YRegularLLG, ℓy::F; with_halos=false) = grid.φᶜᶜᵃ[2] - grid.φᶜᶜᵃ[1]

@inline λspacings(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = λspacings(grid, ℓx; with_halos)
@inline φspacings(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = φspacings(grid, ℓy; with_halos)

@inline λspacing(i, grid::LLG, ::C) = @inbounds grid.λᶠᶠᵃ[i+1] - grid.λᶠᶠᵃ[i]
@inline λspacing(i, grid::LLG, ::F) = @inbounds grid.λᶜᶜᵃ[i]   - grid.λᶜᶜᵃ[i-1]
@inline φspacing(j, grid::LLG, ::C) = @inbounds grid.φᶠᶠᵃ[i+1] - grid.φᶠᶠᵃ[i]
@inline φspacing(j, grid::LLG, ::F) = @inbounds grid.φᶜᶜᵃ[i]   - grid.φᶜᶜᵃ[i-1]

@inline λspacing(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = λspacing(i, grid, ℓx)
@inline φspacing(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = φspacing(j, grid, ℓy)
