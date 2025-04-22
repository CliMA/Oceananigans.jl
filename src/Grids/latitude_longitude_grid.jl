using KernelAbstractions: @kernel, @index

struct LatitudeLongitudeGrid{FT, TX, TY, TZ, Z, DXF, DXC, XF, XC, DYF, DYC, YF, YC,
                             DXCC, DXFC, DXCF, DXFF, DYFC, DYCF, Arch, I} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Z, Arch}
    architecture :: Arch
    Nx :: I
    Ny :: I
    Nz :: I
    Hx :: I
    Hy :: I
    Hz :: I
    Lx :: FT
    Ly :: FT
    Lz :: FT
    # All directions can be either regular (FX, FY, FZ) <: Number
    # or stretched (FX, FY, FZ) <: AbstractVector
    Δλᶠᵃᵃ :: DXF
    Δλᶜᵃᵃ :: DXC
    λᶠᵃᵃ  :: XF
    λᶜᵃᵃ  :: XC
    Δφᵃᶠᵃ :: DYF
    Δφᵃᶜᵃ :: DYC
    φᵃᶠᵃ  :: YF
    φᵃᶜᵃ  :: YC
    z     :: Z
    # Precomputed metrics M <: Nothing means metrics will be computed on the fly
    Δxᶜᶜᵃ :: DXCC
    Δxᶠᶜᵃ :: DXFC
    Δxᶜᶠᵃ :: DXCF
    Δxᶠᶠᵃ :: DXFF
    Δyᶠᶜᵃ :: DYFC
    Δyᶜᶠᵃ :: DYCF
    Azᶜᶜᵃ :: DXCC
    Azᶠᶜᵃ :: DXFC
    Azᶜᶠᵃ :: DXCF
    Azᶠᶠᵃ :: DXFF
    # Spherical radius
    radius :: FT
end

function LatitudeLongitudeGrid{TX, TY, TZ}(architecture::Arch,
                                           Nλ::I, Nφ::I, Nz::I, Hλ::I, Hφ::I, Hz::I,
                                           Lλ :: FT, Lφ :: FT, Lz :: FT,
                                           Δλᶠᵃᵃ :: DXF, Δλᶜᵃᵃ :: DXC,
                                            λᶠᵃᵃ :: XF,   λᶜᵃᵃ :: XC,
                                           Δφᵃᶠᵃ :: DYF, Δφᵃᶜᵃ :: DYC,
                                            φᵃᶠᵃ :: YF,   φᵃᶜᵃ :: YC, z :: Z,
                                           Δxᶜᶜᵃ :: DXCC, Δxᶠᶜᵃ :: DXFC,
                                           Δxᶜᶠᵃ :: DXCF, Δxᶠᶠᵃ :: DXFF,
                                           Δyᶠᶜᵃ :: DYFC, Δyᶜᶠᵃ :: DYCF,
                                           Azᶜᶜᵃ :: DXCC, Azᶠᶜᵃ :: DXFC,
                                           Azᶜᶠᵃ :: DXCF, Azᶠᶠᵃ :: DXFF,
                                           radius :: FT) where {Arch, FT, TX, TY, TZ, Z,
                                                                DXF, DXC, XF, XC,
                                                                DYF, DYC, YF, YC,
                                                                DXFC, DXCF,
                                                                DXFF, DXCC,
                                                                DYFC, DYCF, I}

    return LatitudeLongitudeGrid{FT, TX, TY, TZ, Z,
                                 DXF, DXC, XF, XC,
                                 DYF, DYC, YF, YC,
                                 DXCC, DXFC, DXCF, DXFF,
                                 DYFC, DYCF, Arch, I}(architecture,
                                                      Nλ, Nφ, Nz,
                                                      Hλ, Hφ, Hz,
                                                      Lλ, Lφ, Lz,
                                                      Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                      Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ, z,
                                                      Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                      Δyᶠᶜᵃ, Δyᶜᶠᵃ,
                                                      Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
end

const LLG = LatitudeLongitudeGrid

# Metrics computed on the fly (OTF), arbitrary xy spacing
#                                   ↓↓ FT     TX     TY     TZ     Z      DXF  DXC  XF     XC     DYF  DYC, YF,    YC,
const LLGOTF{DXF, DXC, DYF, DYC} = LLG{<:Any, <:Any, <:Any, <:Any, <:Any, DXF, DXC, <:Any, <:Any, DYF, DYC, <:Any, <:Any,
                                       Nothing, Nothing, Nothing, Nothing, Nothing, Nothing} where {DXF, DXC, DYF, DYC}
#                                   ↑↑ DXCC,    DXFC,    DXCF,    DXFF,    DYFC,    DYCF

# Metrics computed on the fly, constant x-spacing
const XRegLLGOTF     =  LLGOTF{<:Number, <:Number}
# Metrics computed on the fly, constant y-spacing
const YRegLLGOTF     =  LLGOTF{<:Any, <:Any, <:Number, <:Number}

# Identifying grids with various spacing patterns
#                                          ↓↓ FT     TX     TY     TZ     Z  DXF  DXC  XF     XC     DYF  DYC
const LLGSpacing{Z, DXF, DXC, DYF, DYC} = LLG{<:Any, <:Any, <:Any, <:Any, Z, DXF, DXC, <:Any, <:Any, DYF, DYC} where {Z, DXF, DXC, DYF, DYC}

const XRegularLLG    = LLGSpacing{<:Any, <:Number, <:Number}
const YRegularLLG    = LLGSpacing{<:Any, <:Any, <:Any, <:Number, <:Number}
const HRegularLLG    = LLGSpacing{<:Any, <:Number, <:Number, <:Number, <:Number}
const ZRegularLLG    = LLGSpacing{<:RegularVerticalCoordinate}
const HNonRegularLLG = LLGSpacing{<:Any, <:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}
const YNonRegularLLG = LLGSpacing{<:Any, <:Any, <:Any, <:AbstractArray, <:AbstractArray}

@inline metrics_precomputed(::LLGOTF) = false
@inline metrics_precomputed(::LLG) = true

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
                               FT::DataType = Oceananigans.defaults.FloatType;
                               size,
                               longitude = nothing,
                               latitude = nothing,
                               z = nothing,
                               radius = R_Earth,
                               topology = nothing,
                               precompute_metrics = true,
                               halo = nothing)

    topology, size, halo, latitude, longitude, z, precompute_metrics =
        validate_lat_lon_grid_args(topology, size, halo, FT, latitude, longitude, z, precompute_metrics)

    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo

    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real},
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    TX, TY, TZ = topology

    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, topology, size, halo, longitude, :longitude, 1, architecture)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, topology, size, halo, latitude,  :latitude,  2, architecture)
    Lz, z                        = generate_coordinate(FT, topology, size, halo, z,         :z,         3, architecture)

    preliminary_grid = LatitudeLongitudeGrid{TX, TY, TZ}(architecture,
                                                         Nλ, Nφ, Nz,
                                                         Hλ, Hφ, Hz,
                                                         Lλ, Lφ, Lz,
                                                         Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                         Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                         z,
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
    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ = allocate_metrics(grid)

    # Compute Δx spacings and Az areas
    arch = grid.architecture
    dev = Architectures.device(arch)
    workgroup, worksize  = metric_workgroup(grid), metric_worksize(grid)
    loop! = compute_Δx_Az!(dev, workgroup, worksize)
    loop!(grid, Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ)

    # Compute Δy spacings if needed
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
                                             grid.z,
                                             Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ,
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

    if TY() isa Periodic
        throw(ArgumentError("LatitudeLongitudeGrid cannot be Periodic in latitude!"))
    end

    # Validate longitude and latitude
    λ₁, λ₂ = get_domain_extent(longitude, Nλ)
    λ₂ - λ₁ ≤ 360 || throw(ArgumentError("Longitudinal extent cannot be greater than 360 degrees."))
    λ₁ <= λ₂      || throw(ArgumentError("Longitudes must increase west to east."))

    φ₁, φ₂ = get_domain_extent(latitude, Nφ)
    -90 <= φ₁ || throw(ArgumentError("The southernmost latitude cannot be less than -90 degrees."))
    φ₂ <= 90  || throw(ArgumentError("The northern latitude cannot be greater than 90 degrees."))
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
    Ωz = domain(TZ(), size(grid, 3), grid.z.cᵃᵃᶠ)

    x_summary = domain_summary(TX(), "λ", Ωλ)
    y_summary = domain_summary(TY(), "φ", Ωφ)
    z_summary = domain_summary(TZ(), "z", Ωz)

    longest = max(length(x_summary), length(y_summary), length(z_summary))

    x_summary = "longitude: " * dimension_summary(TX(), "λ", Ωλ, grid.Δλᶜᵃᵃ, longest - length(x_summary))
    y_summary = "latitude:  " * dimension_summary(TY(), "φ", Ωφ, grid.Δφᵃᶜᵃ, longest - length(y_summary))
    z_summary = "z:         " * dimension_summary(TZ(), "z", Ωz, grid.z,     longest - length(z_summary))

    if withsummary
        print(io, summary(grid), "\n")
    end

    return print(io, "├── ", x_summary, "\n",
                     "├── ", y_summary, "\n",
                     "└── ", z_summary)
end

@inline x_domain(grid::LLG) = domain(topology(grid, 1)(), grid.Nx, grid.λᶠᵃᵃ)
@inline y_domain(grid::LLG) = domain(topology(grid, 2)(), grid.Ny, grid.φᵃᶠᵃ)

@inline cpu_face_constructor_x(grid::XRegularLLG) = x_domain(grid)
@inline cpu_face_constructor_y(grid::YRegularLLG) = y_domain(grid)

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
                                             Adapt.adapt(to, grid.z),
                                             Adapt.adapt(to, grid.Δxᶜᶜᵃ),
                                             Adapt.adapt(to, grid.Δxᶠᶜᵃ),
                                             Adapt.adapt(to, grid.Δxᶜᶠᵃ),
                                             Adapt.adapt(to, grid.Δxᶠᶠᵃ),
                                             Adapt.adapt(to, grid.Δyᶠᶜᵃ),
                                             Adapt.adapt(to, grid.Δyᶜᶠᵃ),
                                             Adapt.adapt(to, grid.Azᶜᶜᵃ),
                                             Adapt.adapt(to, grid.Azᶠᶜᵃ),
                                             Adapt.adapt(to, grid.Azᶜᶠᵃ),
                                             Adapt.adapt(to, grid.Azᶠᶠᵃ),
                                             grid.radius)
end

#####
##### On-the-fly computation of LatitudeLongitudeGrid metrics
#####

@inline hack_cosd(φ) = cos(π * φ / 180)
@inline hack_sind(φ) = sin(π * φ / 180)

@inline Δxᶠᶜᵃ(i, j, k, grid::LLG) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ[i])
@inline Δxᶜᶠᵃ(i, j, k, grid::LLG) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ[i])
@inline Δxᶠᶠᵃ(i, j, k, grid::LLG) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ[i])
@inline Δxᶜᶜᵃ(i, j, k, grid::LLG) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ[i])
@inline Δyᶜᶠᵃ(i, j, k, grid::LLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
@inline Δyᶠᶜᵃ(i, j, k, grid::LLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])
@inline Azᶠᶜᵃ(i, j, k, grid::LLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))

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

#####
##### Kernels that precompute the z- and x-metric
#####

@inline metric_worksize(grid::LatitudeLongitudeGrid)  = (length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶠᵃ) - 2)
@inline metric_workgroup(grid::LatitudeLongitudeGrid) = (16, 16)

@inline metric_worksize(grid::XRegularLLG)  = length(grid.φᵃᶠᵃ) - 2
@inline metric_workgroup(grid::XRegularLLG) = 16

@kernel function compute_Δx_Az!(grid::LatitudeLongitudeGrid, Δxᶜᶜ, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Azᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ)
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

@kernel function compute_Δx_Az!(grid::XRegularLLG, Δxᶜᶜ, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Azᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ)
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

    Δxᶜᶜ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Δxᶠᶜ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Δxᶜᶠ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Δxᶠᶠ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Azᶜᶜ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Azᶠᶜ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Azᶜᶠ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Azᶠᶠ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)

    if grid isa YRegularLLG
        Δyᶠᶜ = Δyᶠᶜᵃ(1, 1, 1, grid)
        Δyᶜᶠ = Δyᶜᶠᵃ(1, 1, 1, grid)
    else
        parentC = zeros(arch, FT, length(grid.Δφᵃᶜᵃ))
        parentF = zeros(arch, FT, length(grid.Δφᵃᶜᵃ))
        Δyᶠᶜ    = OffsetArray(parentC, grid.Δφᵃᶜᵃ.offsets[1])
        Δyᶜᶠ    = OffsetArray(parentF, grid.Δφᵃᶜᵃ.offsets[1])
    end

    return Δxᶜᶜ, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δyᶠᶜ, Δyᶜᶠ, Azᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ
end

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

# Definitions for node
@inline ξnode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = λnode(i, grid, ℓx)
@inline ηnode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = φnode(j, grid, ℓy)
@inline xnode(i, j, grid::LLG, ℓx, ℓy) = grid.radius * deg2rad(λnode(i, grid, ℓx)) * hack_cosd((φnode(j, grid, ℓy)))
@inline ynode(j, grid::LLG, ℓy)        = grid.radius * deg2rad(φnode(j, grid, ℓy))

# Convenience definitions
@inline λnode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = λnode(i, grid, ℓx)
@inline φnode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = φnode(j, grid, ℓy)
@inline xnode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = xnode(i, j, grid, ℓx, ℓy)
@inline ynode(i, j, k, grid::LLG, ℓx, ℓy, ℓz) = ynode(j, grid, ℓy)

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

# Convenience
@inline λnodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = λnodes(grid, ℓx; with_halos)
@inline φnodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = φnodes(grid, ℓy; with_halos)
@inline xnodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = xnodes(grid, ℓx, ℓy; with_halos)
@inline ynodes(grid::LLG, ℓx, ℓy, ℓz; with_halos=false) = ynodes(grid, ℓy; with_halos)

@inline λnodes(grid::LLG, ℓx::F; with_halos=false) = _property(grid.λᶠᵃᵃ, ℓx, topology(grid, 1), size(grid, 1), with_halos)
@inline λnodes(grid::LLG, ℓx::C; with_halos=false) = _property(grid.λᶜᵃᵃ, ℓx, topology(grid, 1), size(grid, 1), with_halos)
@inline φnodes(grid::LLG, ℓy::F; with_halos=false) = _property(grid.φᵃᶠᵃ, ℓy, topology(grid, 2), size(grid, 2), with_halos)
@inline φnodes(grid::LLG, ℓy::C; with_halos=false) = _property(grid.φᵃᶜᵃ, ℓy, topology(grid, 2), size(grid, 2), with_halos)

# Generalized coordinates
@inline ξnodes(grid::LLG, ℓx; kwargs...) = λnodes(grid, ℓx; kwargs...)
@inline ηnodes(grid::LLG, ℓy; kwargs...) = φnodes(grid, ℓy; kwargs...)

@inline ξnodes(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = λnodes(grid, ℓx; kwargs...)
@inline ηnodes(grid::LLG, ℓx, ℓy, ℓz; kwargs...) = φnodes(grid, ℓy; kwargs...)

#####
##### Grid spacings
#####

@inline xspacings(grid::LLG, ℓx, ℓy) = xspacings(grid, ℓx, ℓy, nothing)
@inline yspacings(grid::LLG, ℓx, ℓy) = yspacings(grid, ℓx, ℓy, nothing)

@inline λspacings(grid::LLG, ℓx) = λspacings(grid, ℓx, nothing, nothing)
@inline φspacings(grid::LLG, ℓy) = φspacings(grid, nothing, ℓy, nothing)
