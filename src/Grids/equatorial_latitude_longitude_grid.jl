using KernelAbstractions: @kernel, @index
using OrderedCollections: OrderedDict

struct EquatorialLatitudeLongitudeGrid{FT, TX, TY, TZ, Z,
    DXF, DXC, XF, XC,
    DYF, DYC, YF, YC,
    DX1, DX2, DX3, DX4,
    DYFC, DYCF,
    AZCC, AZFC, AZCF, AZFF,
    Arch, I} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Z, Arch}

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

    Δλᶠᵃᵃ :: DXF
    Δλᶜᵃᵃ :: DXC
    λᶠᵃᵃ  :: XF
    λᶜᵃᵃ  :: XC
    Δφᵃᶠᵃ :: DYF
    Δφᵃᶜᵃ :: DYC
    φᵃᶠᵃ  :: YF
    φᵃᶜᵃ  :: YC
    z :: Z

    Δxᶜᶜᵃ :: DX1
    Δxᶠᶜᵃ :: DX2
    Δxᶜᶠᵃ :: DX3
    Δxᶠᶠᵃ :: DX4

    Δyᶠᶜᵃ :: DYFC
    Δyᶜᶠᵃ :: DYCF

    Azᶜᶜᵃ :: AZCC
    Azᶠᶜᵃ :: AZFC
    Azᶜᶠᵃ :: AZCF
    Azᶠᶠᵃ :: AZFF

    radius :: FT
end

function EquatorialLatitudeLongitudeGrid{TX, TY, TZ}(architecture::Arch,
                                           Nλ::I, Nφ::I, Nz::I, Hλ::I, Hφ::I, Hz::I,
                                           Lλ :: FT, Lφ :: FT, Lz :: FT,
                                           Δλᶠᵃᵃ :: DXF, Δλᶜᵃᵃ :: DXC,
                                            λᶠᵃᵃ :: XF,   λᶜᵃᵃ :: XC,
                                           Δφᵃᶠᵃ :: DYF, Δφᵃᶜᵃ :: DYC,
                                            φᵃᶠᵃ :: YF,   φᵃᶜᵃ :: YC, z :: Z,
                                           Δxᶜᶜᵃ :: DX1, Δxᶠᶜᵃ :: DX2,
                                           Δxᶜᶠᵃ :: DX3, Δxᶠᶠᵃ :: DX4,
                                           Δyᶠᶜᵃ :: DYFC, Δyᶜᶠᵃ :: DYCF,
                                           Azᶜᶜᵃ :: AZCC, Azᶠᶜᵃ :: AZFC,
                                           Azᶜᶠᵃ :: AZCF, Azᶠᶠᵃ :: AZFF,
                                           radius :: FT) where {Arch, FT, TX, TY, TZ, Z,
                                                                DXF, DXC, XF, XC,
                                                                DYF, DYC, YF, YC,
                                                                DX1, DX2, DX3, DX4,
                                                                DYFC, DYCF,
                                                                AZCC, AZFC, AZCF, AZFF,
                                                                I}

    return EquatorialLatitudeLongitudeGrid{FT, TX, TY, TZ, Z,
        DXF, DXC, XF, XC,
        DYF, DYC, YF, YC,
        DX1, DX2, DX3, DX4,
        DYFC, DYCF,
        AZCC, AZFC, AZCF, AZFF,
        Arch, I}(architecture,
                 Nλ, Nφ, Nz,
                 Hλ, Hφ, Hz,
                 Lλ, Lφ, Lz,
                 Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                 Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ, z,
                 Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                 Δyᶠᶜᵃ, Δyᶜᶠᵃ,
                 Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                 radius)
end

const ELLG = EquatorialLatitudeLongitudeGrid

# Metrics computed on the fly (OTF), arbitrary xy spacing
#                                   ↓↓ FT     TX     TY     TZ     Z      DXF  DXC  XF     XC     DYF  DYC, YF,    YC,
const ELLGOTF{DXF, DXC, DYF, DYC} = ELLG{<:Any, <:Any, <:Any, <:Any, <:Any, DXF, DXC, <:Any, <:Any, DYF, DYC, <:Any, <:Any,
                                       Nothing, Nothing, Nothing, Nothing, Nothing, Nothing} where {DXF, DXC, DYF, DYC}
#                                   ↑↑ DXCC,    DXFC,    DXCF,    DXFF,    DYFC,    DYCF

# Metrics computed on the fly, constant x-spacing
const XRegELLGOTF     =  ELLGOTF{<:Number, <:Number}
# Metrics computed on the fly, constant y-spacing
const YRegELLGOTF     =  ELLGOTF{<:Any, <:Any, <:Number, <:Number}

# Identifying grids with various spacing patterns
#                                          ↓↓ FT     TX     TY     TZ     Z  DXF  DXC  XF     XC     DYF  DYC
const ELLGSpacing{Z, DXF, DXC, DYF, DYC} = ELLG{<:Any, <:Any, <:Any, <:Any, Z, DXF, DXC, <:Any, <:Any, DYF, DYC} where {Z, DXF, DXC, DYF, DYC}

const XRegularELLG    = ELLGSpacing{<:Any, <:Number, <:Number}
const YRegularELLG    = ELLGSpacing{<:Any, <:Any, <:Any, <:Number, <:Number}
const HRegularELLG    = ELLGSpacing{<:Any, <:Number, <:Number, <:Number, <:Number}
const ZRegularELLG    = ELLGSpacing{<:RegularVerticalCoordinate}
const HNonRegularELLG = ELLGSpacing{<:Any, <:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}
const YNonRegularELLG = ELLGSpacing{<:Any, <:Any, <:Any, <:AbstractArray, <:AbstractArray}

@inline metrics_precomputed(::ELLGOTF) = false
@inline metrics_precomputed(::ELLG) = true

regular_dimensions(::ZRegularELLG) = tuple(3)

"""
    EquatorialLatitudeLongitudeGrid([architecture = CPU(), FT = Oceananigans.defaults.FloatType];
                          size,
                          longitude,
                          latitude,
                          z = nothing,
                          radius = Oceananigans.defaults.planet_radius,
                          topology = nothing,
                          precompute_metrics = true,
                          halo = nothing)

Creates an `EquatorialLatitudeLongitudeGrid` with coordinates `(λ, φ, z)` denoting longitude, latitude,
and vertical coordinate respectively. Note that the North and South poles occur in the tropics in this grid.

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
  3. single-argument function that takes an index and returns cell interface location.

  **Note**: the latitude and longitude coordinates extents are expected in degrees.

- `radius`: The radius of the sphere the grid lives on. By default is equal to the radius of Earth.

- `topology`: Tuple of topologies (`Flat`, `Bounded`, `Periodic`) for each direction. The vertical
              `topology[3]` must be `Bounded`, while the latitude-longitude topologies can be
              `Bounded`, `Periodic`, or `Flat`. If no topology is provided then, by default, the
              topology is (`Periodic`, `Bounded`, `Bounded`) if the longitudinal extent is 360 degrees
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

julia> grid = EquatorialLatitudeLongitudeGrid(size=(36, 34, 25),
                                    longitude = (-180, 180),
                                    latitude = (-85, 85),
                                    z = (-1000, 0))
36×34×25 EquatorialLatitudeLongitudeGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo
├── longitude: Periodic λ ∈ [-180.0, 180.0) regularly spaced with Δλ=10.0
├── latitude:  Bounded  φ ∈ [-85.0, 85.0]   regularly spaced with Δφ=5.0
└── z:         Bounded  z ∈ [-1000.0, 0.0]  regularly spaced with Δz=40.0
```

* A bounded spherical sector with cell interfaces stretched hyperbolically near the top:

```jldoctest
using Oceananigans

σ = 1.1 # stretching factor
Nz = 24 # vertical resolution
Lz = 1000 # depth (m)
hyperbolically_spaced_faces(k) = - Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ))

grid = EquatorialLatitudeLongitudeGrid(size=(36, 34, Nz),
                             longitude = (-180, 180),
                             latitude = (-20, 20),
                             z = hyperbolically_spaced_faces,
                             topology = (Bounded, Bounded, Bounded))

# output

36×34×24 EquatorialLatitudeLongitudeGrid{Float64, Bounded, Bounded, Bounded} on CPU with 3×3×3 halo
├── longitude: Bounded  λ ∈ [-180.0, 180.0] regularly spaced with Δλ=10.0
├── latitude:  Bounded  φ ∈ [-20.0, 20.0]   regularly spaced with Δφ=1.17647
└── z:         Bounded  z ∈ [-1000.0, -0.0] variably spaced with min(Δz)=21.3342, max(Δz)=57.2159
```
"""
function EquatorialLatitudeLongitudeGrid(architecture::AbstractArchitecture = CPU(),
                               FT::DataType = Oceananigans.defaults.FloatType;
                               size,
                               longitude = nothing,
                               latitude = nothing,
                               z = nothing,
                               radius = Oceananigans.defaults.planet_radius,
                               topology = nothing,
                               precompute_metrics = true,
                               halo = nothing)

    topology, size, halo, latitude, longitude, z, precompute_metrics =
        validate_equ_lat_lon_grid_args(topology, size, halo, FT, latitude, longitude, z, precompute_metrics)

    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo

    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real},
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    TX, TY, TZ = topology

    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, topology, size, halo, longitude, :longitude, 1, architecture)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, topology, size, halo, latitude,  :latitude,  2, architecture)
    Lz, z                        = generate_coordinate(FT, topology, size, halo, z,         :z,         3, architecture)

    preliminary_grid = EquatorialLatitudeLongitudeGrid{TX, TY, TZ}(architecture,
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
EquatorialLatitudeLongitudeGrid(FT::DataType; kwargs...) = EquatorialLatitudeLongitudeGrid(CPU(), FT; kwargs...)

""" Return a reproduction of `grid` with precomputed metric terms. """
function with_precomputed_metrics(grid::EquatorialLatitudeLongitudeGrid)

    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
    Δyᶠᶜᵃ, Δyᶜᶠᵃ,
    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ = allocate_metrics(grid)

    arch = architecture(grid)
    dev  = Architectures.device(arch)

    # Δx and Az
    workgroup, worksize = metric_workgroup(grid), metric_worksize(grid)
    compute_Δx_Az!(dev, workgroup, worksize)(
        grid, Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
              Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ
    )

    # Δy (always 2D for this grid)
    compute_Δy!(dev, (16, 16), (length(grid.λᶜᵃᵃ), length(grid.φᵃᶜᵃ) - 1))(grid, Δyᶠᶜᵃ, Δyᶜᶠᵃ)

    Nλ, Nφ, Nz = size(grid)
    Hλ, Hφ, Hz = halo_size(grid)
    TX, TY, TZ = topology(grid)

    return EquatorialLatitudeLongitudeGrid{TX, TY, TZ}(
        arch,
        Nλ, Nφ, Nz,
        Hλ, Hφ, Hz,
        grid.Lx, grid.Ly, grid.Lz,
        grid.Δλᶠᵃᵃ, grid.Δλᶜᵃᵃ, grid.λᶠᵃᵃ, grid.λᶜᵃᵃ,
        grid.Δφᵃᶠᵃ, grid.Δφᵃᶜᵃ, grid.φᵃᶠᵃ, grid.φᵃᶜᵃ,
        grid.z,
        Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
        Δyᶠᶜᵃ, Δyᶜᶠᵃ,
        Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
        grid.radius
    )
end

function validate_equ_lat_lon_grid_args(topology, size, halo, FT, latitude, longitude, z, precompute_metrics)
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
    λ₂ - λ₁ ≤ 360 + 10 * eps(FT(360)) || throw(ArgumentError("Longitudinal extent cannot be greater than 360 degrees."))
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

function Base.summary(grid::EquatorialLatitudeLongitudeGrid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)

    return string(size_summary(grid),
                  " EquatorialLatitudeLongitudeGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo")
end

function Base.show(io::IO, grid::EquatorialLatitudeLongitudeGrid, withsummary=true)
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

@inline x_domain(grid::ELLG) = domain(topology(grid, 1)(), grid.Nx, grid.λᶠᵃᵃ)
@inline y_domain(grid::ELLG) = domain(topology(grid, 2)(), grid.Ny, grid.φᵃᶠᵃ)

@inline cpu_face_constructor_x(grid::XRegularELLG) = x_domain(grid)
@inline cpu_face_constructor_y(grid::YRegularELLG) = y_domain(grid)

function constructor_arguments(grid::EquatorialLatitudeLongitudeGrid)
    arch = architecture(grid)
    args = OrderedDict(:architecture => arch, :number_type => eltype(grid))

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

function Base.similar(grid::EquatorialLatitudeLongitudeGrid)
    args, kwargs = constructor_arguments(grid)
    arch = args[:architecture]
    FT = args[:number_type]
    return EquatorialLatitudeLongitudeGrid(arch, FT; kwargs...)
end

function with_number_type(FT, grid::EquatorialLatitudeLongitudeGrid)
    args, kwargs = constructor_arguments(grid)
    arch = args[:architecture]
    return EquatorialLatitudeLongitudeGrid(arch, FT; kwargs...)
end

function with_halo(halo, grid::EquatorialLatitudeLongitudeGrid)
    args, kwargs = constructor_arguments(grid)
    halo = pop_flat_elements(halo, topology(grid))
    kwargs[:halo] = halo
    arch = args[:architecture]
    FT = args[:number_type]
    return EquatorialLatitudeLongitudeGrid(arch, FT; kwargs...)
end

function Architectures.on_architecture(arch::AbstractSerialArchitecture, grid::EquatorialLatitudeLongitudeGrid)
    if arch == architecture(grid)
        return grid
    end

    args, kwargs = constructor_arguments(grid)
    FT = args[:number_type]
    return EquatorialLatitudeLongitudeGrid(arch, FT; kwargs...)
end

function Adapt.adapt_structure(to, grid::EquatorialLatitudeLongitudeGrid)
    TX, TY, TZ = topology(grid)
    return EquatorialLatitudeLongitudeGrid{TX, TY, TZ}(nothing,
                                             grid.Nx, grid.Ny, grid.Nz,
                                             grid.Hx, grid.Hy, grid.Hz,
                                             Adapt.adapt(to, grid.Lx),
                                             Adapt.adapt(to, grid.Ly),
                                             Adapt.adapt(to, grid.Lz),
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
                                             Adapt.adapt(to, grid.radius))
end

#####
##### On-the-fly computation of LatitudeLongitudeGrid metrics
#####

@inline Δxᶠᶜᵃ(i, j, k, grid::ELLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])
@inline Δxᶜᶠᵃ(i, j, k, grid::ELLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
@inline Δxᶠᶠᵃ(i, j, k, grid::ELLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
@inline Δxᶜᶜᵃ(i, j, k, grid::ELLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])
@inline Δyᶜᶠᵃ(i, j, k, grid::ELLG) = @inbounds grid.radius * deg2rad(grid.Δλᶜᵃᵃ[i]) * hack_cosd(grid.φᵃᶠᵃ[j])
@inline Δyᶠᶜᵃ(i, j, k, grid::ELLG) = @inbounds grid.radius * deg2rad(grid.Δλᶠᵃᵃ[i]) * hack_cosd(grid.φᵃᶜᵃ[j])
@inline Azᶠᶜᵃ(i, j, k, grid::ELLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::ELLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::ELLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::ELLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))

#@inline Δxᶠᶜᵃ(i, j, k, grid::XRegularELLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])
#@inline Δxᶜᶠᵃ(i, j, k, grid::XRegularELLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
#@inline Δxᶠᶠᵃ(i, j, k, grid::XRegularELLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
#@inline Δxᶜᶜᵃ(i, j, k, grid::XRegularELLG) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])
#@inline Δyᶜᶠᵃ(i, j, k, grid::YRegularELLG) = @inbounds grid.radius * deg2rad(grid.Δλᶜᵃᵃ[i]) * hack_cosd(grid.φᵃᶠᵃ[j])
#@inline Δyᶠᶜᵃ(i, j, k, grid::YRegularELLG) = @inbounds grid.radius * deg2rad(grid.Δλᶠᵃᵃ[i]) * hack_cosd(grid.φᵃᶜᵃ[j])
#@inline Azᶠᶜᵃ(i, j, k, grid::XRegularELLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
#@inline Azᶜᶠᵃ(i, j, k, grid::XRegularELLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
#@inline Azᶠᶠᵃ(i, j, k, grid::XRegularELLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
#@inline Azᶜᶜᵃ(i, j, k, grid::XRegularELLG) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))

#####
##### Utilities to precompute metrics
#####

#####
##### Kernels that precompute the z- and x-metric
#####

@inline metric_worksize(grid::EquatorialLatitudeLongitudeGrid) = (length(grid.λᶜᵃᵃ), length(grid.φᵃᶠᵃ) - 2)
@inline metric_workgroup(grid::EquatorialLatitudeLongitudeGrid) = (16, 16)

#@inline metric_worksize(grid::XRegularELLG) = (1, length(grid.φᵃᶠᵃ) - 2)
#@inline metric_workgroup(grid::XRegularELLG) = 16

@kernel function compute_Δx_Az!(grid::EquatorialLatitudeLongitudeGrid,
                               Δxᶜᶜ, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ,
                               Azᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ)

    i, j = @index(Global, NTuple)

    i′ = i + grid.λᶜᵃᵃ.offsets[1]
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
    i, j = @index(Global, NTuple)

    i′ = i + grid.λᶜᵃᵃ.offsets[1]
    j′ = j + grid.φᵃᶜᵃ.offsets[1] + 1

    @inbounds begin
        Δyᶜᶠ[i′, j′] = Δyᶜᶠᵃ(i′, j′, 1, grid)
        Δyᶠᶜ[i′, j′] = Δyᶠᶜᵃ(i′, j′, 1, grid)
    end
end

#####
##### Metric memory allocation
#####

function allocate_metrics(grid::EquatorialLatitudeLongitudeGrid)
    FT = eltype(grid)
    arch = grid.architecture

    if grid isa XRegularLLG
        offsets     = grid.φᵃᶜᵃ.offsets[1]
        metric_size = length(grid.φᵃᶜᵃ)
    else
        offsets     = (grid.λᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
        metric_size = (length(grid.λᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    end

    parentC = zeros(arch, FT, length(grid.φᵃᶜᵃ))
    parentF = zeros(arch, FT, length(grid.φᵃᶜᵃ))

    Δxᶠᶜ = OffsetArray(parentC, grid.φᵃᶜᵃ.offsets[1])
    Δxᶜᶠ = OffsetArray(parentF, grid.φᵃᶜᵃ.offsets[1])
    Δxᶜᶜ = Δxᶠᶜ
    Δxᶠᶠ = Δxᶜᶠ

    offsets     = (grid.λᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    metric_size = (length(grid.λᶜᵃᵃ), length(grid.φᵃᶜᵃ))

    parentC = zeros(arch, FT, metric_size...)
    parentF = zeros(arch, FT, metric_size...)

    Δyᶠᶜ = OffsetArray(parentC, offsets...)
    Δyᶜᶠ = OffsetArray(parentF, offsets...)

    offsets     = (grid.λᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    metric_size = (length(grid.λᶜᵃᵃ), length(grid.φᵃᶜᵃ))

    parentAzᶜᶜ = zeros(arch, FT, metric_size...)
    parentAzᶠᶜ = zeros(arch, FT, metric_size...)
    parentAzᶜᶠ = zeros(arch, FT, metric_size...)
    parentAzᶠᶠ = zeros(arch, FT, metric_size...)

    Azᶜᶜ = OffsetArray(parentAzᶜᶜ, offsets...)
    Azᶠᶜ = OffsetArray(parentAzᶠᶜ, offsets...)
    Azᶜᶠ = OffsetArray(parentAzᶜᶠ, offsets...)
    Azᶠᶠ = OffsetArray(parentAzᶠᶠ, offsets...)

    return Δxᶜᶜ, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δyᶠᶜ, Δyᶜᶠ, Azᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ
end

#####
##### Grid nodes
#####

ξname(::ELLG) = :λ
ηname(::ELLG) = :φ
rname(::ELLG) = :z

@inline λnode(i, grid::ELLG, ::Center) = getnode(grid.λᶜᵃᵃ, i)
@inline λnode(i, grid::ELLG, ::Face)   = getnode(grid.λᶠᵃᵃ, i)
@inline φnode(j, grid::ELLG, ::Center) = getnode(grid.φᵃᶜᵃ, j)
@inline φnode(j, grid::ELLG, ::Face)   = getnode(grid.φᵃᶠᵃ, j)

# Definitions for node
@inline ξnode(i, j, k, grid::ELLG, ℓx, ℓy, ℓz) = λnode(i, grid, ℓx)
@inline ηnode(i, j, k, grid::ELLG, ℓx, ℓy, ℓz) = φnode(j, grid, ℓy)
@inline xnode(i, j, grid::ELLG, ℓx, ℓy) = grid.radius * deg2rad(λnode(i, grid, ℓx)) * hack_cosd((φnode(j, grid, ℓy)))
@inline ynode(j, grid::ELLG, ℓy)        = grid.radius * deg2rad(φnode(j, grid, ℓy))

# Convenience definitions
@inline λnode(i, j, k, grid::ELLG, ℓx, ℓy, ℓz) = λnode(i, grid, ℓx)
@inline φnode(i, j, k, grid::ELLG, ℓx, ℓy, ℓz) = φnode(j, grid, ℓy)
@inline xnode(i, j, k, grid::ELLG, ℓx, ℓy, ℓz) = xnode(i, j, grid, ℓx, ℓy)
@inline ynode(i, j, k, grid::ELLG, ℓx, ℓy, ℓz) = ynode(j, grid, ℓy)

function nodes(grid::ELLG, ℓx, ℓy, ℓz; reshape=false, with_halos=false, indices=(Colon(), Colon(), Colon()))
    λ = λnodes(grid, ℓx, ℓy, ℓz; with_halos, indices = indices[1])
    φ = φnodes(grid, ℓx, ℓy, ℓz; with_halos, indices = indices[2])
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
const N = Nothing

@inline function xnodes(grid::ELLG, ℓx, ℓy; with_halos=false)
    λ = λnodes(grid, ℓx; with_halos=with_halos)'
    φ = φnodes(grid, ℓy; with_halos=with_halos)
    R = grid.radius
    return @. R * deg2rad(λ) * hack_cosd(φ)
end

@inline function ynodes(grid::ELLG, ℓy; with_halos=false)
    φ = φnodes(grid, ℓy; with_halos=with_halos)
    R = grid.radius
    return @. R * deg2rad(φ)
end

# Convenience
@inline λnodes(grid::ELLG, ℓx, ℓy, ℓz; with_halos=false, indices=Colon()) = λnodes(grid, ℓx; with_halos, indices)
@inline φnodes(grid::ELLG, ℓx, ℓy, ℓz; with_halos=false, indices=Colon()) = φnodes(grid, ℓy; with_halos, indices)
@inline xnodes(grid::ELLG, ℓx, ℓy, ℓz; with_halos=false) = xnodes(grid, ℓx, ℓy; with_halos)
@inline ynodes(grid::ELLG, ℓx, ℓy, ℓz; with_halos=false) = ynodes(grid, ℓy; with_halos)

@inline λnodes(grid::ELLG, ℓx::F; with_halos=false, indices=Colon()) = view(_property(grid.λᶠᵃᵃ, ℓx, topology(grid, 1), grid.Nx, grid.Hx, with_halos), indices)
@inline λnodes(grid::ELLG, ℓx::C; with_halos=false, indices=Colon()) = view(_property(grid.λᶜᵃᵃ, ℓx, topology(grid, 1), grid.Nx, grid.Hx, with_halos), indices)
@inline λnodes(grid::ELLG, ℓx::N; with_halos=false, indices=Colon()) = nothing
@inline φnodes(grid::ELLG, ℓy::F; with_halos=false, indices=Colon()) = view(_property(grid.φᵃᶠᵃ, ℓy, topology(grid, 2), grid.Ny, grid.Hy, with_halos), indices)
@inline φnodes(grid::ELLG, ℓy::C; with_halos=false, indices=Colon()) = view(_property(grid.φᵃᶜᵃ, ℓy, topology(grid, 2), grid.Ny, grid.Hy, with_halos), indices)
@inline φnodes(grid::ELLG, ℓy::N; with_halos=false, indices=Colon()) = nothing

# Flat topologies
#XFlatELLG = EquatorialLatitudeLongitudeGrid{<:Any, Flat}
#YFlatELLG = EquatorialLatitudeLongitudeGrid{<:Any, <:Any, Flat}
const XFlatELLG = EquatorialLatitudeLongitudeGrid{<:Any, Flat, <:Any, <:Any}
const YFlatELLG = EquatorialLatitudeLongitudeGrid{<:Any, <:Any, Flat, <:Any}

@inline λnodes(grid::XFlatELLG, ℓx::F; with_halos=false, indices=Colon()) = _property(grid.λᶜᵃᵃ, ℓx, topology(grid, 1), grid.Nx, grid.Hx, with_halos)
@inline λnodes(grid::XFlatELLG, ℓx::C; with_halos=false, indices=Colon()) = _property(grid.λᶜᵃᵃ, ℓx, topology(grid, 1), grid.Nx, grid.Hx, with_halos)
@inline φnodes(grid::YFlatELLG, ℓy::F; with_halos=false, indices=Colon()) = _property(grid.φᵃᶠᵃ, ℓy, topology(grid, 2), grid.Ny, grid.Hy, with_halos)
@inline φnodes(grid::YFlatELLG, ℓy::C; with_halos=false, indices=Colon()) = _property(grid.φᵃᶜᵃ, ℓy, topology(grid, 2), grid.Ny, grid.Hy, with_halos)

# Generalized coordinates
@inline ξnodes(grid::ELLG, ℓx; kwargs...) = λnodes(grid, ℓx; kwargs...)
@inline ηnodes(grid::ELLG, ℓy; kwargs...) = φnodes(grid, ℓy; kwargs...)

@inline ξnodes(grid::ELLG, ℓx, ℓy, ℓz; kwargs...) = λnodes(grid, ℓx; kwargs...)
@inline ηnodes(grid::ELLG, ℓx, ℓy, ℓz; kwargs...) = φnodes(grid, ℓy; kwargs...)

#####
##### Grid spacings
#####

@inline xspacings(grid::ELLG, ℓx, ℓy) = xspacings(grid, ℓx, ℓy, nothing)
@inline yspacings(grid::ELLG, ℓx, ℓy) = yspacings(grid, ℓx, ℓy, nothing)

@inline λspacings(grid::ELLG, ℓx) = λspacings(grid, ℓx, nothing, nothing)
@inline φspacings(grid::ELLG, ℓy) = φspacings(grid, nothing, ℓy, nothing)

"""
    EquatorialLatitudeLongitudeGrid(rectilinear_grid::RectilinearGrid;
                          radius = Oceananigans.defaults.planet_radius,
                          origin = (0, 0))

Construct an `EquatorialLatitudeLongitudeGrid` from a `RectilinearGrid`. The horizontal coordinates of the
rectilinear grid are transformed to longitude-latitude coordinates in degrees, accounting for
spherical Earth geometry. The longitudes are computed approximately using the latitudinal origin.

The vertical coordinate and architecture are inherited from the input grid.

Keyword Arguments
================
- `radius`: The radius of the sphere, defaults to Earth's mean radius (≈ 6371 km)
- `origin`: Tuple of (longitude, latitude) in degrees specifying the origin of the rectilinear grid
"""
function EquatorialLatitudeLongitudeGrid(rectilinear_grid::RectilinearGrid;
                               radius = Oceananigans.defaults.planet_radius,
                               origin = (0, 0))

    arch = architecture(rectilinear_grid)
    Hx, Hy, Hz = halo_size(rectilinear_grid)
    Nx, Ny, Nz = size(rectilinear_grid)

    λ₀, φ₀ = origin

    TX, TY, TZ = topology(rectilinear_grid)
    tx, ty, tz = TX(), TY(), TZ()
    triply_bounded = tx isa Bounded && ty isa Bounded && tz isa Bounded
    if !triply_bounded
        msg = string("The source RectilinearGrid for constructing EquatorialLatitudeLongitudeGrid ",
                     "must be triply-bounded, but has topology=($tx, $ty, $tz)!")
        throw(ArgumentError(msg))
    end

    # Get face coordinates from rectilinear grid
    xᶠ = xnodes(rectilinear_grid, Face())
    yᶠ = ynodes(rectilinear_grid, Face())

    xᶠ = on_architecture(CPU(), xᶠ)
    yᶠ = on_architecture(CPU(), yᶠ)

    # Convert y coordinates to latitudes
    R = radius
    φᶠ = @. φ₀ + 180 / π * yᶠ / R

    # Convert x to longitude, using the origin as a reference
    λᶠ = @. λ₀ + 180 / π * xᶠ / (R * cosd(φ₀))

    z = cpu_face_constructor_z(rectilinear_grid)

    return EquatorialLatitudeLongitudeGrid(arch, eltype(rectilinear_grid); z, radius,
                                 topology = (Bounded, Bounded, Bounded),
                                 size = (Nx, Ny, Nz),
                                 halo = (Hx, Hy, Hz),
                                 longitude = λᶠ,
                                 latitude = φᶠ)
end

is_lat_lon_grid(::EquatorialLatitudeLongitudeGrid) = true
