using CubedSphere
using JLD2
using OffsetArrays
using Adapt
using Distances

using Adapt: adapt_structure

using Oceananigans
using Oceananigans.Grids: prettysummary, coordinate_summary, BoundedTopology, length

struct OrthogonalSphericalShellGrid{FT, TX, TY, TZ, A, R, FR, C, Arch} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    Nx :: Int
    Ny :: Int
    Nz :: Int
    Hx :: Int
    Hy :: Int
    Hz :: Int
    Lz :: FT
    λᶜᶜᵃ :: A
    λᶠᶜᵃ :: A
    λᶜᶠᵃ :: A
    λᶠᶠᵃ :: A
    φᶜᶜᵃ :: A
    φᶠᶜᵃ :: A
    φᶜᶠᵃ :: A
    φᶠᶠᵃ :: A
    zᵃᵃᶜ :: R
    zᵃᵃᶠ :: R
    Δxᶜᶜᵃ :: A
    Δxᶠᶜᵃ :: A
    Δxᶜᶠᵃ :: A
    Δxᶠᶠᵃ :: A
    Δyᶜᶜᵃ :: A
    Δyᶜᶠᵃ :: A
    Δyᶠᶜᵃ :: A
    Δyᶠᶠᵃ :: A
    Δzᵃᵃᶜ :: FR
    Δzᵃᵃᶠ :: FR
    Azᶜᶜᵃ :: A
    Azᶠᶜᵃ :: A
    Azᶜᶠᵃ :: A
    Azᶠᶠᵃ :: A
    radius :: FT
    conformal_mapping :: C

    OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture::Arch,
                                             Nx, Ny, Nz,
                                             Hx, Hy, Hz,
                                                Lz :: FT,
                                              λᶜᶜᵃ :: A,  λᶠᶜᵃ :: A,  λᶜᶠᵃ :: A,  λᶠᶠᵃ :: A,
                                              φᶜᶜᵃ :: A,  φᶠᶜᵃ :: A,  φᶜᶠᵃ :: A,  φᶠᶠᵃ :: A, zᵃᵃᶜ :: R, zᵃᵃᶠ :: R,
                                             Δxᶜᶜᵃ :: A, Δxᶠᶜᵃ :: A, Δxᶜᶠᵃ :: A, Δxᶠᶠᵃ :: A,
                                             Δyᶜᶜᵃ :: A, Δyᶜᶠᵃ :: A, Δyᶠᶜᵃ :: A, Δyᶠᶠᵃ :: A, Δzᵃᵃᶜ :: FR, Δzᵃᵃᶠ :: FR,
                                             Azᶜᶜᵃ :: A, Azᶠᶜᵃ :: A, Azᶜᶠᵃ :: A, Azᶠᶠᵃ :: A,
                                             radius :: FT,
                                             conformal_mapping :: C) where {TX, TY, TZ, FT, A, R, FR, C, Arch} =
        new{FT, TX, TY, TZ, A, R, FR, C, Arch}(architecture,
                                            Nx, Ny, Nz,
                                            Hx, Hy, Hz,
                                            Lz,
                                            λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                            φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
                                            Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                            Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                                            Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius, conformal_mapping)
end

const OSSG = OrthogonalSphericalShellGrid
const ZRegOSSG = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const ZRegOrthogonalSphericalShellGrid = ZRegOSSG

# convenience constructor for OSSG without any conformal_mapping properties
OrthogonalSphericalShellGrid(architecture, Nx, Ny, Nz, Hx, Hy, Hz, Lz,
                             λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ, φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
                             Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                             Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius) =
    OrthogonalSphericalShellGrid(architecture, Nx, Ny, Nz, Hx, Hy, Hz, Lz,
                                 λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ, φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
                                 Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                                 Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius, nothing)

"""
    conformal_cubed_sphere_panel(architecture::AbstractArchitecture = CPU(),
                                 FT::DataType = Float64;
                                 size,
                                 z,
                                 topology = (Bounded, Bounded, Bounded),
                                 ξ = (-1, 1),
                                 η = (-1, 1),
                                 radius = R_Earth,
                                 halo = (1, 1, 1),
                                 rotation = nothing)

Create a `OrthogonalSphericalShellGrid` that represents a section of a sphere after it has been 
conformally mapped from the face of a cube. The cube's coordinates are `ξ` and `η` (which, by default,
both take values in the range ``[-1, 1]``.

The mapping from the face of the cube to the sphere is done via the [CubedSphere.jl](https://github.com/CliMA/CubedSphere.jl)
package.

Positional arguments
====================

- `architecture`: Specifies whether arrays of coordinates and spacings are stored
                  on the CPU or GPU. Default: `CPU()`.

- `FT` : Floating point data type. Default: `Float64`.

Keyword arguments
=================

- `size` (required): A 3-tuple prescribing the number of grid points each direction.

- `z` (required): Either a
    1. 2-tuple that specify the end points of the ``z``-domain,
    2. one-dimensional array specifying the cell interface locations, or
    3. a single-argument function that takes an index and returns cell interface location.

- `radius`: The radius of the sphere the grid lives on. By default this is equal to the radius of Earth.

- `halo`: A 3-tuple of integers specifying the size of the halo region of cells surrounding
          the physical interior. The default is 1 halo cells in every direction.

- `rotation :: Rotation`: Rotation of the conformal cubed sphere panel about some axis that passes
                          through the center of the sphere. If `nothing` is provided (default), then
                          the panel includes the North Pole of the sphere in its center. For example,
                          to construct a grid that includes tha South Pole we can pass either
                          `rotation = RotX(π)` or `rotation = RotY(π)`.

Examples
========

* The default conformal cubed sphere panel grid with `Float64` type:

```jldoctest
julia> using Oceananigans, Oceananigans.Grids

julia> grid = conformal_cubed_sphere_panel(size=(36, 34, 25), z=(-1000, 0))
36×34×25 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 1×1×1 halo and with precomputed metrics
├── centered at: North Pole, (λ, φ) = (0.0, 90.0)
├── longitude: Bounded  extent 90.0 degrees variably spaced with min(Δλ)=0.616164, max(Δλ)=2.58892
├── latitude:  Bounded  extent 90.0 degrees variably spaced with min(Δφ)=0.664958, max(Δφ)=2.74119
└── z:         Bounded  z ∈ [-1000.0, 0.0]  regularly spaced with Δz=40.0
```

* The conformal cubed sphere panel that includes the South Pole with `Float32` type:

```jldoctest
julia> using Oceananigans, Oceananigans.Grids, Rotations

julia> grid = conformal_cubed_sphere_panel(Float32, size=(36, 34, 25), z=(-1000, 0), rotation=RotY(π))
36×34×25 OrthogonalSphericalShellGrid{Float32, Bounded, Bounded, Bounded} on CPU with 1×1×1 halo and with precomputed metrics
├── centered at: South Pole, (λ, φ) = (0.0, -90.0)
├── longitude: Bounded  extent 90.0 degrees variably spaced with min(Δλ)=0.616167, max(Δλ)=2.58891
├── latitude:  Bounded  extent 90.0 degrees variably spaced with min(Δφ)=0.664956, max(Δφ)=2.7412
└── z:         Bounded  z ∈ [-1000.0, 0.0]  regularly spaced with Δz=40.0
```
"""
function conformal_cubed_sphere_panel(architecture::AbstractArchitecture = CPU(),
                                      FT::DataType = Float64;
                                      size,
                                      z,
                                      topology = (Bounded, Bounded, Bounded),
                                      ξ = (-1, 1),
                                      η = (-1, 1),
                                      radius = R_Earth,
                                      halo = (1, 1, 1),
                                      rotation = nothing)

    if architecture == GPU() && !has_cuda() 
        throw(ArgumentError("Cannot create a GPU grid. No CUDA-enabled GPU was detected!"))
    end

    radius = FT(radius)

    TX, TY, TZ = topology
    Nξ, Nη, Nz = size
    Hx, Hy, Hz = halo

    ## Use a regular rectilinear grid for the face of the cube

    ξη_grid_topology = (Bounded, Bounded, topology[3])

    # construct the grid on CPU and convert to architecture later
    ξη_grid = RectilinearGrid(CPU(), FT;
                              size=(Nξ, Nη, Nz),
                              topology = ξη_grid_topology,
                              x=ξ, y=η, z, halo)

    ξᶠᵃᵃ = xnodes(ξη_grid, Face())
    ξᶜᵃᵃ = xnodes(ξη_grid, Center())
    ηᵃᶠᵃ = ynodes(ξη_grid, Face())
    ηᵃᶜᵃ = ynodes(ξη_grid, Center())

    ## The vertical coordinates and metrics can come out of the regular rectilinear grid!
     zᵃᵃᶠ = ξη_grid.zᵃᵃᶠ
     zᵃᵃᶜ = ξη_grid.zᵃᵃᶜ
    Δzᵃᵃᶜ = ξη_grid.Δzᵃᵃᶜ
    Δzᵃᵃᶠ = ξη_grid.Δzᵃᵃᶠ
    Lz    = ξη_grid.Lz


    ## Compute staggered grid latitude-longitude (φ, λ) coordinates.

    λᶜᶜᵃ = zeros(FT, Nξ  , Nη  )
    λᶠᶜᵃ = zeros(FT, Nξ+1, Nη  )
    λᶜᶠᵃ = zeros(FT, Nξ  , Nη+1)
    λᶠᶠᵃ = zeros(FT, Nξ+1, Nη+1)

    φᶜᶜᵃ = zeros(FT, Nξ  , Nη  )
    φᶠᶜᵃ = zeros(FT, Nξ+1, Nη  )
    φᶜᶠᵃ = zeros(FT, Nξ  , Nη+1)
    φᶠᶠᵃ = zeros(FT, Nξ+1, Nη+1)

    ξS = (ξᶜᵃᵃ, ξᶠᵃᵃ, ξᶜᵃᵃ, ξᶠᵃᵃ)
    ηS = (ηᵃᶜᵃ, ηᵃᶜᵃ, ηᵃᶠᵃ, ηᵃᶠᵃ)
    λS = (λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ)
    φS = (φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ)

    for (ξ, η, λ, φ) in zip(ξS, ηS, λS, φS)
        for j in 1:length(η), i in 1:length(ξ)
            x, y, z = @inbounds conformal_cubed_sphere_mapping(ξ[i], η[j])

            if !isnothing(rotation)
                x, y, z = rotation * [x, y, z]
            end

            @inbounds φ[i, j], λ[i, j] = cartesian_to_lat_lon(x, y, z)
        end
    end

    any(any.(isnan, λS)) &&
        @warn "OrthogonalSphericalShellGrid contains a grid point at a pole whose longitude is undefined (NaN)."

    ## Grid metrics

    # Horizontal distances

    #=
    Distances Δx and Δy are computed via the haversine formula provided by Distances.jl
    package. For example, Δx = Δσ * radius, where Δσ is the central angle that corresponds
    to the end points of distance Δx.

    For cells near the boundary of the conformal cubed sphere panel, one of the points
    defining, e.g., Δx might lie outside the grid! For example, the central angle
    Δxᶠᶜᵃ[1, j] that corresponds to the cell centered at Face 1, Center j is

        Δxᶠᶜᵃ[1, j] = haversine((λᶜᶜᵃ[1, j], φᶜᶜᵃ[1, j]), (λᶜᶜᵃ[0, j], φᶜᶜᵃ[0, j]), radius)

    Notice that, e.g., point (φᶜᶜᵃ[0, j], λᶜᶜᵃ[0, j]) is outside the boundaries of the grid.
    In those cases, we employ symmetry arguments and compute, e.g, Δxᶠᶜᵃ[1, j] via

        Δxᶠᶜᵃ[1, j] = 2 * haversine((λᶜᶜᵃ[1, j], φᶜᶜᵃ[1, j]), (λᶠᶜᵃ[1, j], φᶠᶜᵃ[1, j]), radius)
    =#


    Δxᶜᶜᵃ = zeros(FT, Nξ  , Nη  )
    Δxᶠᶜᵃ = zeros(FT, Nξ+1, Nη  )
    Δxᶜᶠᵃ = zeros(FT, Nξ  , Nη+1)
    Δxᶠᶠᵃ = zeros(FT, Nξ+1, Nη+1)

    @inbounds begin
        #Δxᶜᶜᵃ

        for i in 1:Nξ, j in 1:Nη
            Δxᶜᶜᵃ[i, j] = haversine((λᶠᶜᵃ[i+1, j], φᶠᶜᵃ[i+1, j]), (λᶠᶜᵃ[i, j], φᶠᶜᵃ[i, j]), radius)
        end


        # Δxᶠᶜᵃ

        for j in 1:Nη, i in 2:Nξ
            Δxᶠᶜᵃ[i, j] = haversine((λᶜᶜᵃ[i, j], φᶜᶜᵃ[i, j]), (λᶜᶜᵃ[i-1, j], φᶜᶜᵃ[i-1, j]), radius)
        end

        for j in 1:Nη
            i = 1
            Δxᶠᶜᵃ[i, j] = 2haversine((λᶜᶜᵃ[i, j], φᶜᶜᵃ[i, j]), (λᶠᶜᵃ[ i , j], φᶠᶜᵃ[ i , j]), radius)
        end

        for j in 1:Nη
            i = Nξ+1
            Δxᶠᶜᵃ[i, j] = 2haversine((λᶠᶜᵃ[i, j], φᶠᶜᵃ[i, j]), (λᶜᶜᵃ[i-1, j], φᶜᶜᵃ[i-1, j]), radius)
        end


        # Δxᶜᶠᵃ

        for j in 1:Nη+1, i in 1:Nξ
            Δxᶜᶠᵃ[i, j] = haversine((λᶠᶠᵃ[i+1, j], φᶠᶠᵃ[i+1, j]), (λᶠᶠᵃ[i, j], φᶠᶠᵃ[i, j]), radius)
        end


        # Δxᶠᶠᵃ

        for j in 1:Nη+1, i in 2:Nξ
            Δxᶠᶠᵃ[i, j] = haversine((λᶜᶠᵃ[i, j], φᶜᶠᵃ[i, j]), (λᶜᶠᵃ[i-1, j], φᶜᶠᵃ[i-1, j]), radius)
        end

        for j in 1:Nη+1
            i = 1
            Δxᶠᶠᵃ[i, j] = 2haversine((λᶜᶠᵃ[i, j], φᶜᶠᵃ[i, j]), (λᶠᶠᵃ[ i , j], φᶠᶠᵃ[ i , j]), radius)
        end

        for j in 1:Nη+1
            i = Nξ+1
            Δxᶠᶠᵃ[i, j] = 2haversine((λᶠᶠᵃ[i, j], φᶠᶠᵃ[i, j]), (λᶜᶠᵃ[i-1, j], φᶜᶠᵃ[i-1, j]), radius)
        end
    end

    Δyᶜᶜᵃ = zeros(FT, Nξ  , Nη  )
    Δyᶠᶜᵃ = zeros(FT, Nξ+1, Nη  )
    Δyᶜᶠᵃ = zeros(FT, Nξ  , Nη+1)
    Δyᶠᶠᵃ = zeros(FT, Nξ+1, Nη+1)

    @inbounds begin
        # Δyᶜᶜᵃ

        for j in 1:Nη, i in 1:Nξ
            Δyᶜᶜᵃ[i, j] = haversine((λᶜᶠᵃ[i, j+1], φᶜᶠᵃ[i, j+1]), (λᶜᶠᵃ[i, j], φᶜᶠᵃ[i, j]), radius)
        end


        # Δyᶜᶠᵃ

        for j in 2:Nη, i in 1:Nξ
            Δyᶜᶠᵃ[i, j] = haversine((λᶜᶜᵃ[i, j], φᶜᶜᵃ[i, j]), (λᶜᶜᵃ[i, j-1], φᶜᶜᵃ[i, j-1]), radius)
        end

        for i in 1:Nξ
            j = 1
            Δyᶜᶠᵃ[i, j] = 2haversine((λᶜᶜᵃ[i, j], φᶜᶜᵃ[i, j]), (λᶜᶠᵃ[i,  j ], φᶜᶠᵃ[i,  j ]), radius)
        end

        for i in 1:Nξ
            j = Nη+1
            Δyᶜᶠᵃ[i, j] = 2haversine((λᶜᶠᵃ[i, j], φᶜᶠᵃ[i, j]), (λᶜᶜᵃ[i, j-1], φᶜᶜᵃ[i, j-1]), radius)
        end


        # Δyᶠᶜᵃ

        for j in 1:Nη, i in 1:Nξ+1
            Δyᶠᶜᵃ[i, j] = haversine((λᶠᶠᵃ[i, j+1], φᶠᶠᵃ[i, j+1]), (λᶠᶠᵃ[i, j], φᶠᶠᵃ[i, j]), radius)
        end


        # Δyᶠᶠᵃ

        for j in 2:Nη, i in 1:Nξ+1
            Δyᶠᶠᵃ[i, j] = haversine((λᶠᶜᵃ[i, j], φᶠᶜᵃ[i, j]), (λᶠᶜᵃ[i, j-1], φᶠᶜᵃ[i, j-1]), radius)
        end

        for i in 1:Nξ+1
            j = 1
            Δyᶠᶠᵃ[i, j] = 2haversine((λᶠᶜᵃ[i, j], φᶠᶜᵃ[i, j]), (λᶠᶠᵃ[i,  j ], φᶠᶠᵃ[i,  j ]), radius)
        end
        
        for i in 1:Nξ+1
            j = Nη+1  
            Δyᶠᶠᵃ[i, j] = 2haversine((λᶠᶠᵃ[i, j], φᶠᶠᵃ[i, j]), (λᶠᶜᵃ[i, j-1], φᶠᶜᵃ[i, j-1]), radius)
        end
    end

    # Area metrics

    #=
    The areas Az correspond to spherical quadrilaterals. To compute Az first we
    find the vertices a, b, c, d of the corresponding quadrilateral and then use

        Az = spherical_area_quadrilateral(a, b, c, d) * radius^2

    For quadrilaterals near the boundary of the conformal cubed sphere panel, some of the
    vertices lie outside the grid! For example, the area Azᶠᶜᵃ[1, j] corresponds to a
    quadrilateral with vertices:

        a = (φᶜᶠᵃ[0,  j ], λᶜᶠᵃ[0,  j ])
        b = (φᶜᶠᵃ[1,  j ], λᶜᶠᵃ[1,  j ])
        c = (φᶜᶠᵃ[1, j+1], λᶜᶠᵃ[1, j+1])
        d = (φᶜᶠᵃ[0, j+1], λᶜᶠᵃ[0, j+1])

    Notice that vertices a and d are outside the boundaries of the grid. In those cases, we
    employ symmetry arguments and, e.g., compute Azᶠᶜᵃ[1, j] as

        2 * spherical_area_quadrilateral(ã, b, c, d̃) * radius^2

    where, ã = (φᶠᶠᵃ[1, j], λᶠᶠᵃ[1, j]) and d̃ = (φᶠᶠᵃ[1, j+1], λᶠᶠᵃ[1, j+1])
    =#

    Azᶜᶜᵃ = zeros(FT, Nξ  , Nη  )
    Azᶠᶜᵃ = zeros(FT, Nξ+1, Nη  )
    Azᶜᶠᵃ = zeros(FT, Nξ  , Nη+1)
    Azᶠᶠᵃ = zeros(FT, Nξ+1, Nη+1)

    @inbounds begin
        # Azᶜᶜᵃ

        for j in 1:Nη, i in 1:Nξ
            a = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)
            b = lat_lon_to_cartesian(φᶠᶠᵃ[i+1,  j ], λᶠᶠᵃ[i+1,  j ], 1)
            c = lat_lon_to_cartesian(φᶠᶠᵃ[i+1, j+1], λᶠᶠᵃ[i+1, j+1], 1)
            d = lat_lon_to_cartesian(φᶠᶠᵃ[ i , j+1], λᶠᶠᵃ[ i , j+1], 1)

            Azᶜᶜᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2
        end


        # Azᶠᶜᵃ

        for j in 1:Nη, i in 2:Nξ
            a = lat_lon_to_cartesian(φᶜᶠᵃ[i-1,  j ], λᶜᶠᵃ[i-1,  j ], 1)
            b = lat_lon_to_cartesian(φᶜᶠᵃ[ i ,  j ], λᶜᶠᵃ[ i ,  j ], 1)
            c = lat_lon_to_cartesian(φᶜᶠᵃ[ i , j+1], λᶜᶠᵃ[ i , j+1], 1)
            d = lat_lon_to_cartesian(φᶜᶠᵃ[i-1, j+1], λᶜᶠᵃ[i-1, j+1], 1)

            Azᶠᶜᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2
        end

        for j in 1:Nη
            i = 1
            a = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)
            b = lat_lon_to_cartesian(φᶜᶠᵃ[ i ,  j ], λᶜᶠᵃ[ i ,  j ], 1)
            c = lat_lon_to_cartesian(φᶜᶠᵃ[ i , j+1], λᶜᶠᵃ[ i , j+1], 1)
            d = lat_lon_to_cartesian(φᶠᶠᵃ[ i , j+1], λᶠᶠᵃ[ i , j+1], 1)

            Azᶠᶜᵃ[i, j] = 2 * spherical_area_quadrilateral(a, b, c, d) * radius^2
        end

        for j in 1:Nη
            i = Nξ+1
            a = lat_lon_to_cartesian(φᶜᶠᵃ[i-1,  j ], λᶜᶠᵃ[i-1,  j ], 1)
            b = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)
            c = lat_lon_to_cartesian(φᶠᶠᵃ[ i , j+1], λᶠᶠᵃ[ i , j+1], 1)
            d = lat_lon_to_cartesian(φᶜᶠᵃ[i-1, j+1], λᶜᶠᵃ[i-1, j+1], 1)

            Azᶠᶜᵃ[i, j] = 2 * spherical_area_quadrilateral(a, b, c, d) * radius^2
        end


        # Azᶜᶠᵃ

        for j in 2:Nη, i in 1:Nξ
            a = lat_lon_to_cartesian(φᶠᶜᵃ[ i , j-1], λᶠᶜᵃ[ i , j-1], 1)
            b = lat_lon_to_cartesian(φᶠᶜᵃ[i+1, j-1], λᶠᶜᵃ[i+1, j-1], 1)
            c = lat_lon_to_cartesian(φᶠᶜᵃ[i+1,  j ], λᶠᶜᵃ[i+1,  j ], 1)
            d = lat_lon_to_cartesian(φᶠᶜᵃ[ i ,  j ], λᶠᶜᵃ[ i ,  j ], 1)

            Azᶜᶠᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2
        end

        for i in 1:Nξ
            j = 1
            a = lat_lon_to_cartesian(φᶠᶠᵃ[ i , j ], λᶠᶠᵃ[ i , j ], 1)
            b = lat_lon_to_cartesian(φᶠᶠᵃ[i+1, j ], λᶠᶠᵃ[i+1, j ], 1)
            c = lat_lon_to_cartesian(φᶠᶜᵃ[i+1, j ], λᶠᶜᵃ[i+1, j ], 1)
            d = lat_lon_to_cartesian(φᶠᶜᵃ[ i , j ], λᶠᶜᵃ[ i , j ], 1)

            Azᶜᶠᵃ[i, j] = 2 * spherical_area_quadrilateral(a, b, c, d) * radius^2
        end

        for i in 1:Nξ
            j = Nη+1
            a = lat_lon_to_cartesian(φᶠᶜᵃ[ i , j-1], λᶠᶜᵃ[ i , j-1], 1)
            b = lat_lon_to_cartesian(φᶠᶜᵃ[i+1, j-1], λᶠᶜᵃ[i+1, j-1], 1)
            c = lat_lon_to_cartesian(φᶠᶠᵃ[i+1,  j ], λᶠᶠᵃ[i+1,  j ], 1)
            d = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)

            Azᶜᶠᵃ[i, j] = 2 * spherical_area_quadrilateral(a, b, c, d) * radius^2
        end


        # Azᶠᶠᵃ

        for j in 2:Nη, i in 2:Nξ
            a = lat_lon_to_cartesian(φᶜᶜᵃ[i-1, j-1], λᶜᶜᵃ[i-1, j-1], 1)
            b = lat_lon_to_cartesian(φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1], 1)
            c = lat_lon_to_cartesian(φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ], 1)
            d = lat_lon_to_cartesian(φᶜᶜᵃ[i-1,  j ], λᶜᶜᵃ[i-1,  j ], 1)

            Azᶠᶠᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2
        end

        for i in 2:Nξ
            j = 1
            a = lat_lon_to_cartesian(φᶜᶠᵃ[i-1, j ], λᶜᶠᵃ[i-1, j ], 1)
            b = lat_lon_to_cartesian(φᶜᶠᵃ[ i , j ], λᶜᶠᵃ[ i , j ], 1)
            c = lat_lon_to_cartesian(φᶜᶜᵃ[ i , j ], λᶜᶜᵃ[ i , j ], 1)
            d = lat_lon_to_cartesian(φᶜᶜᵃ[i-1, j ], λᶜᶜᵃ[i-1, j ], 1)

            Azᶠᶠᵃ[i, j] = 2 * spherical_area_quadrilateral(a, b, c, d) * radius^2
        end

        for i in 2:Nξ
            j = Nη+1
            a = lat_lon_to_cartesian(φᶜᶜᵃ[i-1, j-1], λᶜᶜᵃ[i-1, j-1], 1)
            b = lat_lon_to_cartesian(φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1], 1)
            c = lat_lon_to_cartesian(φᶜᶠᵃ[ i ,  j ], λᶜᶠᵃ[ i ,  j ], 1)
            d = lat_lon_to_cartesian(φᶜᶠᵃ[i-1,  j ], λᶜᶠᵃ[i-1,  j ], 1)

            Azᶠᶠᵃ[i, j] = 2 * spherical_area_quadrilateral(a, b, c, d) * radius^2
        end

        for j in 2:Nη
            i = 1
            a = lat_lon_to_cartesian(φᶠᶜᵃ[ i , j-1], λᶠᶜᵃ[ i , j-1], 1)
            b = lat_lon_to_cartesian(φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1], 1)
            c = lat_lon_to_cartesian(φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ], 1)
            d = lat_lon_to_cartesian(φᶠᶜᵃ[ i ,  j ], λᶠᶜᵃ[ i ,  j ], 1)

            Azᶠᶠᵃ[i, j] = 2 * spherical_area_quadrilateral(a, b, c, d) * radius^2
        end

        for j in 2:Nη
            i = Nξ+1
            a = lat_lon_to_cartesian(φᶜᶜᵃ[i-1, j-1], λᶜᶜᵃ[i-1, j-1], 1)
            b = lat_lon_to_cartesian(φᶠᶜᵃ[ i , j-1], λᶠᶜᵃ[ i , j-1], 1)
            c = lat_lon_to_cartesian(φᶠᶜᵃ[ i ,  j ], λᶠᶜᵃ[ i ,  j ], 1)
            d = lat_lon_to_cartesian(φᶜᶜᵃ[i-1,  j ], λᶜᶜᵃ[i-1,  j ], 1)

            Azᶠᶠᵃ[i, j] = 2 * spherical_area_quadrilateral(a, b, c, d) * radius^2
        end

        i = 1
        j = 1
        a = lat_lon_to_cartesian(φᶠᶠᵃ[i, j], λᶠᶠᵃ[i, j], 1)
        b = lat_lon_to_cartesian(φᶜᶠᵃ[i, j], λᶜᶠᵃ[i, j], 1)
        c = lat_lon_to_cartesian(φᶜᶜᵃ[i, j], λᶜᶜᵃ[i, j], 1)
        d = lat_lon_to_cartesian(φᶠᶜᵃ[i, j], λᶠᶜᵃ[i, j], 1)

        Azᶠᶠᵃ[i, j] = 4 * spherical_area_quadrilateral(a, b, c, d) * radius^2

        i = Nξ+1
        j = Nη+1
        a = lat_lon_to_cartesian(φᶜᶜᵃ[i-1, j-1], λᶜᶜᵃ[i-1, j-1], 1)
        b = lat_lon_to_cartesian(φᶠᶜᵃ[ i , j-1], λᶠᶜᵃ[ i , j-1], 1)
        c = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)
        d = lat_lon_to_cartesian(φᶜᶠᵃ[i-1,  j ], λᶜᶠᵃ[i-1,  j ], 1)

        Azᶠᶠᵃ[i, j] = 4 * spherical_area_quadrilateral(a, b, c, d) * radius^2

        i = Nξ+1
        j = 1
        a = lat_lon_to_cartesian(φᶜᶠᵃ[i-1, j ], λᶜᶠᵃ[i-1, j], 1)
        b = lat_lon_to_cartesian(φᶠᶠᵃ[ i , j ], λᶠᶠᵃ[ i , j], 1)
        c = lat_lon_to_cartesian(φᶠᶜᵃ[ i , j ], λᶠᶜᵃ[ i , j ], 1)
        d = lat_lon_to_cartesian(φᶜᶜᵃ[i-1, j ], λᶜᶜᵃ[i-1, j ], 1)

        Azᶠᶠᵃ[i, j] = 4 * spherical_area_quadrilateral(a, b, c, d) * radius^2

        i = 1
        j = Nη+1
        a = lat_lon_to_cartesian(φᶠᶜᵃ[ i , j-1], λᶠᶜᵃ[ i , j-1], 1)
        b = lat_lon_to_cartesian(φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1], 1)
        c = lat_lon_to_cartesian(φᶜᶠᵃ[ i ,  j ], λᶜᶠᵃ[ i ,  j ], 1)
        d = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)

        Azᶠᶠᵃ[i, j] = 4 * spherical_area_quadrilateral(a, b, c, d) * radius^2
    end

    # In all computations above we used (Bounded, Bounded, topology[3]) for ξ-η grid.
    # This was done to ensure that we had information for the faces at the boundary of
    # the grid.
    #
    # Now we take care the coordinate and metric arrays given the `topology` prescribed.

    warnings = false

    λᶜᶜᵃ = add_halos(λᶜᶜᵃ, (Center, Center, Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    λᶠᶜᵃ = add_halos(λᶠᶜᵃ, (Face,   Center, Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    λᶜᶠᵃ = add_halos(λᶜᶠᵃ, (Center, Face,   Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    λᶠᶠᵃ = add_halos(λᶠᶠᵃ, (Face,   Face,   Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)

    φᶜᶜᵃ = add_halos(φᶜᶜᵃ, (Center, Center, Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    φᶠᶜᵃ = add_halos(φᶠᶜᵃ, (Face,   Center, Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    φᶜᶠᵃ = add_halos(φᶜᶠᵃ, (Center, Face,   Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    φᶠᶠᵃ = add_halos(φᶠᶠᵃ, (Face,   Face,   Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)

    Δxᶜᶜᵃ = add_halos(Δxᶜᶜᵃ, (Center, Center, Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    Δxᶠᶜᵃ = add_halos(Δxᶠᶜᵃ, (Face,   Center, Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    Δxᶜᶠᵃ = add_halos(Δxᶜᶠᵃ, (Center, Face,   Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    Δxᶠᶠᵃ = add_halos(Δxᶠᶠᵃ, (Face,   Face,   Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)

    Δyᶜᶜᵃ = add_halos(Δyᶜᶜᵃ, (Center, Center, Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    Δyᶠᶜᵃ = add_halos(Δyᶠᶜᵃ, (Face,   Center, Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    Δyᶜᶠᵃ = add_halos(Δyᶜᶠᵃ, (Center, Face,   Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    Δyᶠᶠᵃ = add_halos(Δyᶠᶠᵃ, (Face,   Face,   Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)

    Azᶜᶜᵃ = add_halos(Azᶜᶜᵃ, (Center, Center, Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    Azᶠᶜᵃ = add_halos(Azᶠᶜᵃ, (Face,   Center, Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    Azᶜᶠᵃ = add_halos(Azᶜᶠᵃ, (Center, Face,   Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)
    Azᶠᶠᵃ = add_halos(Azᶠᶠᵃ, (Face,   Face,   Nothing), topology, (Nξ, Nη, Nz), (Hx, Hy, Hz); warnings)

    coordinate_arrays = (λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ, φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ)

    metric_arrays = (Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                     Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ,
                     Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                     Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ)

    conformal_mapping = (; ξ, η, rotation)

    grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(CPU(), Nξ, Nη, Nz, Hx, Hy, Hz, Lz,
                                                    coordinate_arrays...,
                                                    metric_arrays...,
                                                    radius,
                                                    conformal_mapping)

    fill_metric_halo_regions!(grid)

    # now convert to proper architecture

    coordinate_arrays = (grid.λᶜᶜᵃ, grid.λᶠᶜᵃ, grid.λᶜᶠᵃ, grid.λᶠᶠᵃ,
                         grid.φᶜᶜᵃ, grid.φᶠᶜᵃ, grid.φᶜᶠᵃ, grid.φᶠᶠᵃ,
                         grid.zᵃᵃᶜ, grid.zᵃᵃᶠ)

    metric_arrays = (grid.Δxᶜᶜᵃ, grid.Δxᶠᶜᵃ, grid.Δxᶜᶠᵃ, grid.Δxᶠᶠᵃ,
                     grid.Δyᶜᶜᵃ, grid.Δyᶜᶠᵃ, grid.Δyᶠᶜᵃ, grid.Δyᶠᶠᵃ,
                     grid.Δzᵃᵃᶜ, grid.Δzᵃᵃᶠ,
                     grid.Azᶜᶜᵃ, grid.Azᶠᶜᵃ, grid.Azᶜᶠᵃ, grid.Azᶠᶠᵃ)

    coordinate_arrays = map(a -> on_architecture(architecture, a), coordinate_arrays)

    metric_arrays = map(a -> on_architecture(architecture, a), metric_arrays)

    grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture, Nξ, Nη, Nz, Hx, Hy, Hz, Lz,
                                                    coordinate_arrays...,
                                                    metric_arrays...,
                                                    radius,
                                                    conformal_mapping)
    return grid
end

"""
    function fill_metric_halo_regions_x!(metric, ℓx, ℓy, tx, ty, Nx, Ny, Hx, Hy)

Fill the `x`-halo regions of the `metric` that lives on locations `ℓx`, `ℓy`, with halo size `Hx`, `Hy`,
and topology `tx`, `ty`.
"""
function fill_metric_halo_regions_x!(metric, ℓx, ℓy, tx::BoundedTopology, ty, Nx, Ny, Hx, Hy)
    # = N+1 for ::BoundedTopology or N otherwise
    Nx⁺ = length(ℓx, tx, Nx)
    Ny⁺ = length(ℓy, ty, Ny)

    @inbounds begin
        for j in 1:Ny⁺
            # fill west halos
            for i in 0:-1:-Hx+1
                metric[i, j] = metric[i+1, j]
            end
            # fill east halos
            for i in Nx⁺+1:Nx⁺+Hx
                metric[i, j] = metric[i-1, j]
            end
        end
    end

    return nothing
end

function fill_metric_halo_regions_x!(metric, ℓx, ℓy, tx::AbstractTopology, ty, Nx, Ny, Hx, Hy)
    # = N+1 for ::BoundedTopology or N otherwise
    Nx⁺ = length(ℓx, tx, Nx)
    Ny⁺ = length(ℓy, ty, Ny)

    @inbounds begin
        for j in 1:Ny⁺
            # fill west halos
            for i in 0:-1:-Hx+1
                metric[i, j] = metric[Nx+i, j]
            end
            # fill east halos
            for i in Nx⁺+1:Nx⁺+Hx
                metric[i, j] = metric[i-Nx, j]
            end
        end
    end

    return nothing
end

"""
    function fill_metric_halo_regions_y!(metric, ℓx, ℓy, tx, ty, Nx, Ny, Hx, Hy)

Fill the `y`-halo regions of the `metric` that lives on locations `ℓx`, `ℓy`, with halo size `Hx`, `Hy`,
and topology `tx`, `ty`.
"""
function fill_metric_halo_regions_y!(metric, ℓx, ℓy, tx, ty::BoundedTopology, Nx, Ny, Hx, Hy)
    # = N+1 for ::BoundedTopology or N otherwise
    Nx⁺ = length(ℓx, tx, Nx)
    Ny⁺ = length(ℓy, ty, Ny)

    @inbounds begin
        for i in 1:Nx⁺
            # fill south halos
            for j in 0:-1:-Hy+1
                metric[i, j] = metric[i, j+1]
            end
            # fill north halos
            for j in Ny⁺+1:Ny⁺+Hy
                metric[i, j] = metric[i, j-1]
            end
        end
    end

    return nothing
end

function fill_metric_halo_regions_y!(metric, ℓx, ℓy, tx, ty::AbstractTopology, Nx, Ny, Hx, Hy)
    # = N+1 for ::BoundedTopology or N otherwise
    Nx⁺ = length(ℓx, tx, Nx)
    Ny⁺ = length(ℓy, ty, Ny)

    @inbounds begin
        for i in 1:Nx⁺
            # fill south halos
            for j in 0:-1:-Hy+1
                metric[i, j] = metric[i, Ny+j]
            end
            # fill north halos
            for j in Ny⁺+1:Ny⁺+Hy
                metric[i, j] = metric[i, j-Ny]
            end
        end
    end

    return nothing
end

"""
    fill_metric_halo_corner_regions!(metric, ℓx, ℓy, tx, ty, Nx, Ny, Hx, Hy)

Fill the corner halo regions of the `metric`  that lives on locations `ℓx`, `ℓy`,
and with halo size `Hx`, `Hy`. We choose to fill with the average of the neighboring
metric in the halo regions. Thus this requires that the metric in the `x`- and `y`-halo
regions have already been filled.
"""
function fill_metric_halo_corner_regions!(metric, ℓx, ℓy, tx, ty, Nx, Ny, Hx, Hy)
    # = N+1 for ::BoundedTopology or N otherwise
    Nx⁺ = length(ℓx, tx, Nx)
    Ny⁺ = length(ℓy, ty, Ny)

    @inbounds begin
        for j in 0:-1:-Hy+1, i in 0:-1:-Hx+1
            metric[i, j] = (metric[i+1, j] + metric[i, j+1]) / 2
        end
        for j in Ny⁺+1:Ny⁺+Hy, i in 0:-1:-Hx+1
            metric[i, j] = (metric[i+1, j] + metric[i, j-1]) / 2
        end
        for j in 0:-1:-Hy+1, i in Nx⁺+1:Nx⁺+Hx
            metric[i, j] = (metric[i-1, j] + metric[i, j+1]) / 2
        end
        for j in Ny⁺+1:Ny⁺+Hy, i in Nx⁺+1:Nx⁺+Hx
            metric[i, j] = (metric[i-1, j] + metric[i, j-1]) / 2
        end
    end

    return nothing
end

function fill_metric_halo_regions!(grid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    TX, TY, _ = topology(grid)

    metric_arrays = (grid.Δxᶜᶜᵃ, grid.Δxᶠᶜᵃ, grid.Δxᶜᶠᵃ, grid.Δxᶠᶠᵃ,
                     grid.Δyᶜᶜᵃ, grid.Δyᶜᶠᵃ, grid.Δyᶠᶜᵃ, grid.Δyᶠᶠᵃ,
                     grid.Azᶜᶜᵃ, grid.Azᶠᶜᵃ, grid.Azᶜᶠᵃ, grid.Azᶠᶠᵃ)

    LXs = (Center, Face, Center, Face, Center, Center, Face, Face, Center, Face, Center, Face)
    LYs = (Center, Center, Face, Face, Center, Face, Center, Face, Center, Center, Face, Face)

    for (metric, LX, LY) in zip(metric_arrays, LXs, LYs)
        fill_metric_halo_regions_x!(metric, LX(), LY(), TX(), TY(), Nx, Ny, Hx, Hy)
        fill_metric_halo_regions_y!(metric, LX(), LY(), TX(), TY(), Nx, Ny, Hx, Hy)
        fill_metric_halo_corner_regions!(metric, LX(), LY(), TX(), TY(), Nx, Ny, Hx, Hy)
    end

    return nothing
end

function lat_lon_to_cartesian(lat, lon, radius)
    abs(lat) > 90 && error("lat must be within -90 ≤ lat ≤ 90")

    return [lat_lon_to_x(lat, lon, radius), lat_lon_to_y(lat, lon, radius), lat_lon_to_z(lat, lon, radius)]
end

lat_lon_to_x(lat, lon, radius) = radius * cosd(lon) * cosd(lat)
lat_lon_to_y(lat, lon, radius) = radius * sind(lon) * cosd(lat)
lat_lon_to_z(lat, lon, radius) = radius * sind(lat)

# architecture = CPU() default, assuming that a DataType positional arg
# is specifying the floating point type.
conformal_cubed_sphere_panel(FT::DataType; kwargs...) = conformal_cubed_sphere_panel(CPU(), FT; kwargs...)

function load_and_offset_cubed_sphere_data(file, FT, arch, field_name, loc, topo, N, H)

    data = on_architecture(arch, file[field_name])
    data = convert.(FT, data)

    return offset_data(data, loc[1:2], topo[1:2], N[1:2], H[1:2])
end

function conformal_cubed_sphere_panel(filepath::AbstractString, architecture = CPU(), FT = Float64;
                                      panel, Nz, z,
                                      topology = (FullyConnected, FullyConnected, Bounded),
                                        radius = R_Earth,
                                          halo = (4, 4, 4),
                                      rotation = nothing)

    TX, TY, TZ = topology
    Hx, Hy, Hz = halo

    ## The vertical coordinates can come out of the regular rectilinear grid!

    z_grid = RectilinearGrid(architecture, FT; size = Nz, z, topology=(Flat, Flat, topology[3]), halo=halo[3])

     zᵃᵃᶠ = z_grid.zᵃᵃᶠ
     zᵃᵃᶜ = z_grid.zᵃᵃᶜ
    Δzᵃᵃᶜ = z_grid.Δzᵃᵃᶜ
    Δzᵃᵃᶠ = z_grid.Δzᵃᵃᶠ
    Lz    = z_grid.Lz

    ## Read everything else from the file

    file = jldopen(filepath, "r")["panel$panel"]

    Nξ, Nη = size(file["λᶠᶠᵃ"])
    Hξ, Hη = halo[1], halo[2]
    Nξ -= 2Hξ
    Nη -= 2Hη

    N = (Nξ, Nη, Nz)
    H = halo

    loc_cc = (Center, Center)
    loc_fc = (Face,   Center)
    loc_cf = (Center, Face)
    loc_ff = (Face,   Face)

     λᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "λᶜᶜᵃ", loc_cc, topology, N, H)
     λᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "λᶠᶠᵃ", loc_ff, topology, N, H)

     φᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "φᶜᶜᵃ", loc_cc, topology, N, H)
     φᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "φᶠᶠᵃ", loc_ff, topology, N, H)

    Δxᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶜᶜᵃ", loc_cc, topology, N, H)
    Δxᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶠᶜᵃ", loc_fc, topology, N, H)
    Δxᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶜᶠᵃ", loc_cf, topology, N, H)
    Δxᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶠᶠᵃ", loc_ff, topology, N, H)

    Δyᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶜᶜᵃ", loc_cc, topology, N, H)
    Δyᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶠᶜᵃ", loc_fc, topology, N, H)
    Δyᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶜᶠᵃ", loc_cf, topology, N, H)
    Δyᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶠᶠᵃ", loc_ff, topology, N, H)

    Azᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶜᶜᵃ", loc_cc, topology, N, H)
    Azᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶠᶜᵃ", loc_fc, topology, N, H)
    Azᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶜᶠᵃ", loc_cf, topology, N, H)
    Azᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶠᶠᵃ", loc_ff, topology, N, H)

    ## Maybe we won't need these?
    Txᶠᶜ = total_length(loc_fc[1](), topology[1](), N[1], H[1])
    Txᶜᶠ = total_length(loc_cf[1](), topology[1](), N[1], H[1])
    Tyᶠᶜ = total_length(loc_fc[2](), topology[2](), N[2], H[2])
    Tyᶜᶠ = total_length(loc_cf[2](), topology[2](), N[2], H[2])

    λᶠᶜᵃ = offset_data(zeros(FT, architecture, Txᶠᶜ, Tyᶠᶜ), loc_fc, topology[1:2], N[1:2], H[1:2])
    λᶜᶠᵃ = offset_data(zeros(FT, architecture, Txᶜᶠ, Tyᶜᶠ), loc_cf, topology[1:2], N[1:2], H[1:2])
    φᶠᶜᵃ = offset_data(zeros(FT, architecture, Txᶠᶜ, Tyᶠᶜ), loc_fc, topology[1:2], N[1:2], H[1:2])
    φᶜᶠᵃ = offset_data(zeros(FT, architecture, Txᶜᶠ, Tyᶜᶠ), loc_cf, topology[1:2], N[1:2], H[1:2])

    conformal_mapping = (ξ = (-1, 1), η = (-1, 1))

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture, Nξ, Nη, Nz, Hx, Hy, Hz, Lz,
                                                     λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                     φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ,
                                                     zᵃᵃᶜ,  zᵃᵃᶠ,
                                                    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                    Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ,
                                                    Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                                                    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                                    radius,
                                                    conformal_mapping)
end

function on_architecture(arch::AbstractSerialArchitecture, grid::OrthogonalSphericalShellGrid)

    coordinates = (:λᶜᶜᵃ,
                   :λᶠᶜᵃ,
                   :λᶜᶠᵃ,
                   :λᶠᶠᵃ,
                   :φᶜᶜᵃ,
                   :φᶠᶜᵃ,
                   :φᶜᶠᵃ,
                   :φᶠᶠᵃ,
                   :zᵃᵃᶜ,
                   :zᵃᵃᶠ)

    grid_spacings = (:Δxᶜᶜᵃ,
                     :Δxᶠᶜᵃ,
                     :Δxᶜᶠᵃ,
                     :Δxᶠᶠᵃ,
                     :Δyᶜᶜᵃ,
                     :Δyᶜᶠᵃ,
                     :Δyᶠᶜᵃ,
                     :Δyᶠᶠᵃ,
                     :Δzᵃᵃᶜ,
                     :Δzᵃᵃᶜ)

    horizontal_areas = (:Azᶜᶜᵃ,
                        :Azᶠᶜᵃ,
                        :Azᶜᶠᵃ,
                        :Azᶠᶠᵃ)

    grid_spacing_data = Tuple(on_architecture(arch, getproperty(grid, name)) for name in grid_spacings)
    coordinate_data = Tuple(on_architecture(arch, getproperty(grid, name)) for name in coordinates)
    horizontal_area_data = Tuple(on_architecture(arch, getproperty(grid, name)) for name in horizontal_areas)

    TX, TY, TZ = topology(grid)

    new_grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(arch,
                                                        grid.Nx, grid.Ny, grid.Nz,
                                                        grid.Hx, grid.Hy, grid.Hz,
                                                        grid.Lz,
                                                        coordinate_data...,
                                                        grid_spacing_data...,
                                                        horizontal_area_data...,
                                                        grid.radius,
                                                        grid.conformal_mapping)

    return new_grid
end

function Adapt.adapt_structure(to, grid::OrthogonalSphericalShellGrid)
    TX, TY, TZ = topology(grid)

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(nothing,
                                                    grid.Nx, grid.Ny, grid.Nz,
                                                    grid.Hx, grid.Hy, grid.Hz,
                                                    grid.Lz,
                                                    adapt(to, grid.λᶜᶜᵃ),
                                                    adapt(to, grid.λᶠᶜᵃ),
                                                    adapt(to, grid.λᶜᶠᵃ),
                                                    adapt(to, grid.λᶠᶠᵃ),
                                                    adapt(to, grid.φᶜᶜᵃ),
                                                    adapt(to, grid.φᶠᶜᵃ),
                                                    adapt(to, grid.φᶜᶠᵃ),
                                                    adapt(to, grid.φᶠᶠᵃ),
                                                    adapt(to, grid.zᵃᵃᶜ),
                                                    adapt(to, grid.zᵃᵃᶠ),
                                                    adapt(to, grid.Δxᶜᶜᵃ),
                                                    adapt(to, grid.Δxᶠᶜᵃ),
                                                    adapt(to, grid.Δxᶜᶠᵃ),
                                                    adapt(to, grid.Δxᶠᶠᵃ),
                                                    adapt(to, grid.Δyᶜᶜᵃ),
                                                    adapt(to, grid.Δyᶜᶠᵃ),
                                                    adapt(to, grid.Δyᶠᶜᵃ),
                                                    adapt(to, grid.Δyᶠᶠᵃ),
                                                    adapt(to, grid.Δzᵃᵃᶜ),
                                                    adapt(to, grid.Δzᵃᵃᶠ),
                                                    adapt(to, grid.Azᶜᶜᵃ),
                                                    adapt(to, grid.Azᶠᶜᵃ),
                                                    adapt(to, grid.Azᶜᶠᵃ),
                                                    adapt(to, grid.Azᶠᶠᵃ),
                                                    grid.radius,
                                                    grid.conformal_mapping)
end

function Base.summary(grid::OrthogonalSphericalShellGrid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)
    metric_computation = isnothing(grid.Δxᶠᶜᵃ) ? "without precomputed metrics" : "with precomputed metrics"

    return string(size_summary(size(grid)),
                  " OrthogonalSphericalShellGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo",
                  " and ", metric_computation)
end

"""
    get_center_and_extents_of_shell(grid::OSSG)

Return the latitude-longitude coordinates of the center of the shell `(λ_center, φ_center)`
and also the longitudinal and latitudinal extend of the shell `(extent_λ, extent_φ)`.
"""
function get_center_and_extents_of_shell(grid::OSSG)
    Nx, Ny, _ = size(grid)

    # find the indices that correspond to the center of the shell
    # ÷ ensures that expressions below work for both odd and even
    i_center = Nx÷2 + 1
    j_center = Ny÷2 + 1

    if mod(Nx, 2) == 0
        ℓx = Face()
    elseif mod(Nx, 2) == 1
        ℓx = Center()
    end

    if mod(Ny, 2) == 0
        ℓy = Face()
    elseif mod(Ny, 2) == 1
        ℓy = Center()
    end

    # latitude and longitudes of the shell's center
    λ_center = CUDA.@allowscalar λnode(i_center, j_center, 1, grid, ℓx, ℓy, Center())
    φ_center = CUDA.@allowscalar φnode(i_center, j_center, 1, grid, ℓx, ℓy, Center())

    # the Δλ, Δφ are approximate if ξ, η are not symmetric about 0
    if mod(Ny, 2) == 0
        extent_λ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δxᶜᶠᵃ[1:Nx, :], dims=1))) / grid.radius
    elseif mod(Ny, 2) == 1
        extent_λ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δxᶜᶜᵃ[1:Nx, :], dims=1))) / grid.radius
    end

    if mod(Nx, 2) == 0
        extent_φ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δyᶠᶜᵃ[:, 1:Ny], dims=2))) / grid.radius
    elseif mod(Nx, 2) == 1
        extent_φ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δyᶠᶜᵃ[:, 1:Ny], dims=2))) / grid.radius
    end

    return (λ_center, φ_center), (extent_λ, extent_φ)
end

function Base.show(io::IO, grid::OrthogonalSphericalShellGrid, withsummary=true)
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)

    Nx_face, Ny_face = total_length(Face(), TX(), Nx, 0), total_length(Face(), TY(), Ny, 0)

    λ₁, λ₂ = minimum(grid.λᶠᶠᵃ[1:Nx_face, 1:Ny_face]), maximum(grid.λᶠᶠᵃ[1:Nx_face, 1:Ny_face])
    φ₁, φ₂ = minimum(grid.φᶠᶠᵃ[1:Nx_face, 1:Ny_face]), maximum(grid.φᶠᶠᵃ[1:Nx_face, 1:Ny_face])
    z₁, z₂ = domain(topology(grid, 3)(), Nz, grid.zᵃᵃᶠ)

    (λ_center, φ_center), (extent_λ, extent_φ) = get_center_and_extents_of_shell(grid)

    λ_center = round(λ_center, digits=4)
    φ_center = round(φ_center, digits=4)

    λ_center = ifelse(λ_center ≈ 0, 0.0, λ_center)
    φ_center = ifelse(φ_center ≈ 0, 0.0, φ_center)

    center_str = "centered at (λ, φ) = (" * prettysummary(λ_center) * ", " * prettysummary(φ_center) * ")"

    if φ_center ≈ 90
        center_str = "centered at: North Pole, (λ, φ) = (" * prettysummary(λ_center) * ", " * prettysummary(φ_center) * ")"
    end

    if φ_center ≈ -90
        center_str = "centered at: South Pole, (λ, φ) = (" * prettysummary(λ_center) * ", " * prettysummary(φ_center) * ")"
    end

    λ_summary = "$(TX)  extent $(prettysummary(extent_λ)) degrees"
    φ_summary = "$(TX)  extent $(prettysummary(extent_φ)) degrees"
    z_summary = domain_summary(TZ(), "z", z₁, z₂)

    longest = max(length(λ_summary), length(φ_summary), length(z_summary))

    padding_λ = length(λ_summary) < longest ? " "^(longest - length(λ_summary)) : ""
    padding_φ = length(φ_summary) < longest ? " "^(longest - length(φ_summary)) : ""

    λ_summary = "longitude: $(TX)  extent $(prettysummary(extent_λ)) degrees" * padding_λ *" " * coordinate_summary(rad2deg.(grid.Δxᶠᶠᵃ[1:Nx_face, 1:Ny_face] ./ grid.radius), "λ")
    φ_summary = "latitude:  $(TX)  extent $(prettysummary(extent_φ)) degrees" * padding_φ *" " * coordinate_summary(rad2deg.(grid.Δyᶠᶠᵃ[1:Nx_face, 1:Ny_face] ./ grid.radius), "φ")
    z_summary = "z:         " * dimension_summary(TZ(), "z", z₁, z₂, grid.Δzᵃᵃᶜ, longest - length(z_summary))

    if withsummary
        print(io, summary(grid), "\n")
    end

    return print(io, "├── ", center_str, "\n",
                     "├── ", λ_summary, "\n",
                     "├── ", φ_summary, "\n",
                     "└── ", z_summary)
end

@inline z_domain(grid::OrthogonalSphericalShellGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)
@inline cpu_face_constructor_z(grid::ZRegOrthogonalSphericalShellGrid) = z_domain(grid)

function with_halo(new_halo, old_grid::OrthogonalSphericalShellGrid; rotation=nothing)

    size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)
    topo = topology(old_grid)

    ξ = old_grid.conformal_mapping.ξ
    η = old_grid.conformal_mapping.η

    z = cpu_face_constructor_z(old_grid)

    new_grid = conformal_cubed_sphere_panel(architecture(old_grid), eltype(old_grid);
                                            size, z, ξ, η,
                                            topology = topo,
                                            radius = old_grid.radius,
                                            halo = new_halo,
                                            rotation)

    return new_grid
end

function nodes(grid::OSSG, ℓx, ℓy, ℓz; reshape=false, with_halos=false)
    λ = λnodes(grid, ℓx, ℓy, ℓz; with_halos)
    φ = φnodes(grid, ℓx, ℓy, ℓz; with_halos)
    z = znodes(grid, ℓx, ℓy, ℓz; with_halos)

    if reshape
        # λ and φ are 2D arrays
        N = (size(λ)..., size(z)...)
        λ = Base.reshape(λ, N[1], Ν[2], 1)
        φ = Base.reshape(φ, N[1], N[2], 1)
        z = Base.reshape(z, 1, 1, N[3])
    end

    return (λ, φ, z)
end

@inline λnodes(grid::OSSG, ℓx::Face,   ℓy::Face, ; with_halos=false) = with_halos ? grid.λᶠᶠᵃ :
    view(grid.λᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline λnodes(grid::OSSG, ℓx::Face,   ℓy::Center; with_halos=false) = with_halos ? grid.λᶠᶜᵃ :
    view(grid.λᶠᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline λnodes(grid::OSSG, ℓx::Center, ℓy::Face, ; with_halos=false) = with_halos ? grid.λᶜᶠᵃ :
    view(grid.λᶜᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline λnodes(grid::OSSG, ℓx::Center, ℓy::Center; with_halos=false) = with_halos ? grid.λᶜᶜᵃ :
    view(grid.λᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline φnodes(grid::OSSG, ℓx::Face,   ℓy::Face, ; with_halos=false) = with_halos ? grid.φᶠᶠᵃ :
    view(grid.φᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φnodes(grid::OSSG, ℓx::Face,   ℓy::Center; with_halos=false) = with_halos ? grid.φᶠᶜᵃ :
    view(grid.φᶠᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φnodes(grid::OSSG, ℓx::Center, ℓy::Face, ; with_halos=false) = with_halos ? grid.φᶜᶠᵃ :
    view(grid.φᶜᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φnodes(grid::OSSG, ℓx::Center, ℓy::Center; with_halos=false) = with_halos ? grid.φᶜᶜᵃ :
    view(grid.φᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline xnodes(grid::OSSG, ℓx, ℓy; with_halos=false) = grid.radius * deg2rad.(λnodes(grid, ℓx, ℓy; with_halos=with_halos)) .* hack_cosd.(φnodes(grid, ℓx, ℓy; with_halos=with_halos))
@inline ynodes(grid::OSSG, ℓx, ℓy; with_halos=false) = grid.radius * deg2rad.(φnodes(grid, ℓx, ℓy; with_halos=with_halos))

@inline znodes(grid::OSSG, ℓz::Face  ; with_halos=false) = with_halos ? grid.zᵃᵃᶠ :
    view(grid.zᵃᵃᶠ, interior_indices(ℓz, topology(grid, 3)(), grid.Nz))
@inline znodes(grid::OSSG, ℓz::Center; with_halos=false) = with_halos ? grid.zᵃᵃᶜ :
    view(grid.zᵃᵃᶜ, interior_indices(ℓz, topology(grid, 3)(), grid.Nz))

# convenience
@inline λnodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = λnodes(grid, ℓx, ℓy; with_halos)
@inline φnodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = φnodes(grid, ℓx, ℓy; with_halos)
@inline znodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = znodes(grid, ℓz    ; with_halos)
@inline xnodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = xnodes(grid, ℓx, ℓy; with_halos)
@inline ynodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = ynodes(grid, ℓx, ℓy; with_halos)

@inline λnode(i, j, grid::OSSG, ::Center, ::Center) = @inbounds grid.λᶜᶜᵃ[i, j]
@inline λnode(i, j, grid::OSSG, ::Face  , ::Center) = @inbounds grid.λᶠᶜᵃ[i, j]
@inline λnode(i, j, grid::OSSG, ::Center, ::Face  ) = @inbounds grid.λᶜᶠᵃ[i, j]
@inline λnode(i, j, grid::OSSG, ::Face  , ::Face  ) = @inbounds grid.λᶠᶠᵃ[i, j]

@inline φnode(i, j, grid::OSSG, ::Center, ::Center) = @inbounds grid.φᶜᶜᵃ[i, j]
@inline φnode(i, j, grid::OSSG, ::Face  , ::Center) = @inbounds grid.φᶠᶜᵃ[i, j]
@inline φnode(i, j, grid::OSSG, ::Center, ::Face  ) = @inbounds grid.φᶜᶠᵃ[i, j]
@inline φnode(i, j, grid::OSSG, ::Face  , ::Face  ) = @inbounds grid.φᶠᶠᵃ[i, j]

@inline xnode(i, j, grid::OSSG, ℓx, ℓy) = grid.radius * deg2rad(λnode(i, j, grid, ℓx, ℓy)) * hack_cosd((φnode(i, j, grid, ℓx, ℓy)))
@inline ynode(i, j, grid::OSSG, ℓx, ℓy) = grid.radius * deg2rad(φnode(i, j, grid, ℓx, ℓy))

@inline znode(k, grid::OSSG, ::Center) = @inbounds grid.zᵃᵃᶜ[k]
@inline znode(k, grid::OSSG, ::Face  ) = @inbounds grid.zᵃᵃᶠ[k]

# convenience
@inline λnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = λnode(i, j, grid, ℓx, ℓy)
@inline φnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = φnode(i, j, grid, ℓx, ℓy)
@inline znode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = znode(k, grid, ℓz)
@inline xnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = xnode(i, j, grid, ℓx, ℓy)
@inline ynode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = ynode(i, j, grid, ℓx, ℓy)

# Definitions for node
@inline ξnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = λnode(i, j, grid, ℓx, ℓy)
@inline ηnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = φnode(i, j, grid, ℓx, ℓy)
@inline rnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = znode(k, grid, ℓz)

ξname(::OSSG) = :λ
ηname(::OSSG) = :φ
rname(::OSSG) = :z

#####
##### Grid spacings in x, y, z (in meters)
#####

@inline xspacings(grid::OSSG, ℓx::Center, ℓy::Center; with_halos=false) =
    with_halos ? grid.Δxᶜᶜᵃ : view(grid.Δxᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline xspacings(grid::OSSG, ℓx::Face  , ℓy::Center; with_halos=false) =
    with_halos ? grid.Δxᶠᶜᵃ : view(grid.Δxᶠᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline xspacings(grid::OSSG, ℓx::Center, ℓy::Face  ; with_halos=false) =
    with_halos ? grid.Δxᶜᶠᵃ : view(grid.Δxᶜᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline xspacings(grid::OSSG, ℓx::Face  , ℓy::Face  ; with_halos=false) =
    with_halos ? grid.Δxᶠᶠᵃ : view(grid.Δxᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline yspacings(grid::OSSG, ℓx::Center, ℓy::Center; with_halos=false) =
    with_halos ? grid.Δyᶜᶜᵃ : view(grid.Δyᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline yspacings(grid::OSSG, ℓx::Face  , ℓy::Center; with_halos=false) =
    with_halos ? grid.Δyᶠᶜᵃ : view(grid.Δyᶠᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline yspacings(grid::OSSG, ℓx::Center, ℓy::Face  ; with_halos=false) =
    with_halos ? grid.Δyᶜᶠᵃ : view(grid.Δyᶜᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline yspacings(grid::OSSG, ℓx::Face  , ℓy::Face  ; with_halos=false) =
    with_halos ? grid.Δyᶠᶠᵃ : view(grid.Δyᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline zspacings(grid::OSSG,     ℓz::Center; with_halos=false) = with_halos ? grid.Δzᵃᵃᶜ :
    view(grid.Δzᵃᵃᶜ, interior_indices(ℓz, topology(grid, 3)(), grid.Nz))
@inline zspacings(grid::ZRegOSSG, ℓz::Center; with_halos=false) = grid.Δzᵃᵃᶜ
@inline zspacings(grid::OSSG,     ℓz::Face;   with_halos=false) = with_halos ? grid.Δzᵃᵃᶠ :
    view(grid.Δzᵃᵃᶠ, interior_indices(ℓz, topology(grid, 3)(), grid.Nz))
@inline zspacings(grid::ZRegOSSG, ℓz::Face;   with_halos=false) = grid.Δzᵃᵃᶠ

@inline xspacings(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = xspacings(grid, ℓx, ℓy; with_halos)
@inline yspacings(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = yspacings(grid, ℓx, ℓy; with_halos)
@inline zspacings(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = zspacings(grid, ℓz; with_halos)
