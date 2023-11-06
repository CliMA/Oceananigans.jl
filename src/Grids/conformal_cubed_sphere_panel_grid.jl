""" the mapping Type for a ConformalCubedSpherePanelGrid. It holds the native cartesian coordinates and the rotation of the panel """
struct ConformalCubedSphereMapping{E, X, R} <: AbstractOrthogonalMapping
    η :: E
    ξ :: X
    rotation :: R
end

Adapt.adapt_structure(to, m::ConformalCubedSphereMapping) = 
    ConformalCubedSphereMapping(Adapt.adapt(to, m.η),
                                Adapt.adapt(to, m.ξ),
                                Adapt.adapt(to, m.rotation))

on_architecture(arch, m::ConformalCubedSphereMapping) = 
    ConformalCubedSphereMapping(on_architecture(arch, m.η),
                                on_architecture(arch, m.ξ),
                                on_architecture(arch, m.rotation))
                            
const ConformalCubedSpherePanelGrid{FT, TX, TY, TZ, FX, FY, FZ, X, Y, Z, Arch} = 
            OrthogonalSphericalShellGrid{FT, <:ConformalCubedSphereMapping, TX, TY, TZ, FX, FY, FZ, X, Y, Z, Arch} where {FT, TX, TY, TZ, FX, FY, FZ, X, Y, Z, Arch}

"""
    ConformalCubedSpherePanelGrid(architecture::AbstractArchitecture = CPU(),
                                  FT::DataType = Float64;
                                  size,
                                  z,
                                  topology = (Bounded, Bounded, Bounded),
                                  ξ = (-1, 1),
                                  η = (-1, 1),
                                  radius = R_Earth,
                                  halo = (1, 1, 1),
                                  rotation = nothing)

Create a `ConformalCubedSpherePanelGrid` that represents a section of a sphere after it has been 
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

julia> grid = ConformalCubedSpherePanelGrid(size=(36, 34, 25), z=(-1000, 0))
36×34×25 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 1×1×1 halo and with precomputed metrics
├── centered at: North Pole, (λ, φ) = (0.0, 90.0)
├── longitude: Bounded  extent 90.0 degrees variably spaced with min(Δλ)=0.616164, max(Δλ)=2.58892
├── latitude:  Bounded  extent 90.0 degrees variably spaced with min(Δφ)=0.664958, max(Δφ)=2.74119
└── z:         Bounded  z ∈ [-1000.0, 0.0]  regularly spaced with Δz=40.0
```

* The conformal cubed sphere panel that includes the South Pole with `Float32` type:

```jldoctest
julia> using Oceananigans, Oceananigans.Grids, Rotations

julia> grid = ConformalCubedSpherePanelGrid(Float32, size=(36, 34, 25), z=(-1000, 0), rotation=RotY(π))
36×34×25 OrthogonalSphericalShellGrid{Float32, Bounded, Bounded, Bounded} on CPU with 1×1×1 halo and with precomputed metrics
├── centered at: South Pole, (λ, φ) = (0.0, -90.0)
├── longitude: Bounded  extent 90.0 degrees variably spaced with min(Δλ)=0.616167, max(Δλ)=2.58891
├── latitude:  Bounded  extent 90.0 degrees variably spaced with min(Δφ)=0.664956, max(Δφ)=2.7412
└── z:         Bounded  z ∈ [-1000.0, 0.0]  regularly spaced with Δz=40.0
```
"""
function ConformalCubedSpherePanelGrid(architecture::AbstractArchitecture = CPU(),
                                       FT::DataType = Float64;
                                       size,
                                       z,
                                       topology = (Bounded, Bounded, Bounded),
                                       ξ = (-1, 1),
                                       η = (-1, 1),
                                       radius = R_Earth,
                                       halo = (1, 1, 1),
                                       rotation = nothing)

    radius = FT(radius)

    TX, TY, TZ = topology
    Nξ, Nη, Nz = size
    Hx, Hy, Hz = halo

    ## Use a regular rectilinear grid for the face of the cube

    ξη_grid_topology = (Bounded, Bounded, topology[3])

    ξη_grid = RectilinearGrid(architecture, FT;
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
        @warn "ConformalCubedSpherePanelGrid contains a grid point at a pole whose longitude is undefined (NaN)."

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

    # convert to
    coordinate_arrays = (λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ, φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, zᵃᵃᶜ,  zᵃᵃᶠ)
    coordinate_arrays = map(a -> arch_array(architecture, a), coordinate_arrays)

    metric_arrays = (Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                     Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                     Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ,
                     Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ)
    metric_arrays = map(a -> arch_array(architecture, a), metric_arrays)

    # the Δλ, Δφ are approximate if ξ, η are not symmetric about 0
    Lx = if mod(Nη, 2) == 0
        maximum(rad2deg.(sum(Δxᶜᶠᵃ[1:Nξ, :], dims=1))) / radius
    else
        maximum(rad2deg.(sum(Δxᶜᶜᵃ[1:Nξ, :], dims=1))) / radius
    end

    Ly = if mod(Nξ, 2) == 0
        maximum(rad2deg.(sum(Δyᶠᶜᵃ[:, 1:Nη], dims=2))) / radius
    elseif mod(Nξ, 2) == 1
        maximum(rad2deg.(sum(Δyᶠᶜᵃ[:, 1:Nη], dims=2))) / radius
    end
    
    grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture, ConformalCubedSphereMapping(ξ, η, rotation), 
                                                    Nξ, Nη, Nz, 
                                                    Hx, Hy, Hz, 
                                                    Lx, Ly, Lz,
                                                    coordinate_arrays...,
                                                    metric_arrays...,
                                                    radius)

    fill_metric_halo_regions!(grid)

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
ConformalCubedSpherePanelGrid(FT::DataType; kwargs...) = ConformalCubedSpherePanelGrid(CPU(), FT; kwargs...)

function load_and_offset_cubed_sphere_data(file, FT, arch, field_name, loc, topo, N, H)

    ii = interior_indices(loc[1](), topo[1](), N[1])
    jj = interior_indices(loc[2](), topo[2](), N[2])

    interior_data = arch_array(arch, file[field_name][ii, jj])

    underlying_data = zeros(FT, arch,
                            total_length(loc[1](), topo[1](), N[1], H[1]),
                            total_length(loc[2](), topo[2](), N[2], H[2]))

    ip = interior_parent_indices(loc[1](), topo[1](), N[1], H[1])
    jp = interior_parent_indices(loc[2](), topo[2](), N[2], H[2])

    view(underlying_data, ip, jp) .= interior_data

    return offset_data(underlying_data, loc[1:2], topo[1:2], N[1:2], H[1:2])
end

function ConformalCubedSpherePanelGrid(filepath::AbstractString, architecture = CPU(), FT = Float64;
                                       panel, Nz, z,
                                       topology = (Bounded, Bounded, Bounded),
                                         radius = R_Earth,
                                           halo = (1, 1, 1),
                                       rotation = nothing)

    TX, TY, TZ = topology
    Hx, Hy, Hz = halo

    ## Use a regular rectilinear grid for the vertical grid
    ## The vertical coordinates can come out of the regular rectilinear grid!

    ξ, η = (-1, 1), (-1, 1)
    ξη_grid = RectilinearGrid(architecture, FT; size = (1, 1, Nz), x = ξ, y = η, z, topology, halo)

     zᵃᵃᶠ = ξη_grid.zᵃᵃᶠ
     zᵃᵃᶜ = ξη_grid.zᵃᵃᶜ
    Δzᵃᵃᶜ = ξη_grid.Δzᵃᵃᶜ
    Δzᵃᵃᶠ = ξη_grid.Δzᵃᵃᶠ
    Lz    = ξη_grid.Lz

    ## Read everything else from the file

    file = jldopen(filepath, "r")["face$panel"]

    Nξ, Nη = size(file["λᶠᶠᵃ"]) .- 1

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

    # the Δλ, Δφ are approximate if ξ, η are not symmetric about 0
    Lx = if mod(Nη, 2) == 0
        maximum(rad2deg.(sum(Δxᶜᶠᵃ[1:Nξ, :], dims=1))) / radius
    else
        maximum(rad2deg.(sum(Δxᶜᶜᵃ[1:Nξ, :], dims=1))) / radius
    end

    Ly = if mod(Nξ, 2) == 0
        maximum(rad2deg.(sum(Δyᶠᶜᵃ[:, 1:Nη], dims=2))) / radius
    elseif mod(Nξ, 2) == 1
        maximum(rad2deg.(sum(Δyᶠᶜᵃ[:, 1:Nη], dims=2))) / radius
    end
    
    return OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture, ConformalCubedSphereMapping(ξ, η, rotation), 
                                                     Nξ, Nη, Nz, 
                                                     Hx, Hy, Hz,
                                                     Lx, Ly, Lz,
                                                     λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                     φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ,
                                                     zᵃᵃᶜ,  zᵃᵃᶠ,  Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                                                    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                    Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ,
                                                    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                                    radius)
end

function with_halo(new_halo, old_grid::ConformalCubedSpherePanelGrid)

    size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)
    topo = topology(old_grid)

    z = cpu_face_constructor_z(old_grid)

    new_grid = ConformalCubedSpherePanelGrid(architecture(old_grid), 
                                             eltype(old_grid);
                                             size, z, 
                                             topology = topo,
                                             radius = old_grid.radius,
                                             rotation = old_grid.mapping.rotation,
                                             ξ = old_grid.mapping.ξ,
                                             η = old_grid.mapping.η,
                                             halo = new_halo)

    return new_grid
end