using CubedSphere
using JLD2
using OffsetArrays
using Adapt
using Adapt: adapt_structure

using LinearAlgebra

using Oceananigans

struct OrthogonalSphericalShellGrid{FT, TX, TY, TZ, A, R, Arch} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    Nx :: Int
    Ny :: Int
    Nz :: Int
    Hx :: Int
    Hy :: Int
    Hz :: Int
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
    Δz    :: FT
    Azᶜᶜᵃ :: A
    Azᶠᶜᵃ :: A
    Azᶜᶠᵃ :: A
    Azᶠᶠᵃ :: A
    radius :: FT

    function OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture::Arch,
                                                      Nx, Ny, Nz,
                                                      Hx, Hy, Hz,
                                                       λᶜᶜᵃ :: A,  λᶠᶜᵃ :: A,  λᶜᶠᵃ :: A,  λᶠᶠᵃ :: A,
                                                       φᶜᶜᵃ :: A,  φᶠᶜᵃ :: A,  φᶜᶠᵃ :: A,  φᶠᶠᵃ :: A, zᵃᵃᶜ :: R, zᵃᵃᶠ :: R,
                                                      Δxᶜᶜᵃ :: A, Δxᶠᶜᵃ :: A, Δxᶜᶠᵃ :: A, Δxᶠᶠᵃ :: A,
                                                      Δyᶜᶜᵃ :: A, Δyᶜᶠᵃ :: A, Δyᶠᶜᵃ :: A, Δyᶠᶠᵃ :: A, Δz :: FT,
                                                      Azᶜᶜᵃ :: A, Azᶠᶜᵃ :: A, Azᶜᶠᵃ :: A, Azᶠᶠᵃ :: A,
                                                      radius :: FT) where {TX, TY, TZ, FT, A, R, Arch}

        return new{FT, TX, TY, TZ, A, R, Arch}(architecture,
                                               Nx, Ny, Nz,
                                               Hx, Hy, Hz,
                                               λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                               φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
                                               Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                               Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δz,
                                               Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
    end
end

"""
    OrthogonalSphericalShellGrid(architecture::AbstractArchitecture = CPU(),
                                 FT::DataType = Float64;
                                 size,
                                 z,
                                 topology = (Bounded, Bounded, Bounded),
                                 ξ = (-1, 1),
                                 η = (-1, 1),
                                 radius = R_Earth,
                                 halo = (1, 1, 1),
                                 rotation = nothing)

Create a `OrthogonalSphericalShellGrid` that represents a section of a spherical cubed after it has been 
mapped from the face of a cube. The cube's coordinates are `ξ` and `η` (which, by default, take values
in the range ``[-1, 1]``.

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

- `radius`: The radius of the sphere the grid lives on. By default is equal to the radius of Earth.

- `halo`: A 3-tuple of integers specifying the size of the halo region of cells surrounding
          the physical interior. The default is 1 halo cells in every direction.

Examples
========

* A default grid with `Float64` type:

```jldoctest
julia> using Oceananigans

julia> grid = OrthogonalSphericalShellGrid(size=(36, 34, 25), z=(-1000, 0))
36×34×25 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 1×1×1 halo and with precomputed metrics
├── longitude: Bounded  λ ∈ [-176.397, 180.0] variably spaced with min(Δλ)=0.0, max(Δλ)=0.0
├── latitude:  Bounded  φ ∈ [0.0, 90.0]       variably spaced with min(Δφ)=0.0, max(Δφ)=0.0
└── z:         Bounded  z ∈ [-1000.0, 0.0]    regularly spaced with Δz=40.0
```
"""
function OrthogonalSphericalShellGrid(architecture::AbstractArchitecture = CPU(),
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

    ξη_grid = RectilinearGrid(architecture, FT; size=(Nξ, Nη, Nz), x=ξ, y=η, z, topology, halo)

    ξᶠᵃᵃ = xnodes(Face, ξη_grid)
    ξᶜᵃᵃ = xnodes(Center, ξη_grid)
    ηᵃᶠᵃ = ynodes(Face, ξη_grid)
    ηᵃᶜᵃ = ynodes(Center, ξη_grid)

    ## The vertical coordinates can come out of the regular rectilinear grid!

    Δz = ξη_grid.Δzᵃᵃᶜ
    zᵃᵃᶠ = ξη_grid.zᵃᵃᶠ
    zᵃᵃᶜ = ξη_grid.zᵃᵃᶜ

    ## Compute staggered grid Cartesian coordinates (X, Y, Z) on the unit sphere.

    Xᶜᶜᵃ = zeros(Nξ,   Nη  )
    Xᶠᶜᵃ = zeros(Nξ+1, Nη  )
    Xᶜᶠᵃ = zeros(Nξ,   Nη+1)
    Xᶠᶠᵃ = zeros(Nξ+1, Nη+1)

    Yᶜᶜᵃ = zeros(Nξ,   Nη  )
    Yᶠᶜᵃ = zeros(Nξ+1, Nη  )
    Yᶜᶠᵃ = zeros(Nξ,   Nη+1)
    Yᶠᶠᵃ = zeros(Nξ+1, Nη+1)

    Zᶜᶜᵃ = zeros(Nξ,   Nη  )
    Zᶠᶜᵃ = zeros(Nξ+1, Nη  )
    Zᶜᶠᵃ = zeros(Nξ,   Nη+1)
    Zᶠᶠᵃ = zeros(Nξ+1, Nη+1)

    ξS = (ξᶜᵃᵃ, ξᶠᵃᵃ, ξᶜᵃᵃ, ξᶠᵃᵃ)
    ηS = (ηᵃᶜᵃ, ηᵃᶜᵃ, ηᵃᶠᵃ, ηᵃᶠᵃ)
    XS = (Xᶜᶜᵃ, Xᶠᶜᵃ, Xᶜᶠᵃ, Xᶠᶠᵃ)
    YS = (Yᶜᶜᵃ, Yᶠᶜᵃ, Yᶜᶠᵃ, Yᶠᶠᵃ)
    ZS = (Zᶜᶜᵃ, Zᶠᶜᵃ, Zᶜᶠᵃ, Zᶠᶠᵃ)

    ## Note: ξ and η above are Arrays (not OffsetArrays) so we can loop over, e.g., 1:length(ξ)

    for (ξ, η, X, Y, Z) in zip(ξS, ηS, XS, YS, ZS)
        for j in 1:length(η), i in 1:length(ξ)
            @inbounds X[i, j], Y[i, j], Z[i, j] = conformal_cubed_sphere_mapping(ξ[i], η[j])
        end
    end

    ## Rotate the face if it's not the +z face (the one containing the North Pole).

    if !isnothing(rotation)
        for (ξ, η, X, Y, Z) in zip(ξS, ηS, XS, YS, ZS)
            for j in 1:length(η), i in 1:length(ξ)
                @inbounds X[i, j], Y[i, j], Z[i, j] = rotation * [X[i, j], Y[i, j], Z[i, j]]
            end
        end
    end

    ## Compute staggered grid latitude-longitude (φ, λ) coordinates.

    λᶜᶜᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    λᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    λᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    λᶠᶠᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    φᶜᶜᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    φᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    φᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    φᶠᶠᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    λS = (λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ)
    φS = (φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ)

    for (ξ, η, X, Y, Z, λ, φ) in zip(ξS, ηS, XS, YS, ZS, λS, φS)
        for j in 1:length(η), i in 1:length(ξ)
            @inbounds φ[i, j], λ[i, j] = cartesian_to_lat_lon(X[i, j], Y[i, j], Z[i, j])
        end
    end

    any(any.(isnan, λS)) &&
        @warn "Cubed sphere face contains a grid point at a pole whose longitude λ is undefined (NaN)."


    ## Grid metrics

    Δxᶜᶜᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    Δxᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    Δxᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    Δxᶠᶠᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    Δyᶜᶜᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    Δyᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    Δyᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    Δyᶠᶠᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    [Δxᶜᶜᵃ[i, j] = central_angled((φᶠᶜᵃ[i+1,  j ], λᶠᶜᵃ[i+1,  j ]), (φᶠᶜᵃ[ i ,  j ], λᶠᶜᵃ[ i ,  j ])) for i in 1:Nξ   , j in 1:Nη   ]

    [Δxᶠᶜᵃ[i, j] = central_angled((φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ]), (φᶜᶜᵃ[i-1,  j ], λᶜᶜᵃ[i-1,  j ])) for i in 2:Nξ   , j in 1:Nη   ]
    [Δxᶠᶜᵃ[i, j] = central_angled((φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ]), (φᶠᶜᵃ[ i ,  j ], λᶠᶜᵃ[ i ,  j ])) for i in (1,)   , j in 1:Nη   ]
    [Δxᶠᶜᵃ[i, j] = central_angled((φᶠᶜᵃ[ i ,  j ], λᶠᶜᵃ[ i ,  j ]), (φᶜᶜᵃ[i-1,  j ], λᶜᶜᵃ[i-1,  j ])) for i in (Nξ+1,), j in 1:Nη   ]

    [Δxᶜᶠᵃ[i, j] = central_angled((φᶠᶠᵃ[i+1,  j ], λᶠᶠᵃ[i+1,  j ]), (φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ])) for i in 1:Nξ   , j in 1:Nη+1 ]

    [Δxᶠᶠᵃ[i, j] = central_angled((φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ]), (φᶜᶜᵃ[i-1,  j ], λᶜᶜᵃ[i-1,  j ])) for i in 2:Nξ   , j in 1:Nη+1 ] 
    [Δxᶠᶠᵃ[i, j] = central_angled((φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ]), (φᶠᶜᵃ[ i ,  j ], λᶠᶜᵃ[ i ,  j ])) for i in (1,)   , j in 1:Nη+1 ]
    [Δxᶠᶠᵃ[i, j] = central_angled((φᶠᶜᵃ[ i ,  j ], λᶠᶜᵃ[ i ,  j ]), (φᶜᶜᵃ[i-1,  j ], λᶜᶜᵃ[i-1,  j ])) for i in (Nξ+1,), j in 1:Nη+1 ]

    [Δyᶜᶜᵃ[i, j] = central_angled((φᶜᶠᵃ[ i , j+1], λᶜᶠᵃ[ i , j+1]), (φᶜᶠᵃ[ i ,  j ], λᶜᶠᵃ[ i ,  j ])) for i in 1:Nξ   , j in 1:Nη   ]

    [Δyᶜᶠᵃ[i, j] = central_angled((φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ]), (φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1])) for i in 1:Nξ   , j in 1:Nη+1 ]
    [Δyᶜᶠᵃ[i, j] = central_angled((φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ]), (φᶜᶠᵃ[ i ,  j ], λᶜᶠᵃ[ i ,  j ])) for i in 1:Nξ   , j in (1,)   ]
    [Δyᶜᶠᵃ[i, j] = central_angled((φᶜᶠᵃ[ i ,  j ], λᶜᶠᵃ[ i ,  j ]), (φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1])) for i in 1:Nξ   , j in (Nη+1,)]

    [Δyᶠᶜᵃ[i, j] = central_angled((φᶠᶠᵃ[ i , j+1], λᶠᶠᵃ[ i , j+1]), (φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ])) for i in 1:Nξ+1 , j in 1:Nη   ]
    
    [Δyᶠᶠᵃ[i, j] = central_angled((φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ]), (φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1])) for i in 1:Nξ+1 , j in 1:Nη+1 ]
    [Δyᶠᶠᵃ[i, j] = central_angled((φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ]), (φᶜᶠᵃ[ i ,  j ], λᶜᶠᵃ[ i ,  j ])) for i in 1:Nξ+1 , j in (1,)   ]
    [Δyᶠᶠᵃ[i, j] = central_angled((φᶜᶠᵃ[ i ,  j ], λᶜᶠᵃ[ i ,  j ]), (φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1])) for i in 1:Nξ+1 , j in (Nη+1,)]

    # convert degrees to meters
    for Δs in [Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ]
        @. Δs = radius * deg2rad(Δs)
    end

    Azᶜᶜᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    Azᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    Azᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    Azᶠᶠᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    for j in 1:Nη, i in 1:Nξ
        a = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)
        b = lat_lon_to_cartesian(φᶠᶠᵃ[i+1,  j ], λᶠᶠᵃ[i+1,  j ], 1)
        c = lat_lon_to_cartesian(φᶠᶠᵃ[i+1, j+1], λᶠᶠᵃ[i+1, j+1], 1)
        d = lat_lon_to_cartesian(φᶠᶠᵃ[ i , j+1], λᶠᶠᵃ[ i , j+1], 1)

        Azᶜᶜᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2
    end

    for j in 1:Nη, i in 1:Nξ+1
        a = lat_lon_to_cartesian(φᶜᶠᵃ[i-1,  j ], λᶠᶠᵃ[i-1,  j ], 1)
        b = lat_lon_to_cartesian(φᶜᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)
        c = lat_lon_to_cartesian(φᶜᶠᵃ[ i , j+1], λᶠᶠᵃ[ i , j+1], 1)
        d = lat_lon_to_cartesian(φᶜᶠᵃ[i-1, j+1], λᶠᶠᵃ[i-1, j+1], 1)

        Azᶠᶜᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2
    end

    for j in 1:Nη+1, i in 1:Nξ
        a = lat_lon_to_cartesian(φᶠᶜᵃ[ i , j-1], λᶠᶜᵃ[ i , j-1], 1)
        b = lat_lon_to_cartesian(φᶠᶜᵃ[i+1, j-1], λᶠᶜᵃ[i+1, j-1], 1)
        c = lat_lon_to_cartesian(φᶠᶜᵃ[i+1,  j ], λᶠᶜᵃ[i+1,  j ], 1)
        d = lat_lon_to_cartesian(φᶠᶜᵃ[ i ,  j ], λᶠᶜᵃ[ i ,  j ], 1)

        Azᶜᶠᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2
    end

    for j in 1:Nη+1, i in 1:Nη+1
        a = lat_lon_to_cartesian(φᶜᶜᵃ[i-1, j-1], λᶜᶜᵃ[i-1, j-1], 1)
        b = lat_lon_to_cartesian(φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1], 1)
        c = lat_lon_to_cartesian(φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ], 1)
        d = lat_lon_to_cartesian(φᶜᶜᵃ[i-1,  j ], λᶜᶜᵃ[i-1,  j ], 1)

        Azᶠᶠᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2
    end

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture, Nξ, Nη, Nz, Hx, Hy, Hz,
                                                     λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                     φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ,
                                                     zᵃᵃᶜ,  zᵃᵃᶠ,
                                                    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                    Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ,
                                                    Δz, Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
end

"""
    spherical_area_triangle(a₁, a₂, a₃)

Return the area of a spherical triangle on the unit sphere whose points are given by 3-vectors,
`a₁`, `a₂`, and `a₃` whose origin is the the center of the sphere. The formula was first given
by Eriksson (1990). 

Denote with ``A``, ``B``, and ``C`` the inner angles of the spherical triangle and with ``a``, ``b``,
and ``c`` the side os the triangle. It's been known since Euler and Lagrange that
``\\tan(E/2) = P / (1 + \\cos a + \\cos b + \\cos c)``, where ``E = A + B + C - π`` is the triangle's
excess and ``P = (1 - \\cos²a - \\cos²b - \\cos²c + 2\\cos a \\cos b \\cos c)^{1/2}``. Erikkson (1990)\
showed that ``P`` is the same as the volume defined by the vectors `a₁`, `a₂`, and `a₃`, that is
``P = |\\boldsymbol{a}_1 \\cdot (\\boldsymbol{a}_2 \\times \\boldsymbol{a}_2)|``.

References
==========
Eriksson, F. (1990) On the Measure of Solid Angles, Mathematics Magazine, 63 (3), 184-187, doi:10.1080/0025570X.1990.11977515
"""
function spherical_area_triangle(a₁, a₂, a₃)
    (sum(a₁.^2) ≈ 1 && sum(a₂.^2) ≈ 1 && sum(a₃.^2) ≈ 1) || error("a₁, a₂, a₃ must be unit vectors")

    t = abs(dot(a₁, cross(a₂, a₃)))
    t /= 1 + dot(a₁, a₂) + dot(a₂, a₃) + dot(a₁, a₃)

    return 2atan(t)
end

"""
    spherical_area_quadrilateral(a₁, a₂, a₃, a₄)

Return the area of a spherical quadrilateral on the unit sphere whose points are given by 3-vectors,
`a₁`, `a₂`, `a₃`, and `a₄`. The area is computed as the sum of two spherical triangles. Care must be
taken so that points `a₁`, `a₂`, `a₃`, and `a₄` correspond to the edges of the quadrilateral in clockwise
or counterclockwise order.
"""
spherical_area_quadrilateral(a₁, a₂, a₃, a₄) = spherical_area_triangle(a₁, a₂, a₃) + spherical_area_triangle(a₁, a₄, a₃)

function lat_lon_to_cartesian(lat, lon, radius)
    abs(lat) > 90 && error("lat must be within -90 ≤ lat ≤ 90")

    return [lat_lon_to_x(lat, lon, radius), lat_lon_to_y(lat, lon, radius), lat_lon_to_z(lat, lon, radius)]
end

lat_lon_to_x(lat, lon, radius) = radius * cosd(lon) * cosd(lat)
lat_lon_to_y(lat, lon, radius) = radius * sind(lon) * cosd(lat)
lat_lon_to_z(lat, lon, radius) = radius * sind(lat)

"""
    hav(x)

Compute haversine of `x`, where `x` is in radians: `hav(x) = sin²(x/2)`.
"""
hav(x) = sin(x/2)^2

"""
    central_angle((φ₁, λ₁), (φ₂, λ₂))

Compute the central angle (in radians) between two points on the sphere with
`(latitude, longitude)` coordinates `(φ₁, λ₁)` and `(φ₂, λ₂)` (in radians).

References
==========
- [Wikipedia, Great-circle distance](https://en.wikipedia.org/wiki/Great-circle_distance)
"""
function central_angle((φ₁, λ₁), (φ₂, λ₂))
    Δφ, Δλ = φ₁ - φ₂, λ₁ - λ₂

    return 2asin(sqrt(hav(Δφ) + (1 - hav(Δφ) - hav(φ₁ + φ₂)) * hav(Δλ)))
end

"""
    central_angled((φ₁, λ₁), (φ₂, λ₂))

Compute the central angle (in degrees) between two points on the sphere with
`(latitude, longitude)` coordinates `(φ₁, λ₁)` and `(φ₂, λ₂)` (in degrees).

See also [`central_angle`](@ref).
"""
central_angled((φ₁, λ₁), (φ₂, λ₂)) = rad2deg(central_angle(deg2rad.((φ₁, λ₁)), deg2rad.((φ₂, λ₂))))

# architecture = CPU() default, assuming that a DataType positional arg
# is specifying the floating point type.
OrthogonalSphericalShellGrid(FT::DataType; kwargs...) = OrthogonalSphericalShellGrid(CPU(), FT; kwargs...)

function load_and_offset_cubed_sphere_data(file, FT, arch, field_name, loc, topo, N, H)

    ii = interior_indices(loc[1], topo[1], N[1])
    jj = interior_indices(loc[2], topo[2], N[2])

    interior_data = arch_array(arch, file[field_name][ii, jj])

    underlying_data = zeros(FT, arch,
                            total_length(loc[1], topo[1], N[1], H[1]),
                            total_length(loc[2], topo[2], N[2], H[2]))

    ip = interior_parent_indices(loc[1], topo[1], N[1], H[1])
    jp = interior_parent_indices(loc[2], topo[2], N[2], H[2])

    view(underlying_data, ip, jp) .= interior_data

    return offset_data(underlying_data, loc[1:2], topo[1:2], N[1:2], H[1:2])
end

function OrthogonalSphericalShellGrid(filepath::AbstractString, architecture = CPU(), FT = Float64;
                                      face, Nz, z,
                                      topology = (Bounded, Bounded, Bounded),
                                        radius = R_Earth,
                                          halo = (1, 1, 1),
                                      rotation = nothing)

    TX, TY, TZ = topology
    Hx, Hy, Hz = halo

    ## Use a regular rectilinear grid for the vertical grid
    ## The vertical coordinates can come out of the regular rectilinear grid!

    ξη_grid = RectilinearGrid(architecture, FT; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z, topology, halo)

    Δz = ξη_grid.Δzᵃᵃᶜ
    zᵃᵃᶠ = ξη_grid.zᵃᵃᶠ
    zᵃᵃᶜ = ξη_grid.zᵃᵃᶜ

    ## Read everything else from the file

    file = jldopen(filepath, "r")["face$face"]
    Nξ, Nη = size(file["λᶠᶠᵃ"]) .- 1

    N = (Nξ, Nη, Nz)
    H = halo

    loc_cc = (Center, Center)
    loc_cf = (Center, Face)
    loc_fc = (Face, Center)
    loc_ff = (Face, Face)

    topo_bbb = (Bounded, Bounded, Bounded)

    λᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "λᶜᶜᵃ", loc_cc, topo_bbb, N, H)
    λᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "λᶠᶠᵃ", loc_ff, topo_bbb, N, H)

    φᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "φᶜᶜᵃ", loc_cc, topo_bbb, N, H)
    φᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "φᶠᶠᵃ", loc_ff, topo_bbb, N, H)

    Δxᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶜᶜᵃ", loc_cc, topo_bbb, N, H)
    Δxᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶠᶜᵃ", loc_fc, topo_bbb, N, H)
    Δxᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶜᶠᵃ", loc_cf, topo_bbb, N, H)
    Δxᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶠᶠᵃ", loc_ff, topo_bbb, N, H)

    Δyᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶜᶜᵃ", loc_cc, topo_bbb, N, H)
    Δyᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶠᶜᵃ", loc_fc, topo_bbb, N, H)
    Δyᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶜᶠᵃ", loc_cf, topo_bbb, N, H)
    Δyᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶠᶠᵃ", loc_ff, topo_bbb, N, H)

    Azᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶜᶜᵃ", loc_cc, topo_bbb, N, H)
    Azᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶠᶜᵃ", loc_fc, topo_bbb, N, H)
    Azᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶜᶠᵃ", loc_cf, topo_bbb, N, H)
    Azᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶠᶠᵃ", loc_ff, topo_bbb, N, H)

    ## Maybe we won't need these?
    Txᶠᶜ = total_length(loc_fc[1], topology[1], N[1], H[1])
    Txᶜᶠ = total_length(loc_cf[1], topology[1], N[1], H[1])
    Tyᶠᶜ = total_length(loc_fc[2], topology[2], N[2], H[2])
    Tyᶜᶠ = total_length(loc_cf[2], topology[2], N[2], H[2])

    λᶠᶜᵃ = offset_data(zeros(FT, architecture, Txᶠᶜ, Tyᶠᶜ), loc_fc, topology[1:2], N[1:2], H[1:2])
    λᶜᶠᵃ = offset_data(zeros(FT, architecture, Txᶜᶠ, Tyᶜᶠ), loc_cf, topology[1:2], N[1:2], H[1:2])
    φᶠᶜᵃ = offset_data(zeros(FT, architecture, Txᶠᶜ, Tyᶠᶜ), loc_fc, topology[1:2], N[1:2], H[1:2])
    φᶜᶠᵃ = offset_data(zeros(FT, architecture, Txᶜᶠ, Tyᶜᶠ), loc_cf, topology[1:2], N[1:2], H[1:2])

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture, Nξ, Nη, Nz, Hx, Hy, Hz,
                                                     λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                     φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ,
                                                     zᵃᵃᶜ,  zᵃᵃᶠ,
                                                    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                    Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ,
                                                       Δz, Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
end

function on_architecture(arch::AbstractArchitecture, grid::OrthogonalSphericalShellGrid)

    horizontal_coordinates = (:λᶜᶜᵃ,
                              :λᶠᶜᵃ,
                              :λᶜᶠᵃ,
                              :λᶠᶠᵃ,
                              :φᶜᶜᵃ,
                              :φᶠᶜᵃ,
                              :φᶜᶠᵃ,
                              :φᶠᶠᵃ)

    horizontal_grid_spacings = (:Δxᶜᶜᵃ,
                                :Δxᶠᶜᵃ,
                                :Δxᶜᶠᵃ,
                                :Δxᶠᶠᵃ,
                                :Δyᶜᶜᵃ,
                                :Δyᶜᶠᵃ,
                                :Δyᶠᶜᵃ,
                                :Δyᶠᶠᵃ)

    horizontal_areas = (:Azᶜᶜᵃ,
                        :Azᶠᶜᵃ,
                        :Azᶜᶠᵃ,
                        :Azᶠᶠᵃ)

    horizontal_grid_spacing_data = Tuple(arch_array(arch, getproperty(grid, name)) for name in horizontal_grid_spacings)
    horizontal_coordinate_data = Tuple(arch_array(arch, getproperty(grid, name)) for name in horizontal_coordinates)
    horizontal_area_data = Tuple(arch_array(arch, getproperty(grid, name)) for name in horizontal_areas)

    zᵃᵃᶜ = arch_array(arch, grid.zᵃᵃᶜ)
    zᵃᵃᶠ = arch_array(arch, grid.zᵃᵃᶠ)

    TX, TY, TZ = topology(grid)

    new_grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(arch,
                                                        grid.Nx, grid.Ny, grid.Nz,
                                                        grid.Hx, grid.Hy, grid.Hz,
                                                        horizontal_coordinate_data..., zᵃᵃᶜ, zᵃᵃᶠ,
                                                        horizontal_grid_spacing_data..., grid.Δz,
                                                        horizontal_area_data..., grid.radius)

    return new_grid
end

function Adapt.adapt_structure(to, grid::OrthogonalSphericalShellGrid)
    TX, TY, TZ = topology(grid)

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(nothing,
                                                    grid.Nx, grid.Ny, grid.Nz,
                                                    grid.Hx, grid.Hy, grid.Hz,
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
                                                    grid.Δz,
                                                    adapt(to, grid.Azᶜᶜᵃ), 
                                                    adapt(to, grid.Azᶠᶜᵃ),
                                                    adapt(to, grid.Azᶜᶠᵃ),
                                                    adapt(to, grid.Azᶠᶠᵃ),
                                                    grid.radius)
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

function Base.show(io::IO, grid::OrthogonalSphericalShellGrid, withsummary=true)
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)

    λ₁, λ₂ = minimum(grid.λᶠᶠᵃ[1:Nx, 1:Ny]), maximum(grid.λᶠᶠᵃ[1:Nx, 1:Ny])
    φ₁, φ₂ = minimum(grid.φᶠᶠᵃ[1:Nx, 1:Ny]), maximum(grid.φᶠᶠᵃ[1:Nx, 1:Ny])
    z₁, z₂ = domain(topology(grid, 3), grid.Nz, grid.zᵃᵃᶠ)

    x_summary = domain_summary(TX(), "λ", λ₁, λ₂)
    y_summary = domain_summary(TY(), "φ", φ₁, φ₂)
    z_summary = domain_summary(TZ(), "z", z₁, z₂)

    longest = max(length(x_summary), length(y_summary), length(z_summary))

    x_summary = "longitude: " * dimension_summary(TX(), "λ", λ₁, λ₂, grid.Δxᶠᶠᵃ[1:Nx, 1:Ny], longest - length(x_summary))
    y_summary = "latitude:  " * dimension_summary(TY(), "φ", φ₁, φ₂, grid.Δyᶠᶠᵃ[1:Nx, 1:Ny], longest - length(y_summary))
    z_summary = "z:         " * dimension_summary(TZ(), "z", z₁, z₂, grid.Δz, longest - length(z_summary))

    if withsummary
        print(io, summary(grid), "\n")
    end

    return print(io, "├── ", x_summary, "\n",
                     "├── ", y_summary, "\n",
                     "└── ", z_summary)
end

@inline xnode(::Face,   ::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶠᶠᵃ[i, j]
@inline xnode(::Face,   ::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶠᶜᵃ[i, j]
@inline xnode(::Center, ::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶜᶠᵃ[i, j]
@inline xnode(::Center, ::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶜᶜᵃ[i, j]

@inline ynode(::Face,   ::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶠᶠᵃ[i, j]
@inline ynode(::Face,   ::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶠᶜᵃ[i, j]
@inline ynode(::Center, ::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶜᶠᵃ[i, j]
@inline ynode(::Center, ::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶜᶜᵃ[i, j]

@inline znode(LX, LY, ::Face,   i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.zᵃᵃᶠ[k]
@inline znode(LX, LY, ::Center, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.zᵃᵃᶜ[k]
