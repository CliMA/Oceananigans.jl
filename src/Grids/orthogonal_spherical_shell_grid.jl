using CubedSphere
using JLD2
using OffsetArrays
using Adapt
using Adapt: adapt_structure

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

const OSSG = OrthogonalSphericalShellGrid
const ZRegOSSG = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}

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

Create a `OrthogonalSphericalShellGrid` that represents a section of a sphere after it has been 
mapped from the face of a cube. The cube's coordinates are `ξ` and `η` (which, by default, take values
in the range ``[-1, 1]``.

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

- `radius`: The radius of the sphere the grid lives on. By default is equal to the radius of Earth.

- `halo`: A 3-tuple of integers specifying the size of the halo region of cells surrounding
          the physical interior. The default is 1 halo cells in every direction.

- `rotation`: Rotation of the spherical shell grid about some axis that passes through the center
              of the sphere. If `nothing` is provided (default), then the spherical shell includes
              the North Pole of the sphere in its center.

Examples
========

* A default grid with `Float64` type:

```@example
julia> using Oceananigans

julia> grid = OrthogonalSphericalShellGrid(size=(36, 34, 25), z=(-1000, 0))
36×34×25 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 1×1×1 halo and with precomputed metrics
├── longitude: Bounded  λ ∈ [-176.397, 180.0] variably spaced with min(Δλ)=48351.7, max(Δλ)=2.87833e5
├── latitude:  Bounded  φ ∈ [35.2644, 90.0]   variably spaced with min(Δφ)=50632.2, max(Δφ)=3.04768e5
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

    @warn "OrthogonalSphericalShellGrid is still under development. Use with caution!"

    radius = FT(radius)

    TX, TY, TZ = topology
    Nξ, Nη, Nz = size
    Hx, Hy, Hz = halo

    ## Use a regular rectilinear grid for the face of the cube

    ξη_grid = RectilinearGrid(architecture, FT; size=(Nξ, Nη, Nz), x=ξ, y=η, z, topology, halo)

    ξᶠᵃᵃ = xnodes(ξη_grid, Face())
    ξᶜᵃᵃ = xnodes(ξη_grid, Center())
    ηᵃᶠᵃ = ynodes(ξη_grid, Face())
    ηᵃᶜᵃ = ynodes(ξη_grid, Center())

    ## The vertical coordinates can come out of the regular rectilinear grid!

    Δz = ξη_grid.Δzᵃᵃᶜ
    zᵃᵃᶠ = ξη_grid.zᵃᵃᶠ
    zᵃᵃᶜ = ξη_grid.zᵃᵃᶜ

    ## Compute staggered grid Cartesian coordinates (X, Y, Z) on the unit sphere.

    Xᶜᶜᵃ = zeros(Nξ  , Nη  )
    Xᶠᶜᵃ = zeros(Nξ+1, Nη  )
    Xᶜᶠᵃ = zeros(Nξ  , Nη+1)
    Xᶠᶠᵃ = zeros(Nξ+1, Nη+1)

    Yᶜᶜᵃ = zeros(Nξ  , Nη  )
    Yᶠᶜᵃ = zeros(Nξ+1, Nη  )
    Yᶜᶠᵃ = zeros(Nξ  , Nη+1)
    Yᶠᶠᵃ = zeros(Nξ+1, Nη+1)

    Zᶜᶜᵃ = zeros(Nξ  , Nη  )
    Zᶠᶜᵃ = zeros(Nξ+1, Nη  )
    Zᶜᶠᵃ = zeros(Nξ  , Nη+1)
    Zᶠᶠᵃ = zeros(Nξ+1, Nη+1)

    ξS = (ξᶜᵃᵃ, ξᶠᵃᵃ, ξᶜᵃᵃ, ξᶠᵃᵃ)
    ηS = (ηᵃᶜᵃ, ηᵃᶜᵃ, ηᵃᶠᵃ, ηᵃᶠᵃ)
    XS = (Xᶜᶜᵃ, Xᶠᶜᵃ, Xᶜᶠᵃ, Xᶠᶠᵃ)
    YS = (Yᶜᶜᵃ, Yᶠᶜᵃ, Yᶜᶠᵃ, Yᶠᶠᵃ)
    ZS = (Zᶜᶜᵃ, Zᶠᶜᵃ, Zᶜᶠᵃ, Zᶠᶠᵃ)

    ## Note: ξ and η above are Arrays (not OffsetArrays) so we can loop over, e.g., 1:length(ξ)

    for (ξ, η, X, Y, Z) in zip(ξS, ηS, XS, YS, ZS)
        for j in 1:length(η), i in 1:length(ξ)
            # maps (ξ, η) from cube's face to (X, Y, Y) on the unit sphere
            @inbounds X[i, j], Y[i, j], Z[i, j] = conformal_cubed_sphere_mapping(ξ[i], η[j])
        end
    end

    ## Rotate the mapped (X, Y, Z) if needed

    if !isnothing(rotation)
        for (ξ, η, X, Y, Z) in zip(ξS, ηS, XS, YS, ZS)
            for j in 1:length(η), i in 1:length(ξ)
                @inbounds X[i, j], Y[i, j], Z[i, j] = rotation * [X[i, j], Y[i, j], Z[i, j]]
            end
        end
    end

    ## Compute staggered grid latitude-longitude (φ, λ) coordinates.

    λᶜᶜᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    λᶠᶜᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    λᶜᶠᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)
    λᶠᶠᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)

    φᶜᶜᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    φᶠᶜᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    φᶜᶠᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)
    φᶠᶠᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)

    λS = (λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ)
    φS = (φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ)

    for (ξ, η, X, Y, Z, λ, φ) in zip(ξS, ηS, XS, YS, ZS, λS, φS)
        for j in 1:length(η), i in 1:length(ξ)
            # convert cartesian (X, Y, Z) to lat-long (φ, λ)
            @inbounds φ[i, j], λ[i, j] = cartesian_to_lat_lon(X[i, j], Y[i, j], Z[i, j])
        end
    end

    any(any.(isnan, λS)) &&
        @warn "Cubed sphere face contains a grid point at a pole whose longitude λ is undefined (NaN)."

    ## Grid metrics

    # Horizontal distances

    #=
    Distances Δx and Δy are computed via, e.g., Δx = Δσ * radius, where Δσ is the
    central angle that corresponds to the end points of distance Δx.

    For cells near the boundary of the OrthogonalSphericalShellGrid one of the points
    defining, e.g., Δx might lie outside the grid! For example, the central angle
    Δσxᶠᶜᵃ[1, j] that corresponds to the cell centered at Face 1, Center j is

        Δσxᶠᶜᵃ[1, j] = central_angle((φᶜᶜᵃ[1, j], λᶜᶜᵃ[1, j]), (φᶜᶜᵃ[0, j], λᶜᶜᵃ[0, j]))

    Notice that point (φᶜᶜᵃ[0, j], λᶜᶜᵃ[0, j]) is outside the boundaries of the grid.
    In those cases, we employ symmetry arguments and compute, e.g, Δσxᶠᶜᵃ[1, j] via

        Δσxᶠᶜᵃ[1, j] = 2 * central_angle_degrees((φᶜᶜᵃ[1, j], λᶜᶜᵃ[1, j]), (φᶠᶜᵃ[1, j], λᶠᶜᵃ[1, j]))
    =#


    # central angles

    Δσxᶜᶜᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    Δσxᶠᶜᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    Δσxᶜᶠᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)
    Δσxᶠᶠᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)


    #Δσxᶜᶜᵃ

    for i in 1:Nξ, j in 1:Nη
        Δσxᶜᶜᵃ[i, j] =  central_angle_degrees((φᶠᶜᵃ[i+1, j], λᶠᶜᵃ[i+1, j]), (φᶠᶜᵃ[i, j], λᶠᶜᵃ[i, j]))
    end


    # Δσxᶠᶜᵃ

    for j in 1:Nη, i in 2:Nξ+1
        Δσxᶠᶜᵃ[i, j] =  central_angle_degrees((φᶜᶜᵃ[i, j], λᶜᶜᵃ[i, j]), (φᶜᶜᵃ[i-1, j], λᶜᶜᵃ[i-1, j]))
    end

    for j in 1:Nη
        i = 1
        Δσxᶠᶜᵃ[i, j] = 2central_angle_degrees((φᶜᶜᵃ[i, j], λᶜᶜᵃ[i, j]), (φᶠᶜᵃ[ i , j], λᶠᶜᵃ[ i , j]))
    end

    for j in 1:Nη
        i = Nξ+1
        Δσxᶠᶜᵃ[i, j] = 2central_angle_degrees((φᶠᶜᵃ[i, j], λᶠᶜᵃ[i, j]), (φᶜᶜᵃ[i-1, j], λᶜᶜᵃ[i-1, j]))
    end


    # Δσxᶜᶠᵃ

    for j in 1:Nη+1, i in 1:Nξ
        Δσxᶜᶠᵃ[i, j] =  central_angle_degrees((φᶠᶠᵃ[i+1, j], λᶠᶠᵃ[i+1, j]), (φᶠᶠᵃ[i, j], λᶠᶠᵃ[i, j]))
    end


    # Δσxᶠᶠᵃ

    for j in 1:Nη+1, i in 2:Nξ+1
        Δσxᶠᶠᵃ[i, j] =  central_angle_degrees((φᶜᶠᵃ[i, j], λᶜᶠᵃ[i, j]), (φᶜᶠᵃ[i-1, j], λᶜᶠᵃ[i-1, j]))
    end

    for j in 1:Nη+1
        i = 1
        Δσxᶠᶠᵃ[i, j] = 2central_angle_degrees((φᶜᶠᵃ[i, j], λᶜᶠᵃ[i, j]), (φᶠᶠᵃ[ i , j], λᶠᶠᵃ[ i , j]))
    end

    for j in 1:Nη+1
        i = Nξ+1
        Δσxᶠᶠᵃ[i, j] = 2central_angle_degrees((φᶠᶠᵃ[i, j], λᶠᶠᵃ[i, j]), (φᶜᶠᵃ[i-1, j], λᶜᶠᵃ[i-1, j]))
    end

    Δxᶜᶜᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    Δxᶠᶜᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    Δxᶜᶠᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)
    Δxᶠᶠᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)

    @. Δxᶜᶜᵃ = radius * deg2rad(Δσxᶜᶜᵃ)
    @. Δxᶠᶜᵃ = radius * deg2rad(Δσxᶠᶜᵃ)
    @. Δxᶜᶠᵃ = radius * deg2rad(Δσxᶜᶠᵃ)
    @. Δxᶠᶠᵃ = radius * deg2rad(Δσxᶠᶠᵃ)


    Δσyᶜᶜᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    Δσyᶠᶜᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    Δσyᶜᶠᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)
    Δσyᶠᶠᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)


    # Δσyᶜᶜᵃ

    for j in 1:Nη, i in 1:Nξ
        Δσyᶜᶜᵃ[i, j] =  central_angle_degrees((φᶜᶠᵃ[i, j+1], λᶜᶠᵃ[i, j+1]), (φᶜᶠᵃ[i, j], λᶜᶠᵃ[i, j]))
    end


    # Δσyᶜᶠᵃ

    for j in 1:Nη+1, i in 1:Nξ
        Δσyᶜᶠᵃ[i, j] =  central_angle_degrees((φᶜᶜᵃ[i, j], λᶜᶜᵃ[i, j]), (φᶜᶜᵃ[i, j-1], λᶜᶜᵃ[i, j-1]))
    end

    for i in 1:Nξ
        j = 1
        Δσyᶜᶠᵃ[i, j] = 2central_angle_degrees((φᶜᶜᵃ[i, j], λᶜᶜᵃ[i, j]), (φᶜᶠᵃ[i,  j ], λᶜᶠᵃ[i,  j ]))
    end

    for i in 1:Nξ
        j = Nη+1
        Δσyᶜᶠᵃ[i, j] = 2central_angle_degrees((φᶜᶠᵃ[i, j], λᶜᶠᵃ[i, j]), (φᶜᶜᵃ[i, j-1], λᶜᶜᵃ[i, j-1]))
    end


    # Δσyᶠᶜᵃ

    for j in 1:Nη, i in 1:Nξ+1
        Δσyᶠᶜᵃ[i, j] =  central_angle_degrees((φᶠᶠᵃ[i, j+1], λᶠᶠᵃ[i, j+1]), (φᶠᶠᵃ[i, j], λᶠᶠᵃ[i, j]))
    end


    # Δσyᶠᶠᵃ

    for j in 1:Nη+1, i in 1:Nξ+1
        Δσyᶠᶠᵃ[i, j] =  central_angle_degrees((φᶠᶜᵃ[i, j], λᶠᶜᵃ[i, j]), (φᶠᶜᵃ[i, j-1], λᶠᶜᵃ[i, j-1]))
    end

    for i in 1:Nξ+1
        j = 1
        Δσyᶠᶠᵃ[i, j] = 2central_angle_degrees((φᶠᶜᵃ[i, j], λᶠᶜᵃ[i, j]), (φᶠᶠᵃ[i,  j ], λᶠᶠᵃ[i,  j ]))
    end
    
    for i in 1:Nξ+1
        j = Nη+1  
        Δσyᶠᶠᵃ[i, j] = 2central_angle_degrees((φᶠᶠᵃ[i, j], λᶠᶠᵃ[i, j]), (φᶠᶜᵃ[i, j-1], λᶠᶜᵃ[i, j-1]))
    end


    Δyᶜᶜᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    Δyᶠᶜᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    Δyᶜᶠᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)
    Δyᶠᶠᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)

    @. Δyᶜᶜᵃ = radius * deg2rad(Δσyᶜᶜᵃ)
    @. Δyᶠᶜᵃ = radius * deg2rad(Δσyᶠᶜᵃ)
    @. Δyᶜᶠᵃ = radius * deg2rad(Δσyᶜᶠᵃ)
    @. Δyᶠᶠᵃ = radius * deg2rad(Δσyᶠᶠᵃ)


    # Area metrics

    #=
    The areas Az correspond to spherical quadrilaterals. To compute areas Az first we
    find the vertices a, b, c, d of the corresponding quadrilateral and then

        Az = spherical_area_quadrilateral(a, b, c, d) * radius^2

    For quadrilaterals near the boundary of the OrthogonalSphericalShellGrid some of the 
    vertices lie outside the grid! For example, the area Azᶠᶜᵃ[1, j] corressponds to a
    quadrilateral with vertices:

        a = (φᶜᶠᵃ[0,  j ], λᶜᶠᵃ[0,  j ])
        b = (φᶜᶠᵃ[1,  j ], λᶜᶠᵃ[1,  j ])
        c = (φᶜᶠᵃ[1, j+1], λᶜᶠᵃ[1, j+1])
        d = (φᶜᶠᵃ[0, j+1], λᶜᶠᵃ[0, j+1])

    Notice that vertices a and d are outside the boundaries of the grid. In those cases, we
    employ symmetry arguments and, e.g., compute Azᶠᶜᵃ[1, j] as

        2 * spherical_area_quadrilateral(ã, b, c, d̃) * radius^2

    where, ã = (φᶠᶠᵃ[1,  j ], λᶠᶠᵃ[1,  j ]) and d̃ = (φᶠᶠᵃ[1, j+1], λᶠᶠᵃ[1, j+1])
    =#

    Azᶜᶜᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    Azᶠᶜᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη   + 2Hy), -Hx, -Hy)
    Azᶜᶠᵃ = OffsetArray(zeros(Nξ   + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)
    Azᶠᶠᵃ = OffsetArray(zeros(Nξ+1 + 2Hx, Nη+1 + 2Hy), -Hx, -Hy)


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
    a = lat_lon_to_cartesian(φᶠᶠᵃ[i-1, j-1], λᶠᶠᵃ[i-1, j-1], 1)
    b = lat_lon_to_cartesian(φᶠᶠᵃ[ i , j-1], λᶠᶠᵃ[ i , j-1], 1)
    c = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)
    d = lat_lon_to_cartesian(φᶠᶠᵃ[i-1,  j ], λᶠᶠᵃ[i-1,  j ], 1)

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


    return OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture, Nξ, Nη, Nz, Hx, Hy, Hz,
                                                     λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                     φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ,
                                                     zᵃᵃᶜ,  zᵃᵃᶠ,
                                                    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                    Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ,
                                                    Δz, Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                                    radius)
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
OrthogonalSphericalShellGrid(FT::DataType; kwargs...) = OrthogonalSphericalShellGrid(CPU(), FT; kwargs...)

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
    Txᶠᶜ = total_length(loc_fc[1](), topology[1](), N[1], H[1])
    Txᶜᶠ = total_length(loc_cf[1](), topology[1](), N[1], H[1])
    Tyᶠᶜ = total_length(loc_fc[2](), topology[2](), N[2], H[2])
    Tyᶜᶠ = total_length(loc_cf[2](), topology[2](), N[2], H[2])

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

"""
    get_center_and_extents_of_shell(grid::OSSG)

Return the latitude-longitude coordinates of the center of the shell `(λ_center, φ_center)`
and also the longitudinal and latitudinal extend of the shell `(extend_λ, extend_φ)`.
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
    λ_center = xnode(i_center, j_center, 1, grid, ℓx, ℓy, Center())
    φ_center = ynode(i_center, j_center, 1, grid, ℓx, ℓy, Center())

    # the Δλ, Δφ are approximate if ξ, η are not symmetric about 0
    if mod(Ny, 2) == 0
        extend_λ = rad2deg(sum(grid.Δxᶜᶠᵃ[:, j_center])) / grid.radius
    elseif mod(Ny, 2) == 1
        extend_λ = rad2deg(sum(grid.Δxᶜᶜᵃ[:, j_center])) / grid.radius
    end

    if mod(Nx, 2) == 0
        extend_φ = rad2deg(sum(grid.Δyᶠᶜᵃ[i_center, :])) / grid.radius
    elseif mod(Nx, 2) == 1
        extend_φ = rad2deg(sum(grid.Δyᶜᶜᵃ[i_center, :])) / grid.radius
    end

    return (λ_center, φ_center), (extend_λ, extend_φ)
end

function Base.show(io::IO, grid::OrthogonalSphericalShellGrid, withsummary=true)
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)

    λ₁, λ₂ = minimum(grid.λᶠᶠᵃ[1:Nx+1, 1:Ny+1]), maximum(grid.λᶠᶠᵃ[1:Nx+1, 1:Ny+1])
    φ₁, φ₂ = minimum(grid.φᶠᶠᵃ[1:Nx+1, 1:Ny+1]), maximum(grid.φᶠᶠᵃ[1:Nx+1, 1:Ny+1])
    z₁, z₂ = domain(topology(grid, 3)(), Nz, grid.zᵃᵃᶠ)

    (λc, φc), (Δλ, Δφ) = get_center_and_extents_of_shell(grid)

    λc = round(λc, digits=4)
    φc = round(φc, digits=4)

    center_str = "centered at (λ, φ) = (" * prettysummary(λc) * ", " * prettysummary(φc) * ")"

    if abs(φc) ≈  90; center_str = "centered at: North Pole, (λ, φ) = (" * prettysummary(λc) * ", " * prettysummary(φc) * ")"; end
    if abs(φc) ≈ -90; center_str = "centered at: South Pole, (λ, φ) = (" * prettysummary(λc) * ", " * prettysummary(φc) * ")"; end

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

#####
##### Grid nodes
#####

function nodes(grid::RectilinearGrid, ℓx, ℓy, ℓz; reshape=false, with_halos=false)
    λ = λnodes(grid, ℓx, ℓy, ℓz; with_halos)
    φ = φnodes(grid, ℓx, ℓy, ℓz; with_halos)
    z = znodes(grid, ℓx, ℓy, ℓz; with_halos)

    if reshape
        N = (length(x), length(y), length(z))
        λ = Base.reshape(λ, N[1], Ν[2], 1)
        φ = Base.reshape(φ, N[1], N[2], 1)
        z = Base.reshape(z, 1, 1, N[3])
    end

    return (x, y, z)
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
@inline ynodes(grid::OSSG, ℓy, ℓy; with_halos=false) = grid.radius * deg2rad.(φnodes(grid, ℓx, ℓy; with_halos=with_halos))

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

@inline zspacings(grid::OSSG, ℓz; with_halos=false) = grid.Δz

@inline xspacings(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = xspacings(grid, ℓx, ℓy; with_halos)
@inline yspacings(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = yspacings(grid, ℓx, ℓy; with_halos)
@inline zspacings(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = zspacings(grid, ℓz; with_halos)
