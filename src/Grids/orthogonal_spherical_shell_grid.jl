using CubedSphere
using JLD2
using OffsetArrays
using Adapt
using Adapt: adapt_structure

using Oceananigans
using Oceananigans.Grids: xnode, ynode,
                          all_x_nodes, all_y_nodes, 
                          prettysummary, coordinate_summary

struct OrthogonalSphericalShellGrid{FT, TX, TY, TZ, A, R, FR, Arch} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    Nx :: Int
    Ny :: Int
    Nz :: Int
    Hx :: Int
    Hy :: Int
    Hz :: Int
    ξₗ :: FT    # left-most domain for cube's ξ coordinate
    ξᵣ :: FT    # right-most domain for cube's ξ coordinate
    ηₗ :: FT    # left-most domain for cube's η coordinate
    ηᵣ :: FT    # right-most domain for cube's η coordinate
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

    OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture::Arch,
                                             Nx, Ny, Nz,
                                             Hx, Hy, Hz, ξₗ, ξᵣ, ηₗ, ηᵣ,
                                              λᶜᶜᵃ :: A,  λᶠᶜᵃ :: A,  λᶜᶠᵃ :: A,  λᶠᶠᵃ :: A,
                                              φᶜᶜᵃ :: A,  φᶠᶜᵃ :: A,  φᶜᶠᵃ :: A,  φᶠᶠᵃ :: A, zᵃᵃᶜ :: R, zᵃᵃᶠ :: R,
                                             Δxᶜᶜᵃ :: A, Δxᶠᶜᵃ :: A, Δxᶜᶠᵃ :: A, Δxᶠᶠᵃ :: A,
                                             Δyᶜᶜᵃ :: A, Δyᶜᶠᵃ :: A, Δyᶠᶜᵃ :: A, Δyᶠᶠᵃ :: A, Δzᵃᵃᶜ :: FR, Δzᵃᵃᶠ :: FR,
                                             Azᶜᶜᵃ :: A, Azᶠᶜᵃ :: A, Azᶜᶠᵃ :: A, Azᶠᶠᵃ :: A,
                                             radius :: FT) where {TX, TY, TZ, FT, A, R, FR, Arch} =
        new{FT, TX, TY, TZ, A, R, FR, Arch}(architecture,
                                            Nx, Ny, Nz,
                                            Hx, Hy, Hz,
                                            ξₗ, ξᵣ, ηₗ, ηᵣ,
                                            λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                            φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
                                            Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                            Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                                            Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
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

    Δzᵃᵃᶜ = ξη_grid.Δzᵃᵃᶜ
    Δzᵃᵃᶠ = ξη_grid.Δzᵃᵃᶠ
    zᵃᵃᶠ = ξη_grid.zᵃᵃᶠ
    zᵃᵃᶜ = ξη_grid.zᵃᵃᶜ

    ## CompuNξᶠte staggered grid Cartesian coordinates (X, Y, Z) on the unit sphere.

    Xᶜᶜᵃ = zeros(total_length(Center, topology[1], Nξ, 0), total_length(Center, topology[2], Nη, 0))
    Xᶠᶜᵃ = zeros(total_length(Face,   topology[1], Nξ, 0), total_length(Center, topology[2], Nη, 0))
    Xᶜᶠᵃ = zeros(total_length(Center, topology[1], Nξ, 0), total_length(Face,   topology[2], Nη, 0))
    Xᶠᶠᵃ = zeros(total_length(Face,   topology[1], Nξ, 0), total_length(Face,   topology[2], Nη, 0))

    Yᶜᶜᵃ = zeros(total_length(Center, topology[1], Nξ, 0), total_length(Center, topology[2], Nη, 0))
    Yᶠᶜᵃ = zeros(total_length(Face,   topology[1], Nξ, 0), total_length(Center, topology[2], Nη, 0))
    Yᶜᶠᵃ = zeros(total_length(Center, topology[1], Nξ, 0), total_length(Face,   topology[2], Nη, 0))
    Yᶠᶠᵃ = zeros(total_length(Face,   topology[1], Nξ, 0), total_length(Face,   topology[2], Nη, 0))

    Zᶜᶜᵃ = zeros(total_length(Center, topology[1], Nξ, 0), total_length(Center, topology[2], Nη, 0))
    Zᶠᶜᵃ = zeros(total_length(Face,   topology[1], Nξ, 0), total_length(Center, topology[2], Nη, 0))
    Zᶜᶠᵃ = zeros(total_length(Center, topology[1], Nξ, 0), total_length(Face,   topology[2], Nη, 0))
    Zᶠᶠᵃ = zeros(total_length(Face,   topology[1], Nξ, 0), total_length(Face,   topology[2], Nη, 0))

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
    λᶜᶜᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    λᶠᶜᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    λᶜᶠᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)
    λᶠᶠᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)

    φᶜᶜᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    φᶠᶜᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    φᶜᶠᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)
    φᶠᶠᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)

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

    Δσxᶜᶜᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    Δσxᶠᶜᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    Δσxᶜᶠᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)
    Δσxᶠᶠᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)


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

    Δxᶜᶜᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    Δxᶠᶜᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    Δxᶜᶠᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)
    Δxᶠᶠᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)

    @. Δxᶜᶜᵃ = radius * deg2rad(Δσxᶜᶜᵃ)
    @. Δxᶠᶜᵃ = radius * deg2rad(Δσxᶠᶜᵃ)
    @. Δxᶜᶠᵃ = radius * deg2rad(Δσxᶜᶠᵃ)
    @. Δxᶠᶠᵃ = radius * deg2rad(Δσxᶠᶠᵃ)


    Δσyᶜᶜᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    Δσyᶠᶜᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    Δσyᶜᶠᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)
    Δσyᶠᶠᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)


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


    Δyᶜᶜᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    Δyᶠᶜᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    Δyᶜᶠᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)
    Δyᶠᶠᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)

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

    Azᶜᶜᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    Azᶠᶜᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Center, topology[2], Nη, Hy)), -Hx, -Hy)
    Azᶜᶠᵃ = OffsetArray(zeros(total_length(Center, topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)
    Azᶠᶠᵃ = OffsetArray(zeros(total_length(Face,   topology[1], Nξ, Hx), total_length(Face,   topology[2], Nη, Hy)), -Hx, -Hy)


    # Azᶜᶜᵃ

    # approximate the areas Az = Δx * Δy

    @. Azᶜᶜᵃ = Δxᶜᶜᵃ * Δyᶜᶜᵃ
    @. Azᶠᶜᵃ = Δxᶠᶜᵃ * Δyᶠᶜᵃ
    @. Azᶜᶠᵃ = Δxᶜᶠᵃ * Δyᶜᶠᵃ
    @. Azᶠᶠᵃ = Δxᶠᶠᵃ * Δyᶠᶠᵃ


    # for interior points compute Az as the area of the corresponding spherical quadrilateral  

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

    coordinate_arrays = (λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ, φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, zᵃᵃᶜ,  zᵃᵃᶠ)
    coordinate_arrays = map(a -> arch_array(architecture, a), coordinate_arrays)

    metric_arrays = (Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δzᵃᵃᶜ, Δzᵃᵃᶠ, Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ)
    metric_arrays = map(a -> arch_array(architecture, a), metric_arrays)

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture, Nξ, Nη, Nz, Hx, Hy, Hz, ξ..., η...,
                                                    coordinate_arrays...,
                                                    metric_arrays...,
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
    ξη_grid = RectilinearGrid(architecture, FT; size=(1, 1, Nz), x=ξ, y=η, z, topology, halo)

     zᵃᵃᶠ = ξη_grid.zᵃᵃᶠ
     zᵃᵃᶜ = ξη_grid.zᵃᵃᶜ
    Δzᵃᵃᶜ = ξη_grid.Δzᵃᵃᶜ
    Δzᵃᵃᶠ = ξη_grid.Δzᵃᵃᶠ

    ## Read everything else from the file

    file = jldopen(filepath, "r")["face$panel"]

    Nξ, Nη = size(file["λᶠᶠᵃ"]) .- 1

    N = (Nξ, Nη, Nz)
    H = halo

    loc_cc = (Center, Center)
    loc_cf = (Center, Face)
    loc_fc = (Face,   Center)
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
    Txᶠᶜ = total_length(loc_fc[1], topology[1], N[1], H[1])
    Txᶜᶠ = total_length(loc_cf[1], topology[1], N[1], H[1])
    Tyᶠᶜ = total_length(loc_fc[2], topology[2], N[2], H[2])
    Tyᶜᶠ = total_length(loc_cf[2], topology[2], N[2], H[2])

    λᶠᶜᵃ = offset_data(zeros(FT, architecture, Txᶠᶜ, Tyᶠᶜ), loc_fc, topology[1:2], N[1:2], H[1:2])
    λᶜᶠᵃ = offset_data(zeros(FT, architecture, Txᶜᶠ, Tyᶜᶠ), loc_cf, topology[1:2], N[1:2], H[1:2])
    φᶠᶜᵃ = offset_data(zeros(FT, architecture, Txᶠᶜ, Tyᶠᶜ), loc_fc, topology[1:2], N[1:2], H[1:2])
    φᶜᶠᵃ = offset_data(zeros(FT, architecture, Txᶜᶠ, Tyᶜᶠ), loc_cf, topology[1:2], N[1:2], H[1:2])

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture, Nξ, Nη, Nz, Hx, Hy, Hz, ξ..., η...,
                                                     λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                     φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ,
                                                     zᵃᵃᶜ,  zᵃᵃᶠ,
                                                    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                    Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ,
                                                    Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                                                    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
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
    Δzᵃᵃᶜ = arch_array(arch, grid.Δzᵃᵃᶜ)
    Δzᵃᵃᶠ = arch_array(arch, grid.Δzᵃᵃᶠ)

    TX, TY, TZ = topology(grid)

    new_grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(arch,
                                                        grid.Nx, grid.Ny, grid.Nz,
                                                        grid.Hx, grid.Hy, grid.Hz,
                                                        grid.ξₗ, grid.ξᵣ, grid.ηₗ, grid.ηᵣ,
                                                        horizontal_coordinate_data..., zᵃᵃᶜ, zᵃᵃᶠ,
                                                        horizontal_grid_spacing_data..., Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                                                        horizontal_area_data..., grid.radius)

    return new_grid
end

function Adapt.adapt_structure(to, grid::OrthogonalSphericalShellGrid)
    TX, TY, TZ = topology(grid)

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(nothing,
                                                    grid.Nx, grid.Ny, grid.Nz,
                                                    grid.Hx, grid.Hy, grid.Hz, 
                                                    grid.ξₗ, grid.ξᵣ,
                                                    grid.ηₗ, grid.ηᵣ,
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

function get_center_and_extents_of_shell(grid)
    Nx, Ny, _ = size(grid)

    Nxc, Nyc = (Nx÷2)+1, (Ny÷2)+1

    if mod(Nx, 2) == 0
        LX = Face()
    elseif mod(Nx, 2) == 1
        LX = Center()
    end

    if mod(Ny, 2) == 0
        LY = Face()
    elseif mod(Ny, 2) == 1
        LY = Center()
    end

    # the shell's center at (λc, φc)
    λc = xnode(LX, LY, Center(), Nxc, Nyc, 1, grid)
    φc = ynode(LX, LY, Center(), Nxc, Nyc, 1, grid)

    # the Δλ, Δφ are approximate if ξ, η are not symmetric about 0
    if mod(Ny, 2) == 0
        Δλ = rad2deg.(sum(grid.Δxᶜᶠᵃ[:, Int(Ny/2+1)])) / grid.radius
    elseif mod(Ny, 2) == 1
        Δλ = rad2deg.(sum(grid.Δxᶜᶜᵃ[:, Int((Ny+1)/2)])) / grid.radius
    end

    if mod(Nx, 2) == 0
        Δφ = rad2deg.(sum(grid.Δyᶠᶜᵃ[Int(Nx/2+1), :])) / grid.radius
    elseif mod(Nx, 2) == 1
        Δφ = rad2deg.(sum(grid.Δyᶜᶜᵃ[Int((Nx+1)/2), :])) / grid.radius
    end

    return (λc, φc), (Δλ, Δφ)
end

function Base.show(io::IO, grid::OrthogonalSphericalShellGrid, withsummary=true)
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)

    λ₁, λ₂ = minimum(grid.λᶠᶠᵃ[1:Nx+1, 1:Ny+1]), maximum(grid.λᶠᶠᵃ[1:Nx+1, 1:Ny+1])
    φ₁, φ₂ = minimum(grid.φᶠᶠᵃ[1:Nx+1, 1:Ny+1]), maximum(grid.φᶠᶠᵃ[1:Nx+1, 1:Ny+1])
    z₁, z₂ = domain(topology(grid, 3), grid.Nz, grid.zᵃᵃᶠ)

    (λc, φc), (Δλ, Δφ) = get_center_and_extents_of_shell(grid)

    λc = round(λc, digits=4)
    φc = round(φc, digits=4)

    center_str = "centered at (λ, φ) = (" * prettysummary(λc) * ", " * prettysummary(φc) * ")"

    if abs(φc) ≈ 90; center_str = "centered at: North Pole, (λ, φ) = (" * prettysummary(λc) * ", " * prettysummary(φc) * ")"; end
    if abs(φc) ≈ -90; center_str = "centered at: South Pole, (λ, φ) = (" * prettysummary(λc) * ", " * prettysummary(φc) * ")"; end

    x_summary = domain_summary(TX(), "λ", λ₁, λ₂)
    y_summary = domain_summary(TY(), "φ", φ₁, φ₂)
    z_summary = domain_summary(TZ(), "z", z₁, z₂)

    longest = max(length(x_summary), length(y_summary), length(z_summary))

    x_summary = "longitude: extent $(prettysummary(Δλ)) " * coordinate_summary(rad2deg.(grid.Δxᶠᶠᵃ[1:Nx+1, 1:Ny+1] ./ grid.radius), "λ")
    y_summary = "latitude:  extent $(prettysummary(Δφ)) " * coordinate_summary(rad2deg.(grid.Δyᶠᶠᵃ[1:Nx+1, 1:Ny+1] ./ grid.radius), "φ")
    z_summary = "z:         " * dimension_summary(TZ(), "z", z₁, z₂, grid.Δzᵃᵃᶜ, longest - length(z_summary))

    if withsummary
        print(io, summary(grid), "\n")
    end

    return print(io, "├── ", center_str, "\n",
                     "├── ", x_summary, "\n",
                     "├── ", y_summary, "\n",
                     "└── ", z_summary)
end

@inline z_domain(grid::OrthogonalSphericalShellGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)
@inline cpu_face_constructor_z(grid::ZRegOrthogonalSphericalShellGrid) = z_domain(grid)

function with_halo(new_halo, old_grid::OrthogonalSphericalShellGrid; rotation=nothing)

    size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)
    topo = topology(old_grid)

    ξ = (old_grid.ξₗ, old_grid.ξᵣ)
    η = (old_grid.ηₗ, old_grid.ηᵣ)

    z = cpu_face_constructor_z(old_grid)

    new_grid = OrthogonalSphericalShellGrid(architecture(old_grid), eltype(old_grid);
                                            size, z, ξ, η,
                                            topology = topo,
                                            radius = old_grid.radius,
                                            halo = new_halo,
                                            rotation)

    return new_grid
end

@inline xnodes(grid::OSSG, LX::Face,   LY::Face, ; with_halos=false) = with_halos ? grid.λᶠᶠᵃ :
    view(grid.λᶠᶠᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline xnodes(grid::OSSG, LX::Face,   LY::Center; with_halos=false) = with_halos ? grid.λᶠᶜᵃ :
    view(grid.λᶠᶜᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline xnodes(grid::OSSG, LX::Center, LY::Face, ; with_halos=false) = with_halos ? grid.λᶜᶠᵃ :
    view(grid.λᶜᶠᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline xnodes(grid::OSSG, LX::Center, LY::Center; with_halos=false) = with_halos ? grid.λᶜᶜᵃ :
    view(grid.λᶜᶜᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))

@inline ynodes(grid::OSSG, LX::Face,   LY::Face, ; with_halos=false) = with_halos ? grid.φᶠᶠᵃ :
    view(grid.φᶠᶠᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline ynodes(grid::OSSG, LX::Face,   LY::Center; with_halos=false) = with_halos ? grid.φᶠᶜᵃ :
    view(grid.φᶠᶜᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline ynodes(grid::OSSG, LX::Center, LY::Face, ; with_halos=false) = with_halos ? grid.φᶜᶠᵃ :
    view(grid.φᶜᶠᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline ynodes(grid::OSSG, LX::Center, LY::Center; with_halos=false) = with_halos ? grid.φᶜᶜᵃ :
    view(grid.φᶜᶜᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))

@inline znodes(grid::OSSG, LZ::Face  ; with_halos=false) = with_halos ? grid.zᵃᵃᶠ : view(grid.zᵃᵃᶠ, interior_indices(typeof(LZ), topology(grid, 3), grid.Nz))
@inline znodes(grid::OSSG, LZ::Center; with_halos=false) = with_halos ? grid.zᵃᵃᶜ : view(grid.zᵃᵃᶜ, interior_indices(typeof(LZ), topology(grid, 3), grid.Nz))

@inline xnodes(grid::OSSG, LX, LY, LZ; with_halos=false) = xnodes(grid, LX, LY; with_halos)
@inline ynodes(grid::OSSG, LX, LY, LZ; with_halos=false) = ynodes(grid, LX, LY; with_halos)
@inline znodes(grid::OSSG, LX, LY, LZ; with_halos=false) = znodes(grid, LZ    ; with_halos)

@inline xnode(i, j, grid::OSSG, ::Center, ::Center) = @inbounds grid.λᶜᶜᵃ[i, j]
@inline xnode(i, j, grid::OSSG, ::Face  , ::Center) = @inbounds grid.λᶠᶜᵃ[i, j]
@inline xnode(i, j, grid::OSSG, ::Center, ::Face  ) = @inbounds grid.λᶜᶠᵃ[i, j]
@inline xnode(i, j, grid::OSSG, ::Face  , ::Face  ) = @inbounds grid.λᶠᶠᵃ[i, j]

@inline ynode(i, j, grid::OSSG, ::Center, ::Center) = @inbounds grid.φᶜᶜᵃ[i, j]
@inline ynode(i, j, grid::OSSG, ::Face  , ::Center) = @inbounds grid.φᶠᶜᵃ[i, j]
@inline ynode(i, j, grid::OSSG, ::Center, ::Face  ) = @inbounds grid.φᶜᶠᵃ[i, j]
@inline ynode(i, j, grid::OSSG, ::Face  , ::Face  ) = @inbounds grid.φᶠᶠᵃ[i, j]

@inline znode(k, grid::OSSG, ::Center) = @inbounds grid.zᵃᵃᶜ[k]
@inline znode(k, grid::OSSG, ::Face  ) = @inbounds grid.zᵃᵃᶠ[k]

@inline xnode(i, j, k, grid::OSSG, LX, LY, LZ) = xnode(i, j, grid, LX, LY)
@inline ynode(i, j, k, grid::OSSG, LX, LY, LZ) = ynode(i, j, grid, LX, LY)
@inline znode(i, j, k, grid::OSSG, LX, LY, LZ) = znode(k, grid, LZ)

#####
##### Grid spacings in x, y, z (in meters)
#####

@inline xspacings(grid::OSSG, LX::Center, LY::Center; with_halos=false) =
    with_halos ? grid.Δxᶜᶜᵃ : view(grid.Δxᶜᶜᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline xspacings(grid::OSSG, LX::Face  , LY::Center; with_halos=false) =
    with_halos ? grid.Δxᶠᶜᵃ : view(grid.Δxᶠᶜᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline xspacings(grid::OSSG, LX::Center, LY::Face  ; with_halos=false) =
    with_halos ? grid.Δxᶜᶠᵃ : view(grid.Δxᶜᶠᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline xspacings(grid::OSSG, LX::Face  , LY::Face  ; with_halos=false) =
    with_halos ? grid.Δxᶠᶠᵃ : view(grid.Δxᶠᶠᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))

@inline yspacings(grid::OSSG, LX::Center, LY::Center; with_halos=false) =
    with_halos ? grid.Δyᶜᶜᵃ : view(grid.Δyᶜᶜᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline yspacings(grid::OSSG, LX::Face  , LY::Center; with_halos=false) =
    with_halos ? grid.Δyᶠᶜᵃ : view(grid.Δyᶠᶜᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline yspacings(grid::OSSG, LX::Center, LY::Face  ; with_halos=false) =
    with_halos ? grid.Δyᶜᶠᵃ : view(grid.Δyᶜᶠᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))
@inline yspacings(grid::OSSG, LX::Face  , LY::Face  ; with_halos=false) =
    with_halos ? grid.Δyᶠᶠᵃ : view(grid.Δyᶠᶠᵃ, interior_indices(typeof(LX), topology(grid, 1), grid.Nx), interior_indices(typeof(LY), topology(grid, 2), grid.Ny))

@inline zspacings(grid::OSSG,     LZ::Center; with_halos=false) = with_halos ? grid.Δzᵃᵃᶜ : view(grid.Δzᵃᵃᶜ, interior_indices(typeof(LZ), topology(grid, 3), grid.Nz))
@inline zspacings(grid::ZRegOSSG, LZ::Center; with_halos=false) = grid.Δzᵃᵃᶜ
@inline zspacings(grid::OSSG,     LZ::Face;   with_halos=false) = with_halos ? grid.Δzᵃᵃᶠ : view(grid.Δzᵃᵃᶠ, interior_indices(typeof(LZ), topology(grid, 3), grid.Nz))
@inline zspacings(grid::ZRegOSSG, LZ::Face;   with_halos=false) = grid.Δzᵃᵃᶠ

@inline xspacings(grid::OSSG, LX, LY, LZ; with_halos=false) = xspacings(grid, LX, LY; with_halos)
@inline yspacings(grid::OSSG, LX, LY, LZ; with_halos=false) = yspacings(grid, LX, LY; with_halos)
@inline zspacings(grid::OSSG, LX, LY, LZ; with_halos=false) = zspacings(grid, LZ; with_halos)

min_Δx(grid::OSSG) = topology(grid)[1] == Flat ? Inf : minimum(xspacings(grid, Center(), Center()))
min_Δy(grid::OSSG) = topology(grid)[2] == Flat ? Inf : minimum(yspacings(grid, Center(), Center()))
min_Δz(grid::OSSG) = topology(grid)[3] == Flat ? Inf : minimum(zspacings(grid, Center()))
