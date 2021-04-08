using Rotations
using Oceananigans.Grids
using Oceananigans.Grids: R_Earth, interior_indices

import Base: show, size, eltype
import Oceananigans.Grids: topology, domain_string

struct CubedSphereFaceConnectivityDetails{F, S}
    face :: F
    side :: S
end

short_string(deets::CubedSphereFaceConnectivityDetails) = "face $(deets.face) $(deets.side) side"

Base.show(io::IO, deets::CubedSphereFaceConnectivityDetails) =
    print(io, "CubedSphereFaceConnectivityDetails: $(short_string(deets))")

struct CubedSphereFaceConnectivity{W, E, S, N}
     west :: W
     east :: E
    south :: S
    north :: N
end

CubedSphereFaceConnectivity(; west, east, south, north) =
    CubedSphereFaceConnectivity(west, east, south, north)

function Base.show(io::IO, connectivity::CubedSphereFaceConnectivity)
    print(io, "CubedSphereFaceConnectivity:\n",
              "├── west: $(short_string(connectivity.west))\n",
              "├── east: $(short_string(connectivity.east))\n",
              "├── south: $(short_string(connectivity.south))\n",
              "└── north: $(short_string(connectivity.north))")
end

function default_face_connectivity()
    # See figure 8.4 of https://mitgcm.readthedocs.io/en/latest/phys_pkgs/exch2.html?highlight=cube%20sphere#fig-6tile
    #
    #                         face  F5   face  F6
    #                       +----------+----------+
    #                       |    ↑↑    |    ↑↑    |
    #                       |    1W    |    1S    |
    #                       |←3N F5 6W→|←5E F6 2S→|
    #                       |    4N    |    4E    |
    #              face  F3 |    ↓↓    |    ↓↓    |
    #            +----------+----------+----------+
    #            |    ↑↑    |    ↑↑    |
    #            |    5W    |    5S    |
    #            |←1N F3 4W→|←3E F4 6S→|
    #            |    2N    |    2E    |
    #            |    ↓↓    |    ↓↓    |
    # +----------+----------+----------+
    # |    ↑↑    |    ↑↑    | face  F4
    # |    3W    |    3S    |
    # |←5N F1 2W→|←1E F2 4S→|
    # |    6N    |    6E    |
    # |    ↓↓    |    ↓↓    |
    # +----------+----------+
    #   face  F1   face  F2

    face1_connectivity = CubedSphereFaceConnectivity(
        west  = CubedSphereFaceConnectivityDetails(5, :north),
        east  = CubedSphereFaceConnectivityDetails(2, :west),
        south = CubedSphereFaceConnectivityDetails(6, :north),
        north = CubedSphereFaceConnectivityDetails(3, :west),
    )

    face2_connectivity = CubedSphereFaceConnectivity(
        west  = CubedSphereFaceConnectivityDetails(1, :east),
        east  = CubedSphereFaceConnectivityDetails(4, :south),
        south = CubedSphereFaceConnectivityDetails(6, :east),
        north = CubedSphereFaceConnectivityDetails(3, :south),
    )

    face3_connectivity = CubedSphereFaceConnectivity(
        west  = CubedSphereFaceConnectivityDetails(1, :north),
        east  = CubedSphereFaceConnectivityDetails(4, :west),
        south = CubedSphereFaceConnectivityDetails(2, :north),
        north = CubedSphereFaceConnectivityDetails(5, :west),
    )

    face4_connectivity = CubedSphereFaceConnectivity(
        west  = CubedSphereFaceConnectivityDetails(3, :east),
        east  = CubedSphereFaceConnectivityDetails(6, :south),
        south = CubedSphereFaceConnectivityDetails(2, :east),
        north = CubedSphereFaceConnectivityDetails(5, :south),
    )

    face5_connectivity = CubedSphereFaceConnectivity(
        west  = CubedSphereFaceConnectivityDetails(3, :north),
        east  = CubedSphereFaceConnectivityDetails(6, :west),
        south = CubedSphereFaceConnectivityDetails(4, :north),
        north = CubedSphereFaceConnectivityDetails(1, :west),
    )


    face6_connectivity = CubedSphereFaceConnectivity(
        west  = CubedSphereFaceConnectivityDetails(5, :east),
        east  = CubedSphereFaceConnectivityDetails(2, :south),
        south = CubedSphereFaceConnectivityDetails(4, :east),
        north = CubedSphereFaceConnectivityDetails(1, :south),
    )

    face_connectivity = (
        face1_connectivity,
        face2_connectivity,
        face3_connectivity,
        face4_connectivity,
        face5_connectivity,
        face6_connectivity
    )

    return face_connectivity
end

# Note: I think we want to keep faces and face_connectivity tuples
# so it's easy to support an arbitrary number of faces.

struct ConformalCubedSphereGrid{FT, F, C}
                faces :: F
    face_connectivity :: C
end

function ConformalCubedSphereGrid(FT=Float64; face_size, z, radius=R_Earth)
    @warn "ConformalCubedSphereGrid is experimental: use with caution!"

    # +z face (face 1)
    z⁺_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=nothing)

    # +x face (face 2)
    x⁺_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotX(π/2))

    # +y face (face 3)
    y⁺_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotY(π/2))

    # -x face (face 4)
    x⁻_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotX(-π/2))

    # -y face (face 5)
    y⁻_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotY(-π/2))

    # -z face (face 6)
    z⁻_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotX(π))

    faces = (
        z⁺_face_grid,
        x⁺_face_grid,
        y⁺_face_grid,
        x⁻_face_grid,
        y⁻_face_grid,
        z⁻_face_grid
    )

    face_connectivity = default_face_connectivity()

    return ConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity)}(faces, face_connectivity)
end

function ConformalCubedSphereGrid(filepath::AbstractString, FT=Float64; Nz, z, radius = R_Earth, halo = (1, 1, 1))
    @warn "ConformalCubedSphereGrid is experimental: use with caution!"

    face_topo = (Periodic, Periodic, Bounded)
    face_kwargs = (Nz=Nz, z=z, topology=face_topo, radius=radius, halo=halo)

    faces = Tuple(ConformalCubedSphereFaceGrid(filepath, FT; face=n, face_kwargs...) for n in 1:6)

    face_connectivity = default_face_connectivity()

    return ConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity)}(faces, face_connectivity)
end

function Base.show(io::IO, grid::ConformalCubedSphereGrid{FT}) where FT
    Nx, Ny, Nz, Nf = size(grid)
    print(io, "ConformalCubedSphereGrid{$FT}: $Nf faces with size = ($Nx, $Ny, $Nz)")
end

#####
##### Nodes for ConformalCubedSphereFaceGrid
#####

λnode(LX::Face,   LY::Face,   LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.λᶠᶠᵃ[i, j]
λnode(LX::Center, LY::Center, LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.λᶜᶜᵃ[i, j]
φnode(LX::Face,   LY::Face,   LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.φᶠᶠᵃ[i, j]
φnode(LX::Center, LY::Center, LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.φᶜᶜᵃ[i, j]

znode(LX, LY, LZ::Face,   i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.zᵃᵃᶠ[k]
znode(LX, LY, LZ::Center, i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.zᵃᵃᶜ[k]

# FIXME!
λnode(LX::Face, LY::Center, LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.λᶠᶠᵃ[i, j]
λnode(LX::Center, LY::Face, LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.λᶠᶠᵃ[i, j]
φnode(LX::Face, LY::Center, LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.φᶠᶠᵃ[i, j]
φnode(LX::Center, LY::Face, LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.φᶠᶠᵃ[i, j]

λnodes(LX::Face, LY::Face, LZ, grid::ConformalCubedSphereFaceGrid{TX, TY}) where {TX, TY} =
    view(grid.λᶠᶠᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

λnodes(LX::Center, LY::Center, LZ, grid::ConformalCubedSphereFaceGrid{TX, TY}) where {TX, TY} =
    view(grid.λᶜᶜᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

φnodes(LX::Face, LY::Face, LZ, grid::ConformalCubedSphereFaceGrid{TX, TY}) where {TX, TY} =
    view(grid.φᶠᶠᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

φnodes(LX::Center, LY::Center, LZ, grid::ConformalCubedSphereFaceGrid{TX, TY}) where {TX, TY} =
    view(grid.φᶜᶜᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

# Nodes for ::ConformalCubedSphereGrid
# Not sure how to best represent these so will concatenate along dim 3 for now.

λnodes(LX, LY, LZ, grid::ConformalCubedSphereGrid) = cat(Tuple(λnodes(LX, LY, LZ, grid_face) for grid_face in grid.faces)..., dims=3)
φnodes(LX, LY, LZ, grid::ConformalCubedSphereGrid) = cat(Tuple(φnodes(LX, LY, LZ, grid_face) for grid_face in grid.faces)..., dims=3)

#####
##### Grid utils
#####

Base.size(grid::ConformalCubedSphereGrid) = (size(grid.faces[1])..., length(grid.faces))

Base.eltype(grid::ConformalCubedSphereGrid{FT}) where FT = FT

topology(::ConformalCubedSphereGrid) = (Bounded, Bounded, Bounded)

# Not sure what to put. Gonna leave it blank so that Base.show(io::IO, operation::AbstractOperation) doesn't error.
domain_string(grid::ConformalCubedSphereFaceGrid) = ""
domain_string(grid::ConformalCubedSphereGrid) = ""
