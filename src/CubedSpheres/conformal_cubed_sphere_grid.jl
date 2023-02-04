using Rotations
using Oceananigans.Grids
using Oceananigans.Grids: R_Earth, interior_indices

import Base: show, size, eltype
import Oceananigans.Grids: topology, architecture, halo_size, on_architecture

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

struct ConformalCubedSphereGrid{FT, F, C, Arch} <: AbstractHorizontallyCurvilinearGrid{FT, FullyConnected, FullyConnected, Bounded, Arch}
         architecture :: Arch
                faces :: F
    face_connectivity :: C
end

function ConformalCubedSphereGrid(arch = CPU(), FT=Float64; face_size, z, face_halo=(1, 1, 1), radius=R_Earth)
    @warn "ConformalCubedSphereGrid is experimental: use with caution!"

    size, halo = face_size, face_halo

    # +x face (face 1)
    x⁺_face_grid = OrthogonalSphericalShellGrid(arch, FT; size, z, halo, radius, rotation=RotX(π/2)*RotY(π/2))

    # +y face (face 2)
    y⁺_face_grid = OrthogonalSphericalShellGrid(arch, FT; size, z, halo, radius, rotation=RotY(π)*RotX(-π/2))

    # +z face (face 3)
    z⁺_face_grid = OrthogonalSphericalShellGrid(arch, FT; size, z, halo, radius, rotation=RotZ(π))

    # -x face (face 4)
    x⁻_face_grid = OrthogonalSphericalShellGrid(arch, FT; size, z, halo, radius, rotation=RotX(π)*RotY(-π/2))

    # -y face (face 5)
    y⁻_face_grid = OrthogonalSphericalShellGrid(arch, FT; size, z, halo, radius, rotation=RotY(π/2)*RotX(π/2))

    # -z face (face 6)
    z⁻_face_grid = OrthogonalSphericalShellGrid(arch, FT; size, z, halo, radius, rotation=RotZ(π/2)*RotX(π))

    faces = (
        x⁺_face_grid,
        y⁺_face_grid,
        z⁺_face_grid,
        x⁻_face_grid,
        y⁻_face_grid,
        z⁻_face_grid
    )

    face_connectivity = default_face_connectivity()

    return ConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity), typeof(arch)}(arch, faces, face_connectivity)
end

function ConformalCubedSphereGrid(filepath::AbstractString, arch = CPU(), FT=Float64; Nz, z, radius = R_Earth, halo = (1, 1, 1))
    @warn "ConformalCubedSphereGrid is experimental: use with caution!"

    face_topo = (FullyConnected, FullyConnected, Bounded)
    face_kwargs = (; Nz, z, topology=face_topo, radius, halo)

    faces = Tuple(OrthogonalSphericalShellGrid(filepath, arch, FT; face=n, face_kwargs...) for n in 1:6)

    face_connectivity = default_face_connectivity()

    grid = ConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity), typeof(arch)}(arch, faces, face_connectivity)

    fill_grid_metric_halos!(grid)
    fill_grid_metric_halos!(grid)

    return grid
end

function Base.show(io::IO, grid::ConformalCubedSphereGrid{FT}) where FT
    Nx, Ny, Nz, Nf = size(grid)
    print(io, "ConformalCubedSphereGrid{$FT}: $Nf faces with size = ($Nx, $Ny, $Nz)")
end

#####
##### Nodes for OrthogonalSphericalShellGrid
#####

@inline λnode(LX::Face,   LY::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶠᶠᵃ[i, j]
@inline λnode(LX::Face,   LY::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶠᶜᵃ[i, j]
@inline λnode(LX::Center, LY::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶜᶠᵃ[i, j]
@inline λnode(LX::Center, LY::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶜᶜᵃ[i, j]

@inline φnode(LX::Face,   LY::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶠᶠᵃ[i, j]
@inline φnode(LX::Face,   LY::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶠᶜᵃ[i, j]
@inline φnode(LX::Center, LY::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶜᶠᵃ[i, j]
@inline φnode(LX::Center, LY::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶜᶜᵃ[i, j]

@inline znode(LX, LY, LZ::Face,   i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.zᵃᵃᶠ[k]
@inline znode(LX, LY, LZ::Center, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.zᵃᵃᶜ[k]

λnodes(LX::Face, LY::Face, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.λᶠᶠᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

λnodes(LX::Face, LY::Center, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.λᶠᶜᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

λnodes(LX::Center, LY::Face, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.λᶜᶠᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

λnodes(LX::Center, LY::Center, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.λᶜᶜᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

φnodes(LX::Face, LY::Face, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.φᶠᶠᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

φnodes(LX::Face, LY::Center, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.φᶠᶜᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

φnodes(LX::Center, LY::Face, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.φᶜᶠᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

φnodes(LX::Center, LY::Center, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.φᶜᶜᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

#####
##### Grid utils
#####

Base.size(grid::ConformalCubedSphereGrid)      = (size(grid.faces[1])..., length(grid.faces))
Base.size(loc, grid::ConformalCubedSphereGrid) = size(loc, grid.faces[1])
Base.size(grid::ConformalCubedSphereGrid, i)   = size(grid)[i]
halo_size(ccsg::ConformalCubedSphereGrid)      = halo_size(first(ccsg.faces)) # hack

Base.eltype(grid::ConformalCubedSphereGrid{FT}) where FT = FT

topology(::ConformalCubedSphereGrid) = (Bounded, Bounded, Bounded)
topology(grid::ConformalCubedSphereGrid, i) = topology(grid)[i] 
architecture(grid::ConformalCubedSphereGrid) = grid.architecture

function on_architecture(arch, grid::ConformalCubedSphereGrid) 

    faces = Tuple(on_architecture(arch, grid.faces[n]) for n in 1:6)
    face_connectivity = grid.face_connectivity
    FT = eltype(grid)
    
    return ConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity), typeof(arch)}(arch, faces, face_connectivity)
end

#####
##### filling grid halos
#####

function grid_metric_halo(grid_metric, grid, location, topo, side)
    LX, LY = location
    TX, TY = topo
    side == :west  && return  underlying_west_halo(grid_metric, grid, LX, TX)
    side == :east  && return  underlying_east_halo(grid_metric, grid, LX, TX)
    side == :south && return underlying_south_halo(grid_metric, grid, LY, TY)
    side == :north && return underlying_north_halo(grid_metric, grid, LY, TY)
end

function grid_metric_boundary(grid_metric, grid, location, topo, side)
    LX, LY = location
    TX, TY = topo
    side == :west  && return  underlying_west_boundary(grid_metric, grid, LX, TX)
    side == :east  && return  underlying_east_boundary(grid_metric, grid, LX, TX)
    side == :south && return underlying_south_boundary(grid_metric, grid, LY, TY)
    side == :north && return underlying_north_boundary(grid_metric, grid, LY, TY)
end

function fill_grid_metric_halos!(grid)

    topo_bb = (Bounded, Bounded)

    loc_cc = (Center, Center)
    loc_cf = (Center, Face  )
    loc_fc = (Face,   Center)
    loc_ff = (Face,   Face  )

    for face_number in 1:6, side in (:west, :east, :south, :north)

        connectivity_info = getproperty(grid.face_connectivity[face_number], side)
        src_face_number = connectivity_info.face
        src_side = connectivity_info.side

        grid_face = grid.faces[face_number]
        src_grid_face = grid.faces[src_face_number]

        if sides_in_the_same_dimension(side, src_side)
            grid_metric_halo(grid_face.Δxᶜᶜᵃ, grid_face, loc_cc, topo_bb, side) .= grid_metric_boundary(grid_face.Δxᶜᶜᵃ, src_grid_face, loc_cc, topo_bb, src_side)
            grid_metric_halo(grid_face.Δyᶜᶜᵃ, grid_face, loc_cc, topo_bb, side) .= grid_metric_boundary(grid_face.Δyᶜᶜᵃ, src_grid_face, loc_cc, topo_bb, src_side)
            grid_metric_halo(grid_face.Azᶜᶜᵃ, grid_face, loc_cc, topo_bb, side) .= grid_metric_boundary(grid_face.Azᶜᶜᵃ, src_grid_face, loc_cc, topo_bb, src_side)

            grid_metric_halo(grid_face.Δxᶜᶠᵃ, grid_face, loc_cf, topo_bb, side) .= grid_metric_boundary(grid_face.Δxᶜᶠᵃ, src_grid_face, loc_cf, topo_bb, src_side)
            grid_metric_halo(grid_face.Δyᶜᶠᵃ, grid_face, loc_cf, topo_bb, side) .= grid_metric_boundary(grid_face.Δyᶜᶠᵃ, src_grid_face, loc_cf, topo_bb, src_side)
            grid_metric_halo(grid_face.Azᶜᶠᵃ, grid_face, loc_cf, topo_bb, side) .= grid_metric_boundary(grid_face.Azᶜᶠᵃ, src_grid_face, loc_cf, topo_bb, src_side)

            grid_metric_halo(grid_face.Δxᶠᶜᵃ, grid_face, loc_fc, topo_bb, side) .= grid_metric_boundary(grid_face.Δxᶠᶜᵃ, src_grid_face, loc_fc, topo_bb, src_side)
            grid_metric_halo(grid_face.Δyᶠᶜᵃ, grid_face, loc_fc, topo_bb, side) .= grid_metric_boundary(grid_face.Δyᶠᶜᵃ, src_grid_face, loc_fc, topo_bb, src_side)
            grid_metric_halo(grid_face.Azᶠᶜᵃ, grid_face, loc_fc, topo_bb, side) .= grid_metric_boundary(grid_face.Azᶠᶜᵃ, src_grid_face, loc_fc, topo_bb, src_side)

            grid_metric_halo(grid_face.Δxᶠᶠᵃ, grid_face, loc_ff, topo_bb, side) .= grid_metric_boundary(grid_face.Δxᶠᶠᵃ, src_grid_face, loc_ff, topo_bb, src_side)
            grid_metric_halo(grid_face.Δyᶠᶠᵃ, grid_face, loc_ff, topo_bb, side) .= grid_metric_boundary(grid_face.Δyᶠᶠᵃ, src_grid_face, loc_ff, topo_bb, src_side)
            grid_metric_halo(grid_face.Azᶠᶠᵃ, grid_face, loc_ff, topo_bb, side) .= grid_metric_boundary(grid_face.Azᶠᶠᵃ, src_grid_face, loc_ff, topo_bb, src_side)
        else
            reverse_dim = src_side in (:west, :east) ? 1 : 2
            grid_metric_halo(grid_face.Δxᶜᶜᵃ, grid_face, loc_cc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δyᶜᶜᵃ, src_grid_face, loc_cc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Δyᶜᶜᵃ, grid_face, loc_cc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δxᶜᶜᵃ, src_grid_face, loc_cc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Azᶜᶜᵃ, grid_face, loc_cc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Azᶜᶜᵃ, src_grid_face, loc_cc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)

            grid_metric_halo(grid_face.Δxᶜᶠᵃ, grid_face, loc_cf, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δyᶠᶜᵃ, src_grid_face, loc_fc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Δyᶜᶠᵃ, grid_face, loc_cf, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δxᶠᶜᵃ, src_grid_face, loc_fc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Azᶜᶠᵃ, grid_face, loc_cf, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Azᶠᶜᵃ, src_grid_face, loc_fc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)

            grid_metric_halo(grid_face.Δxᶠᶜᵃ, grid_face, loc_fc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δyᶜᶠᵃ, src_grid_face, loc_cf, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Δyᶠᶜᵃ, grid_face, loc_fc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δxᶜᶠᵃ, src_grid_face, loc_cf, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Azᶠᶜᵃ, grid_face, loc_fc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Azᶜᶠᵃ, src_grid_face, loc_cf, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)

            grid_metric_halo(grid_face.Δxᶠᶠᵃ, grid_face, loc_ff, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δyᶠᶠᵃ, src_grid_face, loc_ff, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Δyᶠᶠᵃ, grid_face, loc_ff, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δxᶠᶠᵃ, src_grid_face, loc_ff, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Azᶠᶠᵃ, grid_face, loc_ff, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Azᶠᶠᵃ, src_grid_face, loc_ff, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
        end
    end

    return nothing
end
