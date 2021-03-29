using Rotations

using Oceananigans.Grids
using Oceananigans.Grids: R_Earth

import Base: show

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

    face1_connectivity = (west=(5, :north), east=(2, :west),  south=(6, :north), north=(3, :west))
    face2_connectivity = (west=(1, :east),  east=(4, :south), south=(6, :east),  north=(3, :south))
    face3_connectivity = (west=(1, :north), east=(4, :west),  south=(2, :north), north=(5, :west))
    face4_connectivity = (west=(3, :east),  east=(6, :south), south=(2, :east),  north=(5, :south))
    face5_connectivity = (west=(3, :north), east=(6, :west),  south=(4, :north), north=(1, :west))
    face6_connectivity = (west=(5, :east),  east=(2, :south), south=(4, :east),  north=(1, :south))

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

    face_topo = (Periodic, Periodic, Bounded) # assuming all faces are connected
    face_kwargs = (Nz=Nz, z=z, topology=face_topo, radius=radius, halo=halo)

    faces = [ConformalCubedSphereFaceGrid(filepath, FT; face=n, face_kwargs...) for n in 1:6]

    face_connectivity = default_face_connectivity()

    return ConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity)}(faces, face_connectivity)
end

function Base.show(io::IO, grid::ConformalCubedSphereGrid{FT}) where FT
    face = grid.faces[1]
    Nx, Ny, Nz = face.Nx, face.Ny, face.Nz
    print(io, "ConformalCubedSphereGrid{$FT}: $(length(grid.faces)) faces with size = ($Nx, $Ny, $Nz)")
end
