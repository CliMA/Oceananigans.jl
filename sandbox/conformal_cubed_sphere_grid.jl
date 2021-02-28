include("conformal_cubed_sphere_face_grid.jl")

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

    # Construct face connectivity:
    #
    # +---+
    # | 1 |
    # +---+---+---+---+
    # | 2 | 3 | 4 | 5 |
    # +---+---+---+---+
    # | 6 |
    # +---+

    face_connectivity = (
        (right=4, left=2, above=5, below=3),
        (right=3, left=5, above=1, below=6),
        (right=4, left=2, above=1, below=6),
        (right=5, left=3, above=1, below=6),
        (right=2, left=4, above=1, below=6),
        (right=4, left=2, above=3, below=5)
    )

    return ConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity)}(faces, face_connectivity)
end

function show(io::IO, grid::ConformalCubedSphereGrid{FT}) where FT
    face = grid.faces[1]
    Nx, Ny, Nz = face.Nx, face.Ny, face.Nz
    print(io, "ConformalCubedSphereGrid{$FT}: $(length(grid.faces)) faces with size = ($Nx, $Ny, $Nz)")
end
