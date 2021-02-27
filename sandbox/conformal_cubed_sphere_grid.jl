include("conformal_cubed_sphere_face_grid.jl")

struct ConformalCubedSphereGrid{FT, F, C}
                faces :: F
    face_connectivity :: C
end

function ConformalCubedSphereGrid(FT=Float64; face_size, z, radius=R_Earth)
    faces = ConformalCubedSphereFaceGrid[]

    # +z face (face 1)
    z⁺_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=nothing)
    push!(faces, z⁺_face_grid)

    # -z face (face 2)
    z⁻_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotX(π))
    push!(faces, z⁻_face_grid)

    # +x face (face 3)
    x⁺_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotX(π/2))
    push!(faces, x⁺_face_grid)

    # -x face (face 4)
    x⁻_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotX(-π/2))
    push!(faces, x⁻_face_grid)

    # +y face (face 5)
    y⁺_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotY(π/2))
    push!(faces, y⁺_face_grid)

    # -y face (face 6)
    y⁻_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotY(-π/2))
    push!(faces, y⁻_face_grid)

    # Construct face connectivity:
    #
    #     +---+
    #     | 1 |
    # +---+---+---+---+
    # | 2 | 3 | 4 | 5 |
    # +---+---+---+---+
    #     | 6 |
    #     +---+

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
    print(io, "ConformalCubedSphereGrid{$FT}: face size = ($Nx, $Ny, $Nz)")
end
