using Oceananigans.Fields: AbstractField

include("conformal_cubed_sphere_face_field.jl")

struct ConformalCubedSphereField{X, Y, Z, F, G} <: AbstractField{X, Y, Z, F, G}
    faces :: F
     grid :: G
end

function ConformalCubedSphereField(FT::DataType, arch, grid::ConformalCubedSphereGrid, location)
    LX, LY, LZ = location
    faces = Tuple(ConformalCubedSphereFaceField(FT, arch, grid.faces[f], location) for f in 1:length(grid.faces))
    return ConformalCubedSphereField{LX, LY, LZ, typeof(faces), typeof(grid)}(faces, grid)
end

function show(io::IO, field::ConformalCubedSphereField{LX, LY, LZ}) where {LX, LY, LZ}
    n_faces = length(field.faces)
    face = field.faces[1]
    Nx, Ny, Nz = size(face.data) .- 2 .* halo_size(face.grid)
    Hx, Hy, Hz = halo_size(face.grid)
    print(io, "ConformalCubedSphereField{$LX, $LY, $LZ} with $n_faces faces of size ($Nx, $Ny, $Nz) + 2 × ($Hx, $Hy, $Hz)\n")
end
