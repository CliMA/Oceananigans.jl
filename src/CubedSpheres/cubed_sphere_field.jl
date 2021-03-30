using Oceananigans.Fields: AbstractField

import Base: show
import Oceananigans.Fields: Field, CenterField, XFaceField, YFaceField, ZFaceField

struct ConformalCubedSphereField{X, Y, Z, A, G, F, B} <: AbstractField{X, Y, Z, A, G}
                  faces :: F
    boundary_conditions :: B
end

# We need `bcs` and `data` to match the function signature but will ignore them.

function Field(X, Y, Z, arch, grid::ConformalCubedSphereGrid, bcs::Nothing, data)
    faces = Tuple(Field(X, Y, Z, arch, face_grid) for face_grid in grid.faces)

    # Assume all faces have the same boundary conditions.
    # Probably a very bad assumption.
    cubed_sphere_bcs = faces[1].boundary_conditions

    return ConformalCubedSphereField{X, Y, Z, typeof(arch), typeof(grid), typeof(faces), typeof(cubed_sphere_bcs)}(faces, cubed_sphere_bcs)
end

function Field(X, Y, Z, arch, grid::ConformalCubedSphereGrid, bcs, data)
    faces = Tuple(Field(X, Y, Z, arch, face_grid, bcs) for face_grid in grid.faces)

    # Assume all faces have the same boundary conditions.
    # Probably a very bad assumption.
    cubed_sphere_bcs = faces[1].boundary_conditions

    return ConformalCubedSphereField{X, Y, Z, typeof(arch), typeof(grid), typeof(faces), typeof(cubed_sphere_bcs)}(faces, cubed_sphere_bcs)
end

# Gotta short-circuit for `::ConformalCubedSphereGrid` because these
# Field constructors allocate `data` in the function signature.

CenterField(FT::DataType, arch, grid::ConformalCubedSphereGrid, bcs=nothing, data=nothing) = Field(Center, Center, Center, arch, grid, bcs, data)
 XFaceField(FT::DataType, arch, grid::ConformalCubedSphereGrid, bcs=nothing, data=nothing) = Field(Face,   Center, Center, arch, grid, bcs, data)
 YFaceField(FT::DataType, arch, grid::ConformalCubedSphereGrid, bcs=nothing, data=nothing) = Field(Center, Face,   Center, arch, grid, bcs, data)
 ZFaceField(FT::DataType, arch, grid::ConformalCubedSphereGrid, bcs=nothing, data=nothing) = Field(Center, Center, Face,   arch, grid, bcs, data)

function Base.show(io::IO, field::ConformalCubedSphereField{X, Y, Z}) where {X, Y, Z}
    face = field.faces[1]
    Nx, Ny, Nz = size(face.grid)
    print(io, "ConformalCubedSphereField{$X, $Y, $Z}: $(length(field.faces)) faces with size = ($Nx, $Ny, $Nz)")
end
