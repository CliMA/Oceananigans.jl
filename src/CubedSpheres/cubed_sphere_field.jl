using Oceananigans.Fields: AbstractField

import Base: show
import Oceananigans.Fields: Field, CenterField, XFaceField, YFaceField, ZFaceField, interior

struct ConformalCubedSphereField{X, Y, Z, A, G, F, B} <: AbstractField{X, Y, Z, A, G}
                   grid :: G
                  faces :: F
    boundary_conditions :: B
end

# We need `bcs` and `data` to match the function signature but will ignore them.

Field(X, Y, Z, arch, grid::ConformalCubedSphereGrid, ::Nothing, data) =
    Field(X, Y, Z, arch, grid, FieldBoundaryConditions(grid, (X, Y, Z)), data)

function Field(X, Y, Z, arch, grid::ConformalCubedSphereGrid, bcs, data)

    faces = Tuple(
        Field(X, Y, Z, arch, face_grid, inject_cubed_sphere_exchange_boundary_conditions(bcs, face_number, grid.face_connectivity))
        for (face_number, face_grid) in enumerate(grid.faces)
    )

    # This field needs BCs otherwise errors happen so I'll assume all faces have
    # the same boundary conditions. A very bad assumption...
    cubed_sphere_bcs = faces[1].boundary_conditions

    return ConformalCubedSphereField{X, Y, Z, typeof(arch), typeof(grid), typeof(faces), typeof(cubed_sphere_bcs)}(grid, faces, cubed_sphere_bcs)
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

#####
##### Utils
#####

interior(field::ConformalCubedSphereField) = cat(Tuple(interior(field_face) for field_face in field.faces)..., dims=4)

const ConformalCubedSphereFaceField{LX, LY, LZ, A} = AbstractField{LX, LY, LZ, A, <:ConformalCubedSphereFaceGrid}

λnodes(field::ConformalCubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = λnodes(LX(), LY(), LZ(), field.grid)
φnodes(field::ConformalCubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = φnodes(LX(), LY(), LZ(), field.grid)

λnodes(field::ConformalCubedSphereField{LX, LY, LZ}) where {LX, LY, LZ} = λnodes(LX(), LY(), LZ(), field.grid)
φnodes(field::ConformalCubedSphereField{LX, LY, LZ}) where {LX, LY, LZ} = φnodes(LX(), LY(), LZ(), field.grid)
