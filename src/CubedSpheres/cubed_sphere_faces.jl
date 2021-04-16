using Statistics
using Oceananigans.Architectures

using OffsetArrays: OffsetArray

import Base: getindex, size, show, minimum, maximum
import Statistics: mean

import Oceananigans.Fields: AbstractField, ReducedField, Field, new_data
import Oceananigans.BoundaryConditions: FieldBoundaryConditions

struct CubedSphereFaces{E, F}
    faces :: F
end

const CubedSphereData = CubedSphereFaces{<:OffsetArray}

const CubedSphereField = AbstractField{X, Y, Z, A, <:ConformalCubedSphereGrid} where {X, Y, Z, A}
const CubedSphereFaceField = AbstractField{X, Y, Z, A, <:ConformalCubedSphereFaceGrid} where {X, Y, Z, A}
const CubedSphereReducedField = ReducedField{X, Y, Z, A, D, <:ConformalCubedSphereFaceGrid} where {X, Y, Z, A, D}

# There must be a way to dispatch in one function without ambiguity with `new_data.jl`...

function new_data(FT, arch::AbstractCPUArchitecture, grid::ConformalCubedSphereGrid, (X, Y, Z))
    faces = Tuple(new_data(FT, arch, face_grid, (X, Y, Z)) for face_grid in grid.faces)
    return CubedSphereFaces{typeof(faces[1]), typeof(faces)}(faces)
end

function new_data(FT, arch::AbstractGPUArchitecture, grid::ConformalCubedSphereGrid, (X, Y, Z))
    faces = Tuple(new_data(FT, arch, face_grid, (X, Y, Z)) for face_grid in grid.faces)
    return CubedSphereFaces{typeof(faces[1]), typeof(faces)}(faces)
end

function FieldBoundaryConditions(grid::ConformalCubedSphereGrid, (X, Y, Z); user_defined_bcs...)

    faces = Tuple(
        inject_cubed_sphere_exchange_boundary_conditions(
            FieldBoundaryConditions(face_grid, (X, Y, Z); user_defined_bcs...),
            face_number,
            grid.face_connectivity
        )
        for (face_number, face_grid) in enumerate(grid.faces)
    )

    return CubedSphereFaces{typeof(faces[1]), typeof(faces)}(faces)
end

#####
##### Utils
#####

Base.size(data::CubedSphereData) = (size(data.faces[1])..., length(data.faces))

face(field::CubedSphereField{X, Y, Z}, face_number) where {X, Y, Z} =
    Field(X, Y, Z, field.architecture, field.grid.faces[face_number], field.boundary_conditions.faces[face_number], field.data.faces[face_number])

faces(field::CubedSphereField) = Tuple(face(field, face_number) for face_number in 1:length(field.data.faces))

# Base.minimum(field::CubedSphereField; dims=:) = minimum(minimum(face; dims) for field_face in field.data.faces)
# Base.maximum(field::CubedSphereField; dims=:) = maximum(maximum(field_face; dims) for field_face in field.data.faces)
# Statistics.mean(field::CubedSphereField; dims=:) = mean(mean(field_face; dims) for field_face in field.data.faces)

# Base.minimum(f, field::CubedSphereField; dims=:) = minimum(minimum(f, field_face; dims) for field_face in field.data.faces)
# Base.maximum(f, field::CubedSphereField; dims=:) = maximum(maximum(f, field_face; dims) for field_face in field.data.faces)
# Statistics.mean(f, field::CubedSphereField; dims=:) = mean(mean(f, field_face; dims) for field_face in field.data.faces)

# interior(field::AbstractCubedSphereField) = cat(Tuple(interior(field_face) for field_face in field.data.faces)..., dims=4)

# λnodes(field::CubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = λnodes(LX(), LY(), LZ(), field.grid)
# φnodes(field::CubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = φnodes(LX(), LY(), LZ(), field.grid)

# λnodes(field::CubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = λnodes(LX(), LY(), LZ(), field.grid)
# φnodes(field::CubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = φnodes(LX(), LY(), LZ(), field.grid)
