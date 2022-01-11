using Statistics
using Oceananigans.Architectures

using OffsetArrays: OffsetArray
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

import Base: getindex, size, show, minimum, maximum
import Statistics: mean

import Oceananigans.Fields: AbstractField, Field, minimum, maximum, mean, location, short_show, KernelComputedField
import Oceananigans.Grids: new_data
import Oceananigans.BoundaryConditions: FieldBoundaryConditions

struct CubedSphereFaces{E, F}
    faces :: F
end

function CubedSphereFaces(faces::F) where F
    E = typeof(faces[1])
    return CubedSphereFaces{E, F}(faces)
end

@inline Base.getindex(f::CubedSphereFaces, i::Int) = @inbounds f.faces[i]

#####
##### Dispatch the world / insane type unions
#####

const CubedSphereData = CubedSphereFaces{<:OffsetArray}

# Some dispatch foo to make a type union for CubedSphereFaceField...
#
# Conformal cubed sphere grid wrapped in ImmersedBoundaryGrid:
const ImmersedConformalCubedSphereFaceGrid = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:ConformalCubedSphereFaceGrid} where {FT, TX, TY, TZ}

# CubedSphereFaceField:
const NonImmersedCubedSphereFaceField = AbstractField{X, Y, Z, A, <:ConformalCubedSphereFaceGrid} where {X, Y, Z, A}
const ImmersedCubedSphereFaceField    = AbstractField{X, Y, Z, A, <:ImmersedConformalCubedSphereFaceGrid} where {X, Y, Z, A}

const CubedSphereFaceField = Union{NonImmersedCubedSphereFaceField{X, Y, Z, A},
                                      ImmersedCubedSphereFaceField{X, Y, Z, A}} where {X, Y, Z, A}

# CubedSphereField

# Flavors of CubedSphereField
const CubedSphereField         = Field{X, Y, Z, O, A, G, T, <:CubedSphereData} where {X, Y, Z, O, A, G, T}
const CubedSphereAbstractField = AbstractField{X, Y, Z, A, <:ConformalCubedSphereGrid} where {X, Y, Z, A}

const AbstractCubedSphereField{X, Y, Z, A} = Union{CubedSphereAbstractField{X, Y, Z, A},
                                                           CubedSphereField{X, Y, Z, A}} where {X, Y, Z, A}

#####
##### new data
#####

function new_data(FT, grid::ConformalCubedSphereGrid, (X, Y, Z))
    faces = Tuple(new_data(FT, face_grid, (X, Y, Z)) for face_grid in grid.faces)
    return CubedSphereFaces{typeof(faces[1]), typeof(faces)}(faces)
end

#####
##### FieldBoundaryConditions
#####

function FieldBoundaryConditions(grid::ConformalCubedSphereGrid, (X, Y, Z); user_defined_bcs...)

    faces = Tuple(
        inject_cubed_sphere_exchange_boundary_conditions(
            FieldBoundaryConditions(face_grid, (X, Y, Z); user_defined_bcs...),
            face_index,
            grid.face_connectivity
        )
        for (face_index, face_grid) in enumerate(grid.faces)
    )

    return CubedSphereFaces{typeof(faces[1]), typeof(faces)}(faces)
end

#####
##### Utils
#####

function Base.show(io::IO, field::AbstractCubedSphereField)
    LX, LY, LZ = location(field)
    arch = architecture(field)
    A = typeof(arch)
    return print(io, "$(typeof(field).name.wrapper) at ($LX, $LY, $LZ)\n",
          "├── architecture: $A\n",
          "└── grid: $(short_show(field.grid))")
end

@inline function interior(field::AbstractCubedSphereField)
    faces = Tuple(interior(face_field_face) for face_field in faces(field))
    return CubedSphereFaces(faces)
end

Base.size(data::CubedSphereData) = (size(data.faces[1])..., length(data.faces))

@inline get_face(field::CubedSphereField, face_index) =
    Field(location(field), get_face(field.grid, face_index);
          data = get_face(field.data, face_index),
          boundary_conditions = get_face(field.boundary_conditions, face_index))
    
@inline get_face(field::CubedSphereAbstractField, face_index) =
    Field(location(field), get_face(field.grid, face_index);
          data = get_face(field.data, face_index),
          boundary_conditions = get_face(field.boundary_conditions, face_index))

faces(field::AbstractCubedSphereField) = Tuple(get_face(field, face_index) for face_index in 1:length(field.data.faces))

minimum(field::AbstractCubedSphereField; dims=:) = minimum(minimum(face_field; dims) for face_field in faces(field))
maximum(field::AbstractCubedSphereField; dims=:) = maximum(maximum(face_field; dims) for face_field in faces(field))
mean(field::AbstractCubedSphereField; dims=:) = mean(mean(face_field; dims) for face_field in faces(field))

minimum(f::Function, field::AbstractCubedSphereField; dims=:) = minimum(minimum(f, face_field; dims) for face_field in faces(field))
maximum(f::Function, field::AbstractCubedSphereField; dims=:) = maximum(maximum(f, face_field; dims) for face_field in faces(field))
mean(f::Function, field::AbstractCubedSphereField; dims=:) = mean(mean(f, face_field; dims) for face_field in faces(field))

λnodes(field::CubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = λnodes(LX(), LY(), LZ(), field.grid)
φnodes(field::CubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = φnodes(LX(), LY(), LZ(), field.grid)
