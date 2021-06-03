using Statistics
using Oceananigans.Architectures

using OffsetArrays: OffsetArray
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

import Base: getindex, size, show, minimum, maximum
import Statistics: mean

import Oceananigans.Fields: AbstractField, AbstractDataField, AbstractReducedField, Field, ReducedField, minimum, maximum, mean, location, short_show
import Oceananigans.Grids: new_data
import Oceananigans.BoundaryConditions: FieldBoundaryConditions

struct CubedSphereFaces{E, F}
    faces :: F
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

const NonImmersedCubedSphereAbstractReducedFaceField = AbstractReducedField{X, Y, Z, A, D, <:ConformalCubedSphereFaceGrid} where {X, Y, Z, A, D}
const ImmersedCubedSphereAbstractReducedFaceField = AbstractReducedField{X, Y, Z, A, D, <:ImmersedConformalCubedSphereFaceGrid} where {X, Y, Z, A, D}

const CubedSphereAbstractReducedFaceField = Union{NonImmersedCubedSphereAbstractReducedFaceField{X, Y, Z, A},
                                                     ImmersedCubedSphereAbstractReducedFaceField{X, Y, Z, A}} where {X, Y, Z, A}

# CubedSphereField

# Flavors of CubedSphereField
const CubedSphereField                     = Field{X, Y, Z, A, <:CubedSphereData} where {X, Y, Z, A}
const CubedSphereReducedField              = ReducedField{X, Y, Z, A, <:CubedSphereData} where {X, Y, Z, A}
const CubedSphereAbstractField             = AbstractField{X, Y, Z, A, <:ConformalCubedSphereGrid} where {X, Y, Z, A}
const CubedSphereAbstractDataField         = AbstractDataField{X, Y, Z, A, <:ConformalCubedSphereGrid} where {X, Y, Z, A}

const AbstractCubedSphereField{X, Y, Z, A} = Union{    CubedSphereAbstractField{X, Y, Z, A},
                                                   CubedSphereAbstractDataField{X, Y, Z, A},
                                                        CubedSphereReducedField{X, Y, Z, A},
                                                               CubedSphereField{X, Y, Z, A}} where {X, Y, Z, A}

#####
##### new data
#####

function new_data(FT, arch::AbstractCPUArchitecture, grid::ConformalCubedSphereGrid, (X, Y, Z))
    faces = Tuple(new_data(FT, arch, face_grid, (X, Y, Z)) for face_grid in grid.faces)
    return CubedSphereFaces{typeof(faces[1]), typeof(faces)}(faces)
end

function new_data(FT, arch::AbstractGPUArchitecture, grid::ConformalCubedSphereGrid, (X, Y, Z))
    faces = Tuple(new_data(FT, arch, face_grid, (X, Y, Z)) for face_grid in grid.faces)
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

@inline function get_face(field::CubedSphereField, face_index)
    X, Y, Z = location(field)

    # Should we define a new lower-level constructor for Field that doesn't call validate_field_data?
    return Field{X, Y, Z}(get_face(field.data, face_index),
                          field.architecture,
                          get_face(field.grid, face_index),
                          get_face(field.boundary_conditions, face_index))
end

@inline function get_face(reduced_field::CubedSphereReducedField, face_index)
    X, Y, Z = location(reduced_field)

    return ReducedField{X, Y, Z}(get_face(reduced_field.data, face_index),
                                 reduced_field.architecture,
                                 get_face(reduced_field.grid, face_index),
                                 reduced_field.dims,
                                 get_face(reduced_field.boundary_conditions, face_index))
end

faces(field::AbstractCubedSphereField) = Tuple(get_face(field, face_index) for face_index in 1:length(field.data.faces))

minimum(field::AbstractCubedSphereField; dims=:) = minimum(minimum(face_field; dims) for face_field in faces(field))
maximum(field::AbstractCubedSphereField; dims=:) = maximum(maximum(face_field; dims) for face_field in faces(field))
mean(field::AbstractCubedSphereField; dims=:) = mean(mean(face_field; dims) for face_field in faces(field))

minimum(f, field::AbstractCubedSphereField; dims=:) = minimum(minimum(f, face_field; dims) for face_field in faces(field))
maximum(f, field::AbstractCubedSphereField; dims=:) = maximum(maximum(f, face_field; dims) for face_field in faces(field))
mean(f, field::AbstractCubedSphereField; dims=:) = mean(mean(f, face_field; dims) for face_field in faces(field))

λnodes(field::CubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = λnodes(LX(), LY(), LZ(), field.grid)
φnodes(field::CubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = φnodes(LX(), LY(), LZ(), field.grid)
