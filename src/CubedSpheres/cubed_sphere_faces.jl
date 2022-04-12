using Statistics
using CUDA
using Oceananigans.Architectures
using Oceananigans.AbstractOperations: AbstractOperation

using OffsetArrays: OffsetArray
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

import Base: getindex, size, show, minimum, maximum
import Statistics: mean

import Oceananigans.Fields: AbstractField, Field, FieldBoundaryBuffers, minimum, maximum, mean, location, set!
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
const NonImmersedCubedSphereFaceField = AbstractField{LX, LY, LZ, <:ConformalCubedSphereFaceGrid} where {LX, LY, LZ}
const ImmersedCubedSphereFaceField    = AbstractField{LX, LY, LZ, <:ImmersedConformalCubedSphereFaceGrid} where {LX, LY, LZ}

const CubedSphereFaceField = Union{NonImmersedCubedSphereFaceField{LX, LY, LZ},
                                      ImmersedCubedSphereFaceField{LX, LY, LZ}} where {LX, LY, LZ}

# CubedSphereField

# Flavors of CubedSphereField
const CubedSphereField{LX, LY, LZ} =
    Union{Field{LX, LY, LZ, <:Nothing, <:ConformalCubedSphereGrid},
          Field{LX, LY, LZ, <:AbstractOperation, <:ConformalCubedSphereGrid}}

const CubedSphereAbstractField{LX, LY, LZ} = AbstractField{LX, LY, LZ, <:ConformalCubedSphereGrid}

const AbstractCubedSphereField{LX, LY, LZ} =
    Union{CubedSphereAbstractField{LX, LY, LZ},
                  CubedSphereField{LX, LY, LZ}}

#####
##### new data
#####

function new_data(FT::DataType, grid::ConformalCubedSphereGrid, loc, indices)
    faces = Tuple(new_data(FT, face_grid, loc, indices) for face_grid in grid.faces)
    return CubedSphereFaces(faces)
end

#####
##### FieldBoundaryConditions
#####

function FieldBoundaryConditions(grid::ConformalCubedSphereGrid, loc, indices; user_defined_bcs...)

    faces = Tuple(
        inject_cubed_sphere_exchange_boundary_conditions(
            FieldBoundaryConditions(face_grid, loc, indices; user_defined_bcs...),
            face_index,
            grid.face_connectivity
        )
        for (face_index, face_grid) in enumerate(grid.faces)
    )

    return CubedSphereFaces(faces)
end

#####
##### FieldBoundaryBuffers
#####

FieldBoundaryBuffers(grid::ConformalCubedSphereGrid, args...) = FieldBoundaryBuffers()

#####
##### Utils
#####

function Base.show(io::IO, field::Union{CubedSphereField, AbstractCubedSphereField})
    LX, LY, LZ = location(field)
    arch = architecture(field)
    A = typeof(arch)
    return print(io, "$(typeof(field).name.wrapper) at ($LX, $LY, $LZ)\n",
          "├── architecture: $A\n",
          "└── grid: $(summary(field.grid))")
end

@inline function interior(field::AbstractCubedSphereField)
    faces = Tuple(interior(face_field_face) for face_field in faces(field))
    return CubedSphereFaces(faces)
end

Base.size(field::CubedSphereField) = size(field.data)
Base.size(data::CubedSphereData) = (size(data.faces[1])..., length(data.faces))

@inline get_face(field::CubedSphereField, face_index) =
    Field(location(field),
          get_face(field.grid, face_index),
          get_face(field.data, face_index),
          get_face(field.boundary_conditions, face_index),
          field.indices,
          get_face(field.operand, face_index),
          nothing)
    
faces(field::AbstractCubedSphereField) = Tuple(get_face(field, face_index) for face_index in 1:length(field.data.faces))

function Base.fill!(csf::CubedSphereField, val)
    for field in faces(csf)
        fill!(field, val)
    end
    return csf
end

#####
##### set!
#####

function cubed_sphere_set!(u, v)
    for face = 1:length(u.grid.faces)
        set!(get_face(u, face), get_face(v, face))
    end
    return nothing
end

# Resolve ambiguities
set!(u::CubedSphereField, v) = cubed_sphere_set!(u, v)
set!(u::CubedSphereField, v::Function) = cubed_sphere_set!(u, v)
set!(u::CubedSphereField, v::Union{Array, CuArray, OffsetArray}) = cubed_sphere_set!(u, v)
set!(u::CubedSphereField, v::CubedSphereField) = cubed_sphere_set!(u, v)

#####
##### Random utils
#####

minimum(field::AbstractCubedSphereField; dims=:) = minimum(minimum(face_field; dims) for face_field in faces(field))
maximum(field::AbstractCubedSphereField; dims=:) = maximum(maximum(face_field; dims) for face_field in faces(field))
mean(field::AbstractCubedSphereField; dims=:) = mean(mean(face_field; dims) for face_field in faces(field))

minimum(f::Function, field::AbstractCubedSphereField; dims=:) = minimum(minimum(f, face_field; dims) for face_field in faces(field))
maximum(f::Function, field::AbstractCubedSphereField; dims=:) = maximum(maximum(f, face_field; dims) for face_field in faces(field))
mean(f::Function, field::AbstractCubedSphereField; dims=:) = mean(mean(f, face_field; dims) for face_field in faces(field))

λnodes(field::CubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = λnodes(LX(), LY(), LZ(), field.grid)
φnodes(field::CubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = φnodes(LX(), LY(), LZ(), field.grid)
