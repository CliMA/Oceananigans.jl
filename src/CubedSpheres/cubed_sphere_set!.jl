using Oceananigans.Architectures: CPU
using Oceananigans.Fields: AbstractField, ReducedField

import Oceananigans.Fields: set!

const CubedSphereCPUField = CubedSphereField{X, Y, Z, <:CPU} where {X, Y, Z}
const CubedSphereGPUField = CubedSphereField{X, Y, Z, <:GPU} where {X, Y, Z}

const CubedSphereCPUReducedField = CubedSphereReducedField{X, Y, Z, <:CPU} where {X, Y, Z}
const CubedSphereGPUReducedField = CubedSphereReducedField{X, Y, Z, <:GPU} where {X, Y, Z}

const CubedSphereCPUFields = Union{CubedSphereCPUField, CubedSphereCPUReducedField}
const CubedSphereGPUFields = Union{CubedSphereGPUField, CubedSphereGPUReducedField}

# We need to define the function once for CPU fields then again for GPU fields to avoid the method
# ambiguity with Fields.set!.

function set!(u::CubedSphereCPUFields , v::CubedSphereCPUFields)
    for (u_face, v_face) in zip(faces(u), faces(v))
        @. u_face.data.parent = v_face.data.parent
    end
    return nothing
end

function set!(u::CubedSphereGPUField , v::CubedSphereGPUField)
    for (u_face, v_face) in zip(faces(u), faces(v))
        @. u_face.data.parent = v_face.data.parent
    end
    return nothing
end

set!(field::CubedSphereCPUFields, f::Function) = [set_face_field!(field_face, f) for field_face in faces(field)]
set!(field::CubedSphereCPUFields, f::Number) = [set_face_field!(field_face, f) for field_face in faces(field)]
set!(field::CubedSphereGPUFields, f::Function) = [set_face_field!(field_face, f) for field_face in faces(field)]
set!(field::CubedSphereGPUFields, f::Number) = [set_face_field!(field_face, f) for field_face in faces(field)]

set_face_field!(field, a) = set!(field, a)
