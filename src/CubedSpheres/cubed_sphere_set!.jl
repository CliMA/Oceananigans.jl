using Oceananigans.Architectures: AbstractCPUArchitecture
using Oceananigans.Fields: AbstractField, ReducedField

import Oceananigans.Fields: set!

const CubedSphereCPUField = CubedSphereField{X, Y, Z, <:AbstractCPUArchitecture} where {X, Y, Z}
const CubedSphereGPUField = CubedSphereField{X, Y, Z, <:AbstractGPUArchitecture} where {X, Y, Z}

function set!(u::CubedSphereCPUField , v::CubedSphereCPUField)
    for (u_face, v_face) in zip(faces(u), faces(v))
        @. u_face.data.parent = v_face.data.parent
    end
    return nothing
end

set!(field::CubedSphereCPUField, f::Function) = [set_face_field!(field_face, f) for field_face in faces(field)]
set!(field::CubedSphereCPUField, f::Number) = [set_face_field!(field_face, f) for field_face in faces(field)]
set!(field::CubedSphereGPUField, f::Function) = [set_face_field!(field_face, f) for field_face in faces(field)]
set!(field::CubedSphereGPUField, f::Number) = [set_face_field!(field_face, f) for field_face in faces(field)]

set_face_field!(field, a) = set!(field, a)

