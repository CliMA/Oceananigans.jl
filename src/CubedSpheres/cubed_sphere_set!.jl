using Oceananigans.Architectures: CPU
using Oceananigans.Fields: AbstractField

import Oceananigans.Fields: set!

const CubedSphereCPUField = CubedSphereField{LX, LY, LZ, <:CPU} where {LX, LY, LZ}
const CubedSphereGPUField = CubedSphereField{LX, LY, LZ, <:GPU} where {LX, LY, LZ}

# We need to define the function once for CPU fields then again for GPU fields to avoid the method
# ambiguity with Fields.set!.

function set!(u::CubedSphereCPUField, v::CubedSphereCPUField)
    for (u_face, v_face) in zip(faces(u), faces(v))
        @. u_face.data.parent = v_face.data.parent
    end
    return nothing
end

function set!(u::CubedSphereGPUField, v::CubedSphereGPUField)
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
