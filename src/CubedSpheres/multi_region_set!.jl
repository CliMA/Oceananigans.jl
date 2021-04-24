using Oceananigans.Architectures: AbstractCPUArchitecture
using Oceananigans.Fields: AbstractField, ReducedField

import Oceananigans.Fields: set!

const MultiRegionCPUField = MultiRegionAbstractField{X, Y, Z, <:AbstractCPUArchitecture} where {X, Y, Z}
const MultiRegionGPUField = MultiRegionAbstractField{X, Y, Z, <:AbstractGPUArchitecture} where {X, Y, Z}

function set!(u::MultiRegionCPUField , v::MultiRegionCPUField)
    for (u_face, v_face) in zip(regions(u), regions(v))
        parent(u) .= parent(v)
    end

    return nothing
end

set!(field::MultiRegionCPUField, f::Function) = [set_face_field!(regional_field, f) for regional_field in regions(field)]
set!(field::MultiRegionCPUField, f::Number)   = [set_face_field!(regional_field, f) for regional_field in regions(field)]
set!(field::MultiRegionGPUField, f::Function) = [set_face_field!(regional_field, f) for regional_field in regions(field)]
set!(field::MultiRegionGPUField, f::Number)   = [set_face_field!(regional_field, f) for regional_field in regions(field)]

set_face_field!(field, a) = set!(field, a)
