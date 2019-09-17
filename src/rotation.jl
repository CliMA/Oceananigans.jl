abstract type AbstractRotation end

"""
    VerticalRotationAxis{T} <: AbstractRotation

A parameter object for constant rotation around a vertical axis.
"""
struct VerticalRotationAxis{T} <: AbstractRotation
    f :: T
end

"""
    VerticalRotationAxis([T=Float64;] f)

Returns a parameter object for constant rotation at the angular frequency
`2f`, and therefore with background vorticity `f`, around a vertical axis.

Also called `FPlane`, after the "f-plane" approximation for the local effect of 
Earth's rotation in a planar coordinate system tangent to the Earth's surface.
"""
function VerticalRotationAxis(T::DataType=Float64; f) 
    return VerticalRotationAxis{T}(f)
end

const FPlane = VerticalRotationAxis
