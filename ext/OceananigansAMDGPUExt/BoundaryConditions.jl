module BoundaryConditions

using ..Architectures: ROCArray, ROCmGPU
using Oceananigans.Architectures: CPU

import Oceananigans.BoundaryConditions: validate_boundary_condition_architecture

validate_boundary_condition_architecture(::ROCArray, ::ROCmGPU, bc, side) = nothing

validate_boundary_condition_architecture(::ROCArray, ::CPU, bc, side) =
    throw(ArgumentError("$side $bc must use `Array` rather than `ROCArray` on CPU architectures!"))

validate_boundary_condition_architecture(::Array, ::ROCmGPU, bc, side) =
    throw(ArgumentError("$side $bc must use `ROCArray` rather than `Array` on ROCmGPU architectures!"))

end # module
