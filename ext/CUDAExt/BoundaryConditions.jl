module BoundaryConditions
    using CUDA
    using Oceananigans

    using Oceananigans.Architectures
    import Oceananigans.BoundaryConditions: 
        validate_boundary_condition_architecture

    validate_boundary_condition_architecture(::CuArray, ::GPU, bc, side) = nothing

    validate_boundary_condition_architecture(::CuArray, ::CPU, bc, side) =
        throw(ArgumentError("$side $bc must use `Array` rather than `CuArray` on CPU architectures!"))

    validate_boundary_condition_architecture(::Array, ::GPU, bc, side) =
        throw(ArgumentError("$side $bc must use `CuArray` rather than `Array` on GPU architectures!"))
end
