module AbstractOperations

export ∂x, ∂y, ∂z, @at, @unary, @binary, @multiary
export Δx, Δy, Δz, Ax, Ay, Az, volume
export Average, Integral, KernelFunctionOperation
export UnaryOperation, Derivative, BinaryOperation, MultiaryOperation

using Base: @propagate_inbounds

import Adapt
using CUDA

using Oceananigans
using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BoundaryConditions
using Oceananigans.Fields

using Oceananigans.Operators: interpolation_operator
using Oceananigans.Architectures: device
using Oceananigans: AbstractModel

import Oceananigans.Architectures: architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Fields: data, compute_at!

#####
##### Basic functionality
#####

abstract type AbstractOperation{LX, LY, LZ, G, T} <: AbstractField{LX, LY, LZ, G, T, 3} end

const AF = AbstractField # used in unary_operations.jl, binary_operations.jl, etc

# We have no halos to fill
fill_halo_regions!(::AbstractOperation, args...; kwargs...) = nothing

architecture(a::AbstractOperation) = architecture(a.grid)

# AbstractOperation macros add their associated functions to this list
const operators = Set()

"""
    at(loc, abstract_operation)

Return `abstract_operation` relocated to `loc`ation.
"""
at(loc, f) = f # fallback

include("grid_validation.jl")
include("grid_metrics.jl")
include("metric_field_reductions.jl")
include("unary_operations.jl")
include("binary_operations.jl")
include("multiary_operations.jl")
include("derivatives.jl")
include("kernel_function_operation.jl")
include("computed_field.jl")
include("at.jl")
include("broadcasting_abstract_operations.jl")
include("show_abstract_operations.jl")

# Make some operators!

# Number crunching:
import Base: sqrt, sin, cos, exp, tanh, -, +, /, ^, *

@unary sqrt sin cos exp tanh
@unary -

@binary +
@binary -
@binary /
@binary ^

@multiary +

# Comparison operators
import Base: <, >, ==, <=, >=, max, min, &, |
@binary < > == <= >=

@multiary max min

and = Symbol("(&)")
eval(define_multiary_operator(and))
push!(operators, and)
push!(multiary_operators, and)

eval(define_binary_operator(and))
push!(operators, and)
push!(binary_operators, and)

or = Symbol("(|)")
eval(define_multiary_operator(or))
push!(operators, or)
push!(multiary_operators, or)

eval(define_binary_operator(or))
push!(operators, or)
push!(binary_operators, or)

# For unknown reasons, the operator definition macros @binary and @multiary fail to work
# properly for :*. We thus manually define :* for fields.
import Base: *

eval(define_binary_operator(:*))
push!(operators, :*)
push!(binary_operators, :*)

eval(define_multiary_operator(:*))
push!(operators, :*)
push!(multiary_operators, :*)

end # module

