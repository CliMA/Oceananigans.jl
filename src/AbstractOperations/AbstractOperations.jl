module AbstractOperations

export ∂x, ∂y, ∂z, @at, @unary, @binary, @multiary

using Base: @propagate_inbounds

import Adapt
using CUDA

using Oceananigans.Architectures: @hascuda

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
import Oceananigans.Fields: data, compute!

#####
##### Basic functionality
#####

"""
    AbstractOperation{X, Y, Z, G} <: AbstractField{X, Y, Z, Nothing, G}

Represents an operation performed on grid of type `G` at locations `X`, `Y`, and `Z`.
"""
abstract type AbstractOperation{X, Y, Z, G} <: AbstractField{X, Y, Z, Nothing, G} end

const AF = AbstractField

# We (informally) require that all field-like objects define `parent`:
Base.parent(op::AbstractOperation) = op

# AbstractOperation macros add their associated functions to this list
const operators = Set()

include("at.jl")
include("grid_validation.jl")

include("unary_operations.jl")
include("binary_operations.jl")
include("multiary_operations.jl")
include("derivatives.jl")

include("show_abstract_operations.jl")
include("averages_of_operations.jl")

# Make some operators!

# Some unaries:
import Base: sqrt, sin, cos, exp, tanh, -, +, /, ^, *

@unary sqrt sin cos exp tanh
@unary -

@binary +
@binary -
@binary /
@binary ^

@multiary +

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
