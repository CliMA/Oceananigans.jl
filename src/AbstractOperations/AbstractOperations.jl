module AbstractOperations

export ∂x, ∂y, ∂z, @at, Computation, compute!, @unary, @binary, @multiary

import Base: identity
using Base: @propagate_inbounds

import Adapt
using GPUifyLoops: @launch, @loop

using Oceananigans.Architectures: @hascuda
@hascuda using CUDAnative, CUDAdrv, CuArrays

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Architectures
using Oceananigans.Fields
using Oceananigans.Operators
using Oceananigans.BoundaryConditions

using Oceananigans.Architectures: device
using Oceananigans.Models: AbstractModel
using Oceananigans.Diagnostics: HorizontalAverage, normalize_horizontal_sum!
using Oceananigans.Utils: launch_config

import Oceananigans.Architectures: architecture
import Oceananigans.Fields: data
import Oceananigans.Diagnostics: run_diagnostic

#####
##### Basic functionality
#####

"""
    AbstractOperation{X, Y, Z, G} <: AbstractField{X, Y, Z, Nothing, G}

Represents an operation performed on grid of type `G` at locations `X`, `Y`, and `Z`.
"""
abstract type AbstractOperation{X, Y, Z, G} <: AbstractField{X, Y, Z, Nothing, G} end

const AF = AbstractField

# We (informally) require that all field-like objects define `data` and `parent`:
data(op::AbstractOperation) = op
Base.parent(op::AbstractOperation) = op

# AbstractOperation macros add their associated functions to this list
const operators = Set()

include("function_fields.jl")
include("interpolation_utils.jl")
include("grid_validation.jl")

include("unary_operations.jl")
include("binary_operations.jl")
include("multiary_operations.jl")
include("derivatives.jl")

include("computations.jl")
include("show_abstract_operations.jl")

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
