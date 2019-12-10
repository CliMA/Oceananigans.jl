module AbstractOperations

export ∂x, ∂y, ∂z, @at, Computation, compute!, @unary, @binary, @multiary

import Base: identity

using Base: @propagate_inbounds

using Oceananigans: @hascuda

@hascuda using CUDAnative, CUDAdrv, CuArrays

using Oceananigans, Oceananigans.Grids, Adapt

using Oceananigans: AbstractModel, AbstractGrid, AbstractField, AbstractLocatedField, Face, Cell,
                    xnode, ynode, znode, location, show_location, short_show,
                    device, launch_config, architecture, zero_halo_regions!

import Oceananigans: data, architecture

using Oceananigans.Operators

using Oceananigans.Grids: show_domain

using Oceananigans.Diagnostics: HorizontalAverage, normalize_horizontal_sum!

import Oceananigans.Diagnostics: run_diagnostic

using GPUifyLoops: @launch, @loop

#####
##### Basic functionality
#####

"""
    AbstractOperation{X, Y, Z, G} <: AbstractLocatedField{X, Y, Z, Nothing, G} end

Represents an operation performed on grid of type `G` at locations `X`, `Y`, and `Z`.
"""
abstract type AbstractOperation{X, Y, Z, G} <: AbstractLocatedField{X, Y, Z, Nothing, G} end

const ALF = AbstractLocatedField

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
