using KernelAbstractions
using Oceananigans.Utils: work_layout

import Oceananigans.Fields: location, total_size

"""
    Computation{T, R, O, G}

Represents an operation performed over the elements of a field.
"""
struct Computation{T, R, O, G}
      operation :: O
         result :: R
           grid :: G
    return_type :: T
end

"""
    Computation(operation, result; return_type=Array)

Returns a `Computation` representing an `operation` performed over the elements of
`operation.grid` and stored in `result`. `return_type` specifies the output type when the
`Computation` instances is called as a function.
"""
Computation(operation, result; return_type=Array) =
    Computation(operation, result, operation.grid, return_type)

# Utilities for limited field-like behavior
architecture(comp::Computation) = architecture(comp.result)
Base.parent(comp::Computation) = comp # this enables taking a "horizontal average" of a computation
location(comp::Computation) = location(comp.operation)
total_size(comp::Computation) = total_size(comp.result)

"""
    compute!(computation::Computation)

Perform a `computation`. The result is stored in `computation.result`.
"""
function compute!(computation::Computation)
    arch = architecture(computation.result)
    result_data = data(computation.result)

    workgroup, worksize = work_layout(computation.grid, :xyz)

    compute_kernel! = _compute!(device(arch), workgroup, worksize)

    event = compute_kernel!(result_data, computation.grid, computation.operation; dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end

"""Compute an `operation` over `grid` and store in `result`."""
@kernel function _compute!(result, grid, operation)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = operation[i, j, k]
end

"""
    (computation::Computation)(args...)

Performs the `compute(computation)` and returns the result if `isnothing(return_type)`,
or the result after being converted to `return_type`.
"""
function (computation::Computation)(args...)
    compute!(computation)
    return computation.return_type(interior(computation.result))
end

function (computation::Computation{<:Nothing})(args...)
    compute!(computation)
    return computation.result
end

#####
##### Functionality for using computations with HorizontalAverage
#####

"""
    HorizontalAverage(op::AbstractOperation, result; kwargs...)

Returns the representation of a `HorizontalAverage` over the operation `op`, using
`result` as a temporary array to store the result of `operation` computed on `op.grid`.
"""
function HorizontalAverage(op::AbstractOperation, result; kwargs...)
    computation = Computation(op, result)
    return HorizontalAverage(computation; kwargs...)
end

"""
    HorizontalAverage(op::AbstractOperation, model; kwargs...)

Returns the representation of a `HorizontalAverage` over the operation `op`, using
`model.pressures.pHY′` as a temporary array to store the result of `operation` computed on
`op.grid`.
"""
HorizontalAverage(op::AbstractOperation, model::AbstractModel; kwargs...) =
    HorizontalAverage(op, model.pressures.pHY′; kwargs...)

"""Compute the horizontal average of a computation."""
function run_diagnostic(model, havg::HorizontalAverage{<:Computation})
    compute!(havg.field)
    zero_halo_regions!(parent(havg.field.result), model.grid)
    sum!(havg.result, parent(havg.field.result))
    normalize_horizontal_sum!(havg, model.grid)
    return nothing
end
