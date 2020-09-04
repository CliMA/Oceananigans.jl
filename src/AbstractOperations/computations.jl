using KernelAbstractions
using Oceananigans.Utils: work_layout
import Oceananigans.Diagnostics: get_kernel

import Oceananigans.Fields: location, total_size
using Oceananigans.Diagnostics: dims_to_result_size

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
    loc = location(computation.result)

    workgroup, worksize = work_layout(computation.grid, :xyz, include_right_boundaries=true, location=loc)

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
##### Functionality for using computations with Average
#####

"""
    Average(op::AbstractOperation, result; dims, kwargs...)

Returns the representation of an `Average` over the operation `op`, using `result` as
a temporary array to store the result of `operation` computed on `op.grid`.
"""
function Average(op::AbstractOperation, result; dims, kwargs...)
    computation = Computation(op, result)
    return Average(computation; dims=dims, kwargs...)
end

"""
    Average(op::AbstractOperation, model; dims, kwargs...)

Returns the representation of an `Average` over the operation `op`, using
`model.pressures.pHY′` as a temporary array to store the result of `operation` computed on
`op.grid`.
"""
Average(op::AbstractOperation, model::AbstractModel; dims, kwargs...) =
    Average(op, model.pressures.pHY′; dims=dims, kwargs...)

"""
    Average(field::Field; dims, iteration_interval=nothing, time_interval=nothing, return_type=Array)

Construct an `Average` of `field` along the dimensions specified by the tuple `dims`.

After the average is computed it will be stored in the `result` property.

The `Average` can be used as a callable object that computes and returns the average.

An `iteration_interval` or `time_interval` (or both) can be passed to indicate how often to
run this diagnostic if it is part of `simulation.diagnostics`. `iteration_interval` is a
number of iterations while `time_interval` is a time interval in units of `model.clock.time`.

A `return_type` can be used to specify the type returned when the `Average` is
used as a callable object. The default `return_type=Array` is useful when running a GPU
model and you want to save the output to disk by passing it to an output writer.
"""
function Average(field::Computation; dims, iteration_interval=nothing, time_interval=nothing, return_type=Array,
                 with_halos=true)

    dims isa Union{Int, Tuple} || error("Average dims must be an integer or tuple!")
    dims isa Int && (dims = tuple(dims))

    length(dims) == 0 && error("dims is empty! Must average over at least one dimension.")
    length(dims) > 3  && error("Models are 3-dimensional. Cannot average over 4+ dimensions.")
    all(1 <= d <= 3 for d in dims) || error("Dimensions must be one of 1, 2, 3.")

    arch = architecture(field)
    result_size = dims_to_result_size(field, dims, field.grid)
    result = zeros(arch, field.grid, result_size...)

    return Average(field, dims, result, iteration_interval, time_interval, 0.0,
                   return_type, with_halos)
end

"""Compute the average of a computation."""
function run_diagnostic(model, avg::Average{<:Computation})
    compute!(avg.field)
    zero_halo_regions!(parent(avg.field.result), model.grid)
    sum!(avg.result, parent(avg.field.result))
    normalize_sum!(avg)
    return nothing
end

#####
##### Functionality for using computations with WindowedTimeAverage
#####

function get_kernel(computation::Computation)
    compute!(computation)
    return parent(computation.result)
end
