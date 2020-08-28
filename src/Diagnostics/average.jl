using Oceananigans.Architectures
using Oceananigans.Grids: halo_size, total_size
using Oceananigans.BoundaryConditions
using Oceananigans.Utils

"""
    Average{F, R, D, P, I, T} <: AbstractDiagnostic

A diagnostic for computing the averages of a field along particular dimensions.
"""
mutable struct Average{F, R, D, P, I, T} <: AbstractDiagnostic
                 field :: F
                  dims :: D
                result :: P
    iteration_interval :: I
         time_interval :: T
              previous :: Float64
           return_type :: R
            with_halos :: Bool
end

function dims_to_result_size(field, dims, grid)
    field_size = total_size(parent(field))
    return Tuple(d in dims ? 1 : field_size[d] for d in 1:3)
end

"""
    Average(field; dims, iteration_interval=nothing, time_interval=nothing, return_type=Array)

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
function Average(field; dims, iteration_interval=nothing, time_interval=nothing, return_type=Array,
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

"""
    normalize_sum!(avg)

Normalize the sum by the number of grid points averaged over to get the average.
"""
function normalize_sum!(avg)
    N = size(avg.field.grid)
    avg.result ./= prod(N[d] for d in avg.dims)
    return nothing
end

"""
    run_diagnostic(model, avg::Average)

Compute the horizontal average of `avg.field` and store the result in `avg.result`.
"""
function run_diagnostic(model, avg::Average)
    zero_halo_regions!(parent(avg.field), avg.field.grid)
    sum!(avg.result, parent(avg.field))
    normalize_sum!(avg)
    return nothing
end

function (avg::Average)(model)
    run_diagnostic(model, avg)
    N, H = size(model.grid), halo_size(model.grid)
    result = avg.return_type(avg.result)
    if avg.with_halos
        return result
    else
        return result[(d in avg.dims ? Colon() : (1+H[d]:N[d]+H[d]) for d in 1:3)...]
    end
end
