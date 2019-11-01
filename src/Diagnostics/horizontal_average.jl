using Oceananigans: architecture, zero_halo_regions!, tupleit, parenttuple

"""
    HorizontalAverage{F, R, P, I, Ω} <: AbstractDiagnostic

A diagnostic for computing horizontal average of a field.
"""
mutable struct HorizontalAverage{F, R, P, I, Ω, G} <: AbstractDiagnostic
          field :: F
         result :: P
      frequency :: Ω
       interval :: I
       previous :: Float64
    return_type :: R
           grid :: G
end

"""
    HorizontalAverage(model, field; frequency=nothing, interval=nothing, return_type=Array)

Construct a `HorizontalAverage` of `field`.

After the horizontal average is computed it will be stored in the `result` property.

The `HorizontalAverage` can be used as a callable object that computes and returns the
horizontal average.

A `frequency` or `interval` (or both) can be passed to indicate how often to run this
diagnostic if it is part of `model.diagnostics`. `frequency` is a number of iterations
while `interval` is a time interval in units of `model.clock.time`.

A `return_type` can be used to specify the type returned when the `HorizontalAverage` is
used as a callable object. The default `return_type=Array` is useful when running a GPU
model and you want to save the output to disk by passing it to an output writer.
"""
function HorizontalAverage(field; frequency=nothing, interval=nothing, return_type=Array)
    arch = architecture(field)
    result = zeros(arch, field.grid, 1, 1, field.grid.Tz)
    return HorizontalAverage(field, result, frequency, interval, 0.0, return_type, field.grid)
end

# Normalize a horizontal sum to get the horizontal average.
normalize_horizontal_sum!(havg, grid) = havg.result /= (grid.Nx * grid.Ny)

"""
    run_diagnostic(model, havg::HorizontalAverage{NTuple{1}})

Compute the horizontal average of `havg.field` and store the result in `havg.result`.
"""
function run_diagnostic(model, havg::HorizontalAverage)
    zero_halo_regions!(parent(havg.field), model.grid)
    sum!(havg.result, parent(havg.field))
    normalize_horizontal_sum!(havg, model.grid)
    return nothing
end

function (havg::HorizontalAverage{F, Nothing})(model) where F
    run_diagnostic(model, havg)
    return havg.result
end

function (havg::HorizontalAverage)(model)
    run_diagnostic(model, havg)
    return havg.return_type(havg.result)
end
