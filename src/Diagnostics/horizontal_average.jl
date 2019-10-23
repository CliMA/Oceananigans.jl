"""
    HorizontalAverage{F, R, P, I, Ω} <: AbstractDiagnostic

A diagnostic for computing horizontal average of a field or the product of multiple fields.
"""
mutable struct HorizontalAverage{F, R, P, I, Ω} <: AbstractDiagnostic
        profile :: P
         fields :: F
      frequency :: Ω
       interval :: I
       previous :: Float64
    return_type :: R
end

"""
    HorizontalAverage(model, fields; frequency=nothing, interval=nothing, return_type=Array)

Construct a `HorizontalAverage` diagnostic for `model`.

After the horizontal average is computed it will be stored in the `profile` property.

The `HorizontalAverage` can be used as a callable object that computes and returns the
horizontal average.

If a single field is passed to `fields` the the horizontal average of that single field
will be computed. If multiple fields are passed to `fields`, then the horizontal average
of their product will be computed.

A `frequency` or `interval` (or both) can be passed to indicate how often to run this
diagnostic if it is part of `model.diagnostics`. `frequency` is a number of iterations
while `interval` is a time interval in units of `model.clock.time`.

A `return_type` can be used to specify the type returned when the `HorizontalAverage` is
used as a callable object. The default `return_type=Array` is useful when running a GPU
model and you want to save the output to disk by passing it to an output writer.

Warning
=======
Right now taking products of multiple fields does not take into account their locations
on the staggered grid and no attempt is made to interpolate all the different fields onto
a common location before calculating the product.
"""
function HorizontalAverage(model, fields; frequency=nothing, interval=nothing, return_type=Array)
    fields = parenttuple(tupleit(fields))
    profile = zeros(model.architecture, model.grid, 1, 1, model.grid.Tz)
    return HorizontalAverage(profile, fields, frequency, interval, 0.0, return_type)
end

# Normalize a horizontal sum to get the horizontal average.
normalize_horizontal_sum!(hsum, grid) = hsum.profile /= (grid.Nx * grid.Ny)

"""
    run_diagnostic(model, havg::HorizontalAverage{NTuple{1}})

Compute the horizontal average of `havg.fields` and store the result in `havg.profile`.
If length(fields) > 1, compute the product of the elements of fields (without taking
into account the possibility that they may have different locations in the staggered
grid) before computing the horizontal average.
"""
function run_diagnostic(model, havg::HorizontalAverage{NTuple{1}})
    zero_halo_regions!(havg.fields[1], model.grid)
    sum!(havg.profile, havg.fields[1])
    normalize_horizontal_sum!(havg, model.grid)
    return nothing
end

function run_diagnostic(model::Model, havg::HorizontalAverage)
    zero_halo_regions!(havg.fields, model.grid)

    # Use pressure as scratch space for the product of fields.
    tmp = model.pressures.pNHS.data.parent
    zero_halo_regions!(tmp, model.grid)

    @. tmp = *(havg.fields...)
    sum!(havg.profile, tmp)
    normalize_horizontal_sum!(havg, model.grid)

    return nothing
end

function (havg::HorizontalAverage{F, Nothing})(model) where F
    run_diagnostic(model, havg)
    return havg.profile
end

function (havg::HorizontalAverage)(model)
    run_diagnostic(model, havg)
    return havg.return_type(havg.profile)
end
