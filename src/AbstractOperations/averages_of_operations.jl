using Oceananigans.Fields: ComputedField
using Statistics

import Oceananigans.Fields: AveragedField

"""
    AveragedField(op::AbstractOperation; dims, data=nothing, computed_data=nothing)

Forms a `ComputedField` to store the result of computing `op`, and returns an
`AveragedField` whose operand is the new `ComputedField`, representing an average over
`dims`.

If `data` is not provided, memory is allocated to store the result of the average.
See `AveragedField(field::AbstratField)`.

The keyword argument `computed_data` can be used to specify memory or scratch space
for the new `ComputedField` data.
"""
function AveragedField(op::AbstractOperation; dims, data=nothing, computed_data=nothing)
    computed = ComputedField(op, data=computed_data)
    return AveragedField(computed, dims=dims, data=data)
end

"""
    mean(op::AbstractOperation; kwargs...)

Returns an Oceananigans.AveragedField representing the an average over `op`eration.
See `Oceananigans.AbstractField`.
"""
Statistics.mean(op::AbstractOperation; kwargs...) = AveragedField(op; kwargs...)
