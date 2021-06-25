using Oceananigans.Fields: ComputedField
using Statistics

import Oceananigans.Fields: AveragedField

"""
    AveragedField(op::AbstractOperation; dims, data=nothing, operand_data=nothing,
                  recompute_safely=false, recompute_operand_safely=false)

Forms a `ComputedField` to store the result of computing `op`, and returns an
`AveragedField` whose operand is the new `ComputedField`, representing an average over
`dims`.

If `data` is not provided, memory is allocated to store the result of the average.
See `AveragedField(field::AbstractField)`.

The keyword argument `operand_data` can be used to specify memory or scratch space
for the new `ComputedField` data.
"""
function AveragedField(op::AbstractOperation; dims, data=nothing, operand_data=nothing,
                       recompute_safely=false, recompute_operand_safely=false)

    computed = ComputedField(op, data=operand_data, recompute_safely=recompute_operand_safely)

    return AveragedField(computed, dims=dims, data=data, recompute_safely=recompute_safely)
end
