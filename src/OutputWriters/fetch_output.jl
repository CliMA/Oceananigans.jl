using CUDA

using Oceananigans.Fields: AbstractField, compute!

fetch_output(output, model, field_slicer) = output(model)

# Needed to support `fetch_output` with `model::Nothing`.
time(model) = model.clock.time
time(::Nothing) = nothing

function fetch_output(field::AbstractField, model, field_slicer)
    compute!(field, time(model))
    return slice_parent(field_slicer, field)
end

convert_output(output, writer) = output
convert_output(output::AbstractArray, writer) = CUDA.@allowscalar writer.array_type(output)

fetch_and_convert_output(output, model, writer) =
    convert_output(fetch_output(output, model, writer.field_slicer), writer)
