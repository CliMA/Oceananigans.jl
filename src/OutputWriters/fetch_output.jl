using Oceananigans.Fields: AbstractField, compute!

fetch_output(output, model, field_slicer) = output(model)

function fetch_output(field::AbstractField, model, field_slicer)
    compute!(field, model.clock.time)
    return slice_parent(field_slicer, field)
end

convert_output(output, writer) = output
convert_output(output::AbstractArray, writer) = writer.array_type(output)

fetch_and_convert_output(output, model, writer) =
    convert_output(fetch_output(output, model, writer.field_slicer), writer)
