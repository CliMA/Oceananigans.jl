using Oceananigans.Fields: AbstractField, compute!

fetch_output(output, model, writer) = output(model)

convert_output(output, writer) = output
convert_output(output::AbstractArray, writer) = writer.array_type(parent(output))

function fetch_output(field::AbstractField, model, writer)
    compute!(field)
    return convert_output(data(field), writer)
end
