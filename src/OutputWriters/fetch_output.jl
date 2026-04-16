using Oceananigans.Fields: AbstractField, compute_at!, ZeroField
import Oceananigans.Grids: grid
import Oceananigans.Fields: location, indices

struct DeferredSlicedOutput{SO, I}
    source_output :: SO
    write_indices :: I
end
grid(output::DeferredSlicedOutput) = grid(output.source_output)
location(output::DeferredSlicedOutput) = location(output.source_output)
indices(output::DeferredSlicedOutput) = output.write_indices
boundary_conditions(output::DeferredSlicedOutput) = boundary_conditions(output.source_output)


# TODO: figure out how to support this
# using Oceananigans.OutputReaders: FieldTimeSeries
# using Oceananigans.Units: Time
# fetch_output(fts::FieldTimeSeries, model) = fetch_output(fts[Time(model.clock)])

# Needed to support `fetch_output` with `model::Nothing`.
time(model) = model.clock.time
time(::Nothing) = nothing

fetch_output(output, model) = output(model)

function fetch_output(field::AbstractField, model)
    compute_at!(field, time(model))
    return parent(field)
end

function fetch_output(output::DeferredSlicedOutput, model)
    full_output = fetch_output(output.source_output, model)
    return view(full_output, output.write_indices...)
end

convert_output(output, writer) = output

function convert_output(output::AbstractArray, array_type)
    if architecture(output) isa GPU
        output_array = array_type(undef, size(output)...)
        copyto!(output_array, output)
    else
        output_array = convert(array_type, output)
    end

    return output_array
end

convert_output(output::AbstractArray, writer::AbstractOutputWriter) = convert_output(output, writer.array_type)

# Need to broadcast manually because of https://github.com/JuliaLang/julia/issues/30836
convert_output(outputs::NamedTuple, writer) =
    NamedTuple(name => convert_output(outputs[name], writer) for name in keys(outputs))

function fetch_and_convert_output(output, model, writer)
    fetched = fetch_output(output, model)
    return convert_output(fetched, writer)
end

fetch_and_convert_output(output::ZeroField, model, writer) = zero(eltype(output))
