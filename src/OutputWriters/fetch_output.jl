using CUDA

using Oceananigans.Fields: AbstractField, ZeroField, compute_at!, reduced_dimensions

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

const XWindowedIndices = Tuple{<:UnitRange, Colon, Colon}
const YWindowedIndices = Tuple{Colon, <:UnitRange, Colon}
const ZWindowedIndices = Tuple{Colon, Colon, <:UnitRange}
const XYWindowedIndices = Tuple{<:UnitRange, <:UnitRange, Colon}
const XZWindowedIndices = Tuple{<:UnitRange, Colon, <:UnitRange}
const YZWindowedIndices = Tuple{Colon, <:UnitRange, <:UnitRange}
const XYZWindowedIndices = Tuple{<:UnitRange, <:UnitRange, <:UnitRange}
const WindowedIndices = Union{XWindowedIndices, YWindowedIndices, ZWindowedIndices, XYWindowedIndices, XZWindowedIndices, YZWindowedIndices, XYZWindowedIndices}
const WindowedFieldByIndices = Field{<:Any, <:Any, <:Any, <:Any, <:Any, <:WindowedIndices}

function fetch_output(field::WindowedFieldByIndices, model)
    compute_at!(field, time(model))
    data = parent(field)

    reduced_dims = reduced_dimensions(field)
    if !isempty(reduced_dims)
        data = dropdims(data; dims=reduced_dims)
    end

    return data
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
