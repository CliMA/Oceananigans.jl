function new_data(FT, grid, loc, indices, Nt, path, name, ::InMemory)
    space_size = total_size(grid, loc, indices)
    underlying_data = zeros(FT, architecture(grid), space_size..., Nt)
    return offset_data(underlying_data, grid, loc, indices)
end

function new_data(FT, grid, loc, indices, Nt, path, name, backend::Chunked) 
    space_size = total_size(grid, loc, indices)
    underlying_data = zeros(FT, architecture(grid), space_size..., backend.chunk_size)
    data = offset_data(underlying_data, grid, loc, indices)
    return ChunkedData(path, name, data, collect(1:backend.chunk_size))
end

# FieldTimeSeries with allocated memory
const MFTS = Union{InMemoryFieldTimeSeries, ChunkedFieldTimeSeries}
 
@propagate_inbounds Base.getindex(f::InMemoryFieldTimeSeries, i::Int, j::Int, k::Int, n::Int) = f.data[i, j, k, n]
@propagate_inbounds Base.getindex(f::ChunkedFieldTimeSeries, i::Int, j::Int, k::Int, n::Int) = f.data.data_in_memory[i, j, k, n]
@propagate_inbounds Base.setindex!(f::ChunkedFieldTimeSeries, v, inds...) = setindex!(f.data.data_in_memory, v, inds...)

Base.parent(fts::InMemoryFieldTimeSeries) = parent(fts.data)
Base.parent(fts::ChunkedFieldTimeSeries)  = parent(fts.data.data_in_memory)

displaced_index(n,   ::InMemoryFieldTimeSeries) = n
displaced_index(n, fts::ChunkedFieldTimeSeries) = n - fts.data.index_range[1] + 1

function Base.getindex(fts::MFTS, n::Int)
    n = displaced_index(n, fts)
    underlying_data = view(parent(fts), :, :, :, n) 
    data = offset_data(underlying_data, fts.grid, location(fts), fts.indices)
    boundary_conditions = fts.boundary_conditions
    indices = fts.indices
    return Field(location(fts), fts.grid; data, boundary_conditions, indices)
end

set!(time_series::InMemoryFieldTimeSeries, f, index::Int) = set!(time_series[index], f)

function set!(time_series::MFTS, path::String, name::String)

    file = jldopen(path)
    index_range = time_range(time_series)
    file_iterations = parse.(Int, keys(file["timeseries/t"]))[index_range]
    file_times = [file["timeseries/t/$i"] for i in file_iterations]
    close(file)

    for (n, time) in zip(index_range, time_series.times[index_range])
        file_index = findfirst(t -> t ≈ time, file_times)
        file_iter = file_iterations[file_index]

        field_n = Field(location(time_series), path, name, file_iter,
                        indices = time_series.indices,
                        boundary_conditions = time_series.boundary_conditions,
                        grid = time_series.grid)

        set!(time_series[n], field_n)
    end

    return nothing
end

time_range(fts::InMemoryFieldTimeSeries) = 1:length(fts)
time_range(fts::ChunkedFieldTimeSeries) = fts.data.index_range

function set!(time_series::ChunkedFieldTimeSeries, index_range::UnitRange)
    time_series.data.index_range .= index_range
    set!(time_series, time_series.data.path, time_series.data.name)

    return nothing
end