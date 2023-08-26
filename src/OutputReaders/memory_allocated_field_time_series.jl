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
 
@propagate_inbounds Base.getindex(f::InMemoryFieldTimeSeries, i, j, k, n::Int) = f.data[i, j, k, n]
@propagate_inbounds Base.getindex(f::ChunkedFieldTimeSeries, i, j, k, n::Int) = f.data.data_in_memory[i, j, k, n - f.data.index_range[1] + 1]
@propagate_inbounds Base.setindex!(f::ChunkedFieldTimeSeries, v, i, j, k, n::Int) = setindex!(f.data.data_in_memory, v, i, j, k, n - f.data.index_range[1] + 1)

Base.parent(fts::InMemoryFieldTimeSeries) = parent(fts.data)
Base.parent(fts::ChunkedFieldTimeSeries)  = parent(fts.data.data_in_memory)

function Base.getindex(fts::InMemoryFieldTimeSeries, n::Int)
    underlying_data = view(parent(fts), :, :, :, n) 
    data = offset_data(underlying_data, fts.grid, location(fts), fts.indices)
    return Field(location(fts), fts.grid; data, fts.boundary_conditions, fts.indices)
end

# If `n` is not in memory, getindex automatically sets the data in memory to have the `n`
# as the second index (to allow interpolation with the previous time step)
# If n is `1` or within the end the timeseries different rules apply
function Base.getindex(fts::ChunkedFieldTimeSeries, n::Int)
    if !(n ∈ fts.data.index_range)
        Nt = length(fts.times)
        Ni = length(fts.data.index_range)
        if n == 1
            set!(fts, 1:Ni)
        elseif n > Nt - Ni
            set!(fts, Nt-Ni+1:Nt)
        else
            set!(fts, n-1:n+Ni-2)
        end
    end
    underlying_data = view(parent(fts), :, :, :,  n - fts.data.index_range[1] + 1) 
    data = offset_data(underlying_data, fts.grid, location(fts), fts.indices)
    return Field(location(fts), fts.grid; data, fts.boundary_conditions, fts.indices)
end

set!(time_series::MFTS, f, index::Int) = set!(time_series[index], f)

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