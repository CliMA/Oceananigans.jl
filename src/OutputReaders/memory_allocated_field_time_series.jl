
function new_data(FT, grid, loc, indices, Nt, backend::InMemory) 
    space_size = total_size(grid, loc, indices)
    Nt = ifelse(backend.index_range == Colon(), Nt, length(backend.index_range))
    underlying_data = zeros(FT, architecture(grid), space_size..., Nt)
    data = offset_data(underlying_data, grid, loc, indices)
    return data
end
 
@propagate_inbounds Base.getindex(f::InMemoryFieldTimeSeries, i, j, k, n::Int) = f.data[i, j, k, n - f.backend.index_range[1] + 1]
@propagate_inbounds Base.setindex!(f::InMemoryFieldTimeSeries, v, i, j, k, n::Int) = setindex!(f.data, v, i, j, k, n - f.backend.index_range[1] + 1)

Base.parent(fts::InMemoryFieldTimeSeries) = parent(fts.data)

# If `n` is not in memory, getindex automatically sets the data in memory to have the `n`
# as the second index (to allow interpolation with the previous time step)
# If n is `1` or within the end the timeseries different rules apply
function Base.getindex(fts::InMemoryFieldTimeSeries, n::Int)
    update_time_series!(fts, n)
    underlying_data = view(parent(fts), :, :, :, n) 
    data = offset_data(underlying_data, fts.grid, location(fts), fts.indices)
    return Field(location(fts), fts.grid; data, fts.boundary_conditions, fts.indices)
end

set!(fts::InMemoryFieldTimeSeries, f, index::Int) = set!(fts[index], f)

function set!(fts::InMemoryFieldTimeSeries, path::String, name::String)
    file = jldopen(path)
    index_range = fts.backend.index_range
    file_iterations = parse.(Int, keys(file["timeseries/t"]))[index_range]
    file_times = [file["timeseries/t/$i"] for i in file_iterations]
    close(file)

    for (n, time) in zip(index_range, fts.times[index_range])
        file_index = findfirst(t -> t ≈ time, file_times)
        file_iter = file_iterations[file_index]
        
        field_n = Field(location(fts), path, name, file_iter,
                        indices = fts.indices,
                        boundary_conditions = fts.boundary_conditions,
                        grid = fts.grid)

        set!(fts[n], field_n)
    end

    return nothing
end
