
function new_data(FT, grid, loc, indices, Nt, backend::InMemory) 
    space_size = total_size(grid, loc, indices)
    Nt = backend.indices == Colon() ? Nt : length(backend.indices)
    underlying_data = zeros(FT, architecture(grid), space_size..., Nt)
    data = offset_data(underlying_data, grid, loc, indices)
    return data
end
 
Base.parent(fts::InMemoryFieldTimeSeries) = parent(fts.data)

compute_time_index(indices, n) = n - indices[1] + 1
compute_time_index(::Colon, n) = n

# If `n` is not in memory, getindex automatically sets the data in memory to have the `n`
# as the second index (to allow interpolation with the previous time step)
# If n is `1` or within the end the timeseries different rules apply
function Base.getindex(fts::InMemoryFieldTimeSeries, n::Int)
    update_field_time_series!(fts, n, n)

    indices = fts.backend.indices
    time_index = compute_time_index(indices, n)

    time_index = in_memory_time_index(fts.time_extrapolation, indices, n)
    underlying_data = view(parent(fts), :, :, :, time_index)

    data = offset_data(underlying_data, fts.grid, location(fts), fts.indices)

    return Field(location(fts), fts.grid; data, fts.boundary_conditions, fts.indices)
end

set!(fts::InMemoryFieldTimeSeries, f, index::Int) = set!(fts[index], f)

iterations_from_file(file, ::Colon) = parse.(Int, keys(file["timeseries/t"]))

function iterations_from_file(file, indices::Tuple)
    I = iterations_from_file(file, Colon())
    return [I[n] for n in indices]
end

time_indices(fts::InMemoryFieldTimeSeries) = time_indices(fts.backend.indices, fts.times)
time_indices(::Colon, times) = UnitRange(1, length(times))
time_indices(indices, times) = indices

find_time_index(time::Number, file_times)       = findfirst(t -> t ≈ time, file_times)
find_time_index(time::AbstractTime, file_times) = findfirst(t -> t == time, file_times)

function set!(fts::InMemoryFieldTimeSeries, path::String, name::String)
    indices = fts.backend.indices

    file = jldopen(path)
    file_iterations = iterations_from_file(file, indices)
    file_times = [file["timeseries/t/$i"] for i in file_iterations]
    close(file)

    indices = [i for i in indices]
    times = fts.times[indices]

    for (n, time) in zip(indices, times)
        file_index = find_time_index(time, file_times)
        file_iter = file_iterations[file_index]
        
        field_n = Field(location(fts), path, name, file_iter,
                        indices = fts.indices,
                        boundary_conditions = fts.boundary_conditions,
                        grid = fts.grid)

        set!(fts[n], field_n)
    end

    return nothing
end

const MAX_FTS_TUPLE_SIZE = 10

function fill_halo_regions!(fts::InMemoryFieldTimeSeries)
    partitioned_indices = Iterators.partition(time_indices(fts), MAX_FTS_TUPLE_SIZE) |> collect
    Ni = length(partitioned_indices)

    asyncmap(1:Ni) do i
        indices = partitioned_indices[i]
        fts_tuple = Tuple(fts[n] for n in indices)
        fill_halo_regions!(fts_tuple)
    end

    return nothing
end

