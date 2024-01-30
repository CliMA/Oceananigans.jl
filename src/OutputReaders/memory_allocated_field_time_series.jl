
function new_data(FT, grid, loc, indices, Nt, backend::InMemory) 
    space_size = total_size(grid, loc, indices)
    Nt = backend.index_range == Colon() ? Nt : length(backend.index_range)
    underlying_data = zeros(FT, architecture(grid), space_size..., Nt)
    data = offset_data(underlying_data, grid, loc, indices)
    return data
end
 
@propagate_inbounds Base.getindex(f::InMemoryFieldTimeSeries, i, j, k, n::Int) =
    f.data[i, j, k, n - f.backend.index_range[1] + 1]

@propagate_inbounds function Base.getindex(f::CyclicalFTS{InMemory{Tuple}}, i, j, k, n::Int)
    Ni = length(f.backend.index_range)
    # Should find n₁ == n₂
    n₁, n₂ = index_binary_search(f.backend.index_range, n, Ni)
    return f.data[i, j, k, n₁]
end

@propagate_inbounds Base.getindex(f::TotallyInMemoryFieldTimeSeries, i, j, k, n::Int) =
    f.data[i, j, k, n]

@propagate_inbounds Base.setindex!(f::InMemoryFieldTimeSeries, v, i, j, k, n::Int) =
    setindex!(f.data, v, i, j, k, n - f.backend.index_range[1] + 1)

@propagate_inbounds Base.setindex!(f::TotallyInMemoryFieldTimeSeries, v, i, j, k, n::Int) =
    setindex!(f.data, v, i, j, k, n)

Base.parent(fts::InMemoryFieldTimeSeries) = parent(fts.data)

compute_time_index(index_range, n) = n - index_range[1] + 1
compute_time_index(::Colon, n) = n

# If `n` is not in memory, getindex automatically sets the data in memory to have the `n`
# as the second index (to allow interpolation with the previous time step)
# If n is `1` or within the end the timeseries different rules apply
function Base.getindex(fts::InMemoryFieldTimeSeries, n::Int)
    update_field_time_series!(fts, n)

    index_range = fts.backend.index_range
    time_index = compute_time_index(index_range, n)
    underlying_data = view(parent(fts), :, :, :, time_index)

    data = offset_data(underlying_data, fts.grid, location(fts), fts.indices)

    return Field(location(fts), fts.grid; data, fts.boundary_conditions, fts.indices)
end

set!(fts::InMemoryFieldTimeSeries, f, index::Int) = set!(fts[index], f)

iterations_from_file(file, ::Colon) = parse.(Int, keys(file["timeseries/t"]))

function iterations_from_file(file, index_range::Tuple)
    all_iterations = iterations_from_file(file, Colon())
    return all_iterations[index_range]
end

time_indices(fts::InMemoryFieldTimeSeries) = time_indices(fts.backend.index_range, fts.times)
time_indices(::Colon, times) = UnitRange(1, length(times))
time_indices(index_range, times) = index_range

find_time_index(time::Number, file_times)       = findfirst(t -> t ≈ time, file_times)
find_time_index(time::AbstractTime, file_times) = findfirst(t -> t == time, file_times)

function set!(fts::InMemoryFieldTimeSeries, path::String, name::String)
    index_range = fts.backend.index_range

    file = jldopen(path)
    file_iterations = iterations_from_file(file, index_range)
    file_times = [file["timeseries/t/$i"] for i in file_iterations]
    close(file)

    times = fts.times[index_range]
    indices = time_indices(fts)

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

