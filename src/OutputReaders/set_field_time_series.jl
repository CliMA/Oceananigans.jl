using Printf
using Oceananigans.Architectures: cpu_architecture

import Oceananigans.Fields: set!

function set!(u::InMemoryFTS, v::InMemoryFTS)
    if child_architecture(u) === child_architecture(v)
        # Note: we could try to copy first halo point even when halo
        # regions are a different size. That's a bit more complicated than
        # the below so we leave it for the future.
        
        try # to copy halo regions along with interior data
            parent(u) .= parent(v)
        catch # this could fail if the halo regions are different sizes?
            # copy just the interior data
            interior(u) .= interior(v)
        end
    else
        v_data = on_architecture(child_architecture(u), v.data)
        
        # As above, we permit ourselves a little ambition and try to copy halo data:
        try
            parent(u) .= parent(v_data)
        catch
            interior(u) .= interior(v_data, location(v), v.grid, v.indices)
        end
    end

    return u
end

function set!(u::InMemoryFTS, v::Function)
    # Supports serial and distributed
    arch = architecture(u)
    child_arch = child_architecture(u)
    LX, LY, LZ = location(u)

    # Determine cpu_grid and cpu_u
    if child_arch isa GPU
        cpu_arch = cpu_architecture(arch)
        cpu_grid = on_architecture(cpu_arch, u.grid)
        cpu_times = on_architecture(cpu_arch, u.times)
        cpu_u = FieldTimeSeries{LX, LY, LZ}(cpu_grid, cpu_times; indices=indices(u))
    elseif child_arch isa CPU
        cpu_arch = child_arch
        cpu_grid = u.grid
        cpu_times = u.times
        cpu_u = u
    end

    launch!(cpu_arch, cpu_grid, size(cpu_u),
            _set_fts_to_function!, cpu_u, (LX(), LY(), LZ()), cpu_grid, cpu_times, v)

    # Transfer data to GPU if u is on the GPU
    child_arch isa GPU && set!(u, cpu_u)
    
    return u
end

@kernel function _set_fts_to_function!(fts, loc, grid, times, func)
    i, j, k, n = @index(Global, NTuple)
    X = node(i, j, k, grid, loc...)
    @inbounds begin
        fts[i, j, k, n] = func(X..., times[n])
    end
end

#####
##### set!
#####

iterations_from_file(file) = parse.(Int, keys(file["timeseries/t"]))

find_time_index(time::Number, file_times)       = findfirst(t -> t ≈ time, file_times)
find_time_index(time::AbstractTime, file_times) = findfirst(t -> t == time, file_times)

function set!(fts::InMemoryFTS, path::String=fts.path, name::String=fts.name; warn_missing_data=true)
    file = jldopen(path; fts.reader_kw...)
    file_iterations = iterations_from_file(file)
    file_times = [file["timeseries/t/$i"] for i in file_iterations]
    close(file)

    arch = architecture(fts)

    # TODO: a potential optimization here might be to load
    # all of the data into a single array, and then transfer that
    # to parent(fts).
    for n in time_indices(fts)
        t = fts.times[n]
        file_index = find_time_index(t, file_times)

        if isnothing(file_index) # the time does not exist in the file
            if warn_missing_data
                msg = @sprintf("No data found for time %.1e and time index %d\n", t, n)
                msg *= @sprintf("for field %s at path %s", path, name)
                @warn msg
            end
        else
            file_iter = file_iterations[file_index]

            # Note: use the CPU for this step
            field_n = Field(location(fts), path, name, file_iter,
                            grid = on_architecture(CPU(), fts.grid),
                            architecture = cpu_architecture(arch),
                            indices = fts.indices,
                            boundary_conditions = fts.boundary_conditions)

            # Potentially transfer from CPU to GPU
            set!(fts[n], field_n)
        end
    end

    return fts
end

set!(fts::InMemoryFTS, v, n::Int) = set!(fts[n], value)

function set!(fts::InMemoryFTS, fields_vector::AbstractVector{<:AbstractField})
    raw_data = parent(fts)
    file = jldopen(path; fts.reader_kw...)

    for (n, field) in enumerate(fields_vector)
        nth_raw_data = view(raw_data, :, :, :, n)
        copyto!(nth_raw_data, parent(field))
        # raw_data[:, :, :, n] .= parent(field)
    end

    close(file)

    return fts
end

# Write property only if it does not already exist
function maybe_write_property!(file, property, data)
    try
        test = file[property]
    catch
        file[property] = data
    end
end

"""
    set!(fts::OnDiskFieldTimeSeries, field::Field, n::Int, time=fts.times[time_index])

Write the data in `parent(field)` to the file at `fts.path`,
under `fts.name` and at `time_index`. The save field is assigned `time`,
which is extracted from `fts.times[time_index]` if not provided.
"""
function set!(fts::OnDiskFTS, field::Field, n::Int, time=fts.times[n])
    fts.grid == field.grid || error("The grids attached to the Field and \
                                    FieldTimeSeries appear to be different.")
    path = fts.path
    name = fts.name
    jldopen(path, "a+") do file
        initialize_file!(file, name, fts)
        maybe_write_property!(file, "timeseries/t/$n", time)
        maybe_write_property!(file, "timeseries/$name/$n", Array(parent(field)))
    end

    return fts
end

function initialize_file!(file, name, fts)
    maybe_write_property!(file, "serialized/grid", fts.grid)
    maybe_write_property!(file, "timeseries/$name/serialized/location", location(fts))
    maybe_write_property!(file, "timeseries/$name/serialized/indices", indices(fts))
    maybe_write_property!(file, "timeseries/$name/serialized/boundary_conditions", boundary_conditions(fts))
    return nothing
end

set!(fts::OnDiskFTS, path::String, name::String) = fts

