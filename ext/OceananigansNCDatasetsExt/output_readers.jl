#####
##### FieldTimeSeries from NetCDF
#####

function FieldTimeSeries_from_netcdf(path::String, name::String;
                                     backend = InMemory(),
                                     architecture = nothing,
                                     grid = nothing,
                                     location = nothing,
                                     boundary_conditions = UnspecifiedBoundaryConditions(),
                                     time_indexing = Linear(),
                                     iterations = nothing,
                                     times = nothing,
                                     reader_kw = NamedTuple())

    file = NCDataset(path; reader_kw...)

    indices = try
        file[name].attrib["indices"] |> materialize_from_netcdf
    catch
        (:, :, :)
    end

    if isnothing(architecture) # determine architecture
        if isnothing(grid) # go to default
            architecture = CPU()
        else # there's a grid, use that architecture
            architecture = Oceananigans.Architectures.architecture(grid)
        end
    end

    isnothing(grid) && (grid = reconstruct_grid(file))

    @warn "Reading boundary conditions from NetCDF files is not supported for FieldTimeSeries. Defaulting to UnspecifiedBoundaryConditions."
    boundary_conditions = UnspecifiedBoundaryConditions()

    isnothing(location) && (location = file[name].attrib["location"] |> materialize_from_netcdf)
    LX, LY, LZ = location
    loc = (LX(), LY(), LZ())

    isnothing(times) && (times = file["time"] |> collect)

    Nt = time_indices_length(backend, times)
    data = new_data(eltype(grid), grid, loc, indices, Nt)

    time_series = FieldTimeSeries{LX, LY, LZ}(data, grid, backend, boundary_conditions, indices,
                                              times, path, name, time_indexing, reader_kw)

    set!(time_series, path, name)
    close(file)

    return time_series
end

iterations_from_file(file::NCDataset) = 1:length(keys(file["time"][:]))

function find_time_index(t, file_times, Δt)
    # Find the index in file_times that is closest to t
    for (i, file_time) in enumerate(file_times)
        if abs(file_time - t) < Δt / 2
            return i
        end
    end
    return nothing
end

function set_from_netcdf!(fts::InMemoryFTS, path::String, name; warn_missing_data=true)
    file = NCDataset(path)
    file_iterations = iterations_from_file(file)
    file_times = file["time"]

    # Compute a timescale for comparisons
    Δt = mean(diff(file_times))

    arch = architecture(fts)

    # TODO: a potential optimization here might be to load
    # all of the data into a single array, and then transfer that
    # to parent(fts).

    # Index times on the CPU
    cpu_times = on_architecture(CPU(), fts.times)

    for n in time_indices(fts)
        t = cpu_times[n]
        file_index = find_time_index(t, file_times, Δt)

        if isnothing(file_index) # the time does not exist in the file
            if warn_missing_data
                msg = @sprintf("No data found for time %.1e and time index %d\n", t, n)
                msg *= @sprintf("for field %s at path %s", file.path, name)
                @warn msg
            end
        else
            file_iter = file_iterations[file_index]

            # Note: use the CPU for this step
            field_n = Field(instantiated_location(fts), file, name, file_iter,
                            grid = on_architecture(CPU(), fts.grid),
                            architecture = cpu_architecture(arch),
                            indices = fts.indices,
                            boundary_conditions = fts.boundary_conditions)

            # Potentially transfer from CPU to GPU
            set!(fts[n], field_n)
        end
    end
    close(file)

    return nothing
end
