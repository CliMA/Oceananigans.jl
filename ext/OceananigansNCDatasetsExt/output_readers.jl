using Oceananigans.Fields: instantiated_location
using Oceananigans.Architectures: cpu_architecture
using Oceananigans.Utils: @apply_regionally
using Oceananigans.Grids: offset_data
using Oceananigans.BoundaryConditions: FieldBoundaryConditions

import Oceananigans.Fields: Field


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


    isnothing(location) && (location = file[name].attrib["location"] |> materialize_from_netcdf)
    LX, LY, LZ = location
    loc = (LX(), LY(), LZ())

    @warn "Reading boundary conditions from NetCDF files is not supported for FieldTimeSeries. Using default FieldBoundaryConditions for `grid` and `location`."
    boundary_conditions = FieldBoundaryConditions(grid, loc)

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
            # Main.@infiltrate
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

"""
    Field(location, file::NCDataset, name::String, iter;
          grid = nothing,
          architecture = nothing,
          indices = (:, :, :),
          boundary_conditions = nothing,
          reader_kw = NamedTuple())

Load a field called `name` saved in a NetCDF file at `path` at `iter`ation.
Unless specified, the `grid` is loaded from `path`.
"""
function Field(location, file::NCDataset, name::String, iter;
               grid = nothing,
               architecture = nothing,
               indices = (:, :, :),
               boundary_conditions = nothing,
               reader_kw = NamedTuple())

    # Default to CPU if neither architecture nor grid is specified
    if isnothing(architecture)
        if isnothing(grid)
            architecture = CPU()
        else
            architecture = Architectures.architecture(grid)
        end
    end

    isnothing(grid) && (grid = reconstruct_grid(file))
    variable_dimensions = dimnames(file[name])
    time_slice = (ntuple(_ -> :, length(variable_dimensions)-1)..., iter)
    raw_data = file[name][time_slice...]

    # Change grid to specified architecture?
    grid = on_architecture(architecture, grid)
    raw_data = on_architecture(architecture, raw_data)
    # The following line is commented out because I can't make it work with @apply_regionally
    @apply_regionally data = offset_data(raw_data, grid, location, indices)

    return Field(location, grid; boundary_conditions, indices, data)
end
