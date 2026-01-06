using Printf
using Oceananigans.Architectures: cpu_architecture
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Fields: set_to_function!

#####
##### set!
#####

iterations_from_file(file::JLD2.JLDFile) = parse.(Int, keys(file["timeseries/t"]))

function find_time_index(time::Number, file_times, Δt)
    # We introduce an additional absolute tolerance to accomodate
    # time values very close to zero, for which a relative tolerance will not work
    # (see https://github.com/CliMA/Oceananigans.jl/pull/4505)
    ϵa = 100 * eps(Δt)
    ϵr = sqrt(eps(eltype(file_times))) # The default relative tolerance used by `isapprox` when atol == 0
    return findfirst(t -> isapprox(t, time; atol=ϵa, rtol=ϵr), file_times)
end

find_time_index(time::AbstractTime, file_times, Δt) = findfirst(t -> t == time, file_times)

set_from_netcdf!(fts, path::String, args...; kwargs...) = error("Setting FieldTimeSeries from NetCDF files requires NCDatasets")

function set!(fts::InMemoryFTS, path::String=fts.path, args...; kwargs...)
    if endswith(path, ".jld2")
        file = jldopen(path; fts.reader_kw...)
        set!(fts, file, args...; kwargs...)
        close(file)
    elseif endswith(path, ".nc")
        return set_from_netcdf!(fts, path, args...; kwargs...)
    else
        error("Unsupported file extension: $(path)")
    end
end

function set!(fts::InMemoryFTS, file::JLD2.JLDFile, name::String=fts.name; warn_missing_data=true)
    file_iterations = iterations_from_file(file)
    file_times = [file["timeseries/t/$i"] for i in file_iterations]

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

set!(fts::InMemoryFTS, value, n::Int) = set!(fts[n], value)

function set!(fts::InMemoryFTS, fields_vector::AbstractVector{<:AbstractField})
    raw_data = parent(fts)
    file = jldopen(path; fts.reader_kw...)

    for (n, field) in enumerate(fields_vector)
        nth_raw_data = view(raw_data, :, :, :, n)
        copyto!(nth_raw_data, parent(field))
        # raw_data[:, :, :, n] .= parent(field)
    end

    close(file)

    return nothing
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
end

function initialize_file!(file, name, fts)
    maybe_write_property!(file, "serialized/grid", fts.grid)
    maybe_write_property!(file, "timeseries/$name/serialized/location", location(fts))
    maybe_write_property!(file, "timeseries/$name/serialized/indices", indices(fts))
    maybe_write_property!(file, "timeseries/$name/serialized/boundary_conditions", boundary_conditions(fts))
    return nothing
end

set!(fts::OnDiskFTS, path::String, name::String) = nothing

function set!(fts::InMemoryFTS, f::Function)
    cpu_times = on_architecture(CPU(), fts.times)
    n1 = first(time_indices(fts))
    clock = Clock(time=cpu_times[n1])

    for n in time_indices(fts)
        clock.time = cpu_times[n]
        set_to_function!(fts[n], f, clock)
    end

    return fts
end
