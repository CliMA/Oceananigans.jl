#####
##### FieldTimeSeries reader for Zarr stores
#####

using Oceananigans.Architectures: on_architecture, cpu_architecture, architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Fields: instantiated_location, set!, location, indices, interior
using Oceananigans.Grids: offset_data
using Oceananigans.OutputReaders:
    InMemoryFTS,
    InMemory, OnDisk, Linear, UnspecifiedBoundaryConditions,
    new_data, time_indices, time_indices_length, ZarrPath

import Oceananigans.OutputReaders: set_from_zarr!, FieldTimeSeries
import Oceananigans.Fields: Field

#####
##### FieldTimeSeries from Zarr
#####

function _open_zarr_for_read(path::AbstractString)
    if endswith(path, ".zip")
        bytes = read(path)
        store = Zarr.ZipStore(bytes)
        return Zarr.zopen(store)
    else
        return Zarr.zopen(path)
    end
end

using Oceananigans.Grids: Center, Face

# Parse a location string like "Center" or "Face" back to a type.
function _parse_location(s::AbstractString)
    s == "Nothing" && return Nothing
    s == "Center"  && return Center
    s == "Face"    && return Face
    error("Unknown location string: $s")
end

# Parse an indices string like ":", "1:4", "3" back to its index value.
function _parse_index(s::AbstractString)
    s == ":" && return Colon()
    if occursin(":", s)
        parts = split(s, ":")
        return parse(Int, parts[1]):parse(Int, parts[2])
    else
        return parse(Int, s)
    end
end

function FieldTimeSeries(typed_path::ZarrPath, name::String;
                         backend = InMemory(),
                         architecture = CPU(),
                         grid = nothing,
                         location = nothing,
                         boundary_conditions = UnspecifiedBoundaryConditions(),
                         time_indexing = Linear(),
                         iterations = nothing,
                         times = nothing,
                         reader_kw = NamedTuple())

    path  = typed_path.path
    group = _open_zarr_for_read(path)
    arr   = group[name]

    grid_index = try Int(arr.attrs["grid_index"]) catch; 1 end
    isnothing(grid) && (grid = reconstruct_zarr_grid(group; grid_index, architecture))

    if isnothing(location)
        loc_strs = collect(arr.attrs["location"])
        location = Tuple(_parse_location(s) for s in loc_strs)
    end
    LX, LY, LZ = location
    loc = (LX(), LY(), LZ())

    file_indices = try
        Tuple(_parse_index(s) for s in collect(arr.attrs["indices"]))
    catch
        (:, :, :)
    end

    if !(boundary_conditions isa FieldBoundaryConditions)
        @warn "Reading boundary conditions from Zarr stores is not supported. Using default FieldBoundaryConditions for `grid` and `location`."
        boundary_conditions = FieldBoundaryConditions(grid, loc)
    end

    isnothing(times) && (times = group["time"][:] |> collect)

    Nt = time_indices_length(backend, times)
    data = new_data(eltype(grid), grid, loc, file_indices, Nt)

    time_series = FieldTimeSeries{LX, LY, LZ}(data, grid, backend, boundary_conditions, file_indices,
                                              times, path, name, time_indexing, reader_kw)

    set!(time_series, path, name)
    return time_series
end

#####
##### set! / set_from_zarr!
#####

function set_from_zarr!(fts::InMemoryFTS, path::String, name; warn_missing_data=true)
    group = _open_zarr_for_read(path)
    arr   = group[name]
    file_times = group["time"][:]

    arch = architecture(fts)
    cpu_times = on_architecture(CPU(), fts.times)
    grid = on_architecture(arch, fts.grid)
    loc = instantiated_location(fts)

    # The on-disk Zarr array is (spatial..., Nt). We read slice n into the corresponding
    # fts entry. For each requested fts time index, find the closest matching file time.
    Δt = length(file_times) > 1 ? abs(file_times[2] - file_times[1]) : one(eltype(file_times))

    for n in time_indices(fts)
        t = cpu_times[n]
        file_index = findfirst(ft -> isapprox(ft, t; atol=100*eps(Δt), rtol=sqrt(eps(eltype(file_times)))), file_times)
        if isnothing(file_index)
            warn_missing_data && @warn "No data at time $t (index $n) for $name in $path"
            continue
        end

        field_n = Field(loc, arr, name, file_index;
                        grid = on_architecture(CPU(), fts.grid),
                        architecture = cpu_architecture(arch),
                        indices = fts.indices,
                        boundary_conditions = fts.boundary_conditions)
        set!(fts[n], field_n)
    end
    return nothing
end

# Load a single timestep from a Zarr array into a Field. The on-disk data may have been
# saved with halos (`with_halos=true`) or interior-only (`with_halos=false`). Distinguish
# by shape: if `raw` matches the halo-extended parent shape, use `offset_data`;
# otherwise it should match the interior, in which case we allocate a fresh Field and
# copy the interior in.
function Field(loc::Tuple, arr::Zarr.ZArray, name::String, time_index::Int;
               grid,
               architecture = CPU(),
               indices = (:, :, :),
               boundary_conditions = nothing)

    nd = ndims(arr)
    time_slice = (ntuple(_ -> :, nd - 1)..., time_index)
    raw = arr[time_slice...]

    grid_arch = on_architecture(architecture, grid)
    raw_arch  = on_architecture(architecture, raw)

    fld = Oceananigans.Fields.Field(loc, grid_arch; boundary_conditions, indices)
    parent_size = size(parent(fld))
    if size(raw_arch) == parent_size[1:length(size(raw_arch))]
        # raw matches halo-extended parent — copy parent-to-parent.
        copyto!(parent(fld), raw_arch)
    else
        # raw matches the interior — copy into the interior view.
        interior(fld) .= raw_arch
    end
    return fld
end
