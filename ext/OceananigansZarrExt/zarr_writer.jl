#####
##### Zarr output writer for Oceananigans
#####
##### Phase 1: skeleton — constructor sets up state but does not touch the store.
##### Time-axis writing, initialization on disk, and write_output! come in Phase 2.
#####

function ZarrWriter(model::AbstractModel, outputs;
                    filename = nothing,
                    schedule,
                    dir = ".",
                    indices = (:, :, :),
                    with_halos = true,
                    array_type = Array{Float32},
                    file_splitting = NoFileSplitting(),
                    overwrite_existing = false,
                    verbose = false,
                    part = 1,
                    store = nothing,
                    chunks = nothing,
                    compressor = nothing,
                    dimensions = Dict{String, Any}())

    # Reject ZipStore explicitly — it's read-only in Zarr.jl by design.
    if store isa Zarr.ZipStore
        throw(ArgumentError("""
            ZipStore is read-only in Zarr.jl; ZarrWriter cannot write to one.

            Write to a DirectoryStore (or DictStore) and finalize to zip at the end:

                writer = ZarrWriter(model, outputs; filename="out", ...)
                run!(simulation)
                open("out.zip", "w") do io
                    Zarr.writezip(io, writer.store)
                end
            """))
    end

    # Resolve filepath + store.
    if isnothing(store)
        isnothing(filename) && throw(ArgumentError("ZarrWriter requires either `filename` or `store`"))
        mkpath(dir)
        filename = auto_extension(filename, ".zarr")
        filepath = abspath(joinpath(dir, filename))
        # Construct DirectoryStore eagerly so the struct's `store` field has a concrete
        # type. The directory itself is not created until `initialize!` runs.
        store = Zarr.DirectoryStore(filepath)
    else
        filepath = string(store)
    end

    initialize!(file_splitting, model)
    update_file_splitting_schedule!(file_splitting, filepath)

    nt_outputs = NamedTuple(Symbol(name) => construct_output(outputs[name], indices, with_halos)
                            for name in keys(outputs))
    schedule = materialize_schedule(schedule)
    schedule, d_outputs = time_average_outputs(schedule, nt_outputs, model)

    # Detect unique grids across all outputs. Outputs without a grid (functions,
    # particles) map to grid_index = nothing.
    output_grids = Dict{String, Any}()
    for (n, out) in pairs(d_outputs)
        g = output_grid(out)
        output_grids[string(n)] = g
    end
    unique_grids = Tuple(unique(objectid, filter(!isnothing, collect(values(output_grids)))))
    output_grid_map = Dict{String, Any}(
        name => (isnothing(g) ? nothing : findfirst(ug -> ug === g, unique_grids))
        for (name, g) in output_grids
    )

    return ZarrWriter(filepath,
                      store,
                      unique_grids,
                      output_grid_map,
                      d_outputs,
                      schedule,
                      array_type,
                      indices,
                      with_halos,
                      overwrite_existing,
                      verbose,
                      part,
                      file_splitting,
                      compressor,
                      chunks,
                      Dict{String, Any}(string(k) => v for (k, v) in pairs(dimensions)),
                      false)
end

# Extract grid for an output. Field-like outputs delegate to `grid(...)`; non-field
# outputs return nothing.
output_grid(field::AbstractField)                            = grid(field)
output_grid(wta::WindowedTimeAverage{<:AbstractField})       = grid(wta.operand)
output_grid(other)                                           = nothing

#####
##### Initialization
#####

"""
    initialize!(writer::ZarrWriter, model)

Create the Zarr store on disk (or in the user-supplied store) and create one Zarr
array per output plus a 1D growing `time` array.
"""
function initialize!(writer::ZarrWriter, model)
    writer.initialized && return nothing

    distributed = is_distributed_arch(model)
    is_root = !distributed || mpi_rank(global_communicator()) == 0
    starting_fresh = model.clock.iteration == 0

    # Particles under MPI: not supported in v1.
    if distributed
        for (n, out) in pairs(writer.outputs)
            if out isa LagrangianParticles
                throw(ArgumentError("ZarrWriter under MPI does not support LagrangianParticles outputs (saw `$n`)."))
            end
        end
    end

    # Overwrite removes the existing directory (root only).
    if writer.overwrite_existing && starting_fresh && writer.store isa Zarr.DirectoryStore
        if is_root && !isempty(writer.filepath) && isdir(writer.filepath)
            rm(writer.filepath; recursive=true, force=true)
        end
        zarr_barrier()
        writer.store = Zarr.DirectoryStore(writer.filepath)
    end

    # `initialize_zarr_store!` may call MPI collectives (concatenate_local_sizes) when
    # the model is distributed, so *all* ranks must enter it together. The Zarr write
    # primitives inside are root-gated.
    if zarr_store_has_group_safely(writer, is_root)
        if is_root
            validate_existing_zarr_store(writer)
        end
        zarr_barrier()
    else
        initialize_zarr_store!(writer, model)
        zarr_barrier()
    end

    writer.initialized = true
    return nothing
end

# Root checks the filesystem; result is broadcast to other ranks via barrier semantics
# (no actual MPI broadcast — just both ranks ask the FS and we trust they see the same
# thing after the rm step's barrier). For simplicity we ask every rank, since reading
# `.zgroup` is cheap and idempotent.
zarr_store_has_group_safely(writer, is_root) = zarr_store_has_group(writer.store)

#####
##### Distributed helpers
#####

is_distributed_arch(model) = architecture(model) isa Distributed
zarr_barrier()             = mpi_initialized() && (global_barrier(); nothing)

# Cheap optional trace; gated on env var.
function _ztrace(msg::AbstractString)
    get(ENV, "OCEANANIGANS_ZARR_TRACE", "0") == "1" || return nothing
    r = mpi_initialized() ? mpi_rank(global_communicator()) : 0
    println("[zarr rank $r] $msg")
    flush(stdout)
    return nothing
end

# Per-axis offsets (0-based) for this rank's slab within the global field.
function rank_global_offsets(field::AbstractField)
    arch = architecture(field.grid)
    if !(arch isa Distributed)
        return ntuple(_ -> 0, length(size(field)))
    end
    local_sz = size(field)
    per_axis_locals = concatenate_local_sizes(local_sz, arch)
    li = arch.local_index
    return ntuple(d -> sum(@view(per_axis_locals[d][1:li[d]-1])), length(local_sz))
end

# Global shape of a Field on a (possibly distributed) grid.
function global_field_size(field::AbstractField)
    arch = architecture(field.grid)
    arch isa Distributed || return size(field)
    local_sz = size(field)
    per_axis_locals = concatenate_local_sizes(local_sz, arch)
    return map(sum, per_axis_locals)
end

# GCD of per-rank local extents along each axis. Used as the default chunk shape
# for distributed writers; ensures every rank's slab is chunk-aligned.
function distributed_gcd_chunks(field::AbstractField)
    arch = architecture(field.grid)
    arch isa Distributed || return size(field)
    local_sz = size(field)
    per_axis_locals = concatenate_local_sizes(local_sz, arch)
    return map(t -> Int(gcd(t...)), per_axis_locals)
end

"""
    zarr_store_has_group(store) -> Bool

True if the store already contains a Zarr v2 group (i.e. a `.zgroup` key at root).
"""
zarr_store_has_group(store) = Zarr.is_zgroup(Zarr.ZarrFormat(Val(2)), store, "")

"""
    validate_existing_zarr_store(writer)

Open the existing Zarr store and check that each output's dtype matches what
`array_type` would produce. Errors with a clear message on mismatch.
"""
function validate_existing_zarr_store(writer::ZarrWriter)
    g = Zarr.zopen(writer.store, "w")
    FT_requested = eltype(writer.array_type)
    for (name, _) in pairs(writer.outputs)
        name_str = string(name)
        haskey(g.arrays, name_str) || continue
        arr_eltype = eltype(g[name_str])
        if arr_eltype !== FT_requested && !(arr_eltype === Int8 && FT_requested === Bool)
            throw(ArgumentError(
                "Zarr array `$name_str` in $(writer.filepath) has dtype $arr_eltype " *
                "but `array_type` requests $FT_requested. " *
                "Re-use the original `array_type` or pass `overwrite_existing=true`."))
        end
    end
    return nothing
end

function initialize_zarr_store!(writer::ZarrWriter, model)
    arch = architecture(model)
    distributed = arch isa Distributed
    is_root = !distributed || mpi_rank(global_communicator()) == 0
    _ztrace("initialize_zarr_store! enter")

    # Record rank topology at root level for restart validation.
    root_attrs = Dict{String, Any}()
    if distributed
        root_attrs["rank_topology"] = [arch.ranks[1], arch.ranks[2], arch.ranks[3]]
    end

    # Compute the "global" grids that we want to serialize. For distributed grids,
    # `reconstruct_global_grid` is a collective — all ranks must call it. For non-
    # distributed grids it's an identity.
    serialize_grids = Tuple(
        grid isa DistributedGrid ?
            Oceananigans.DistributedComputations.reconstruct_global_grid(grid) : grid
        for grid in writer.grids
    )
    _ztrace("computed serialize_grids")

    if is_root
        _ztrace("zgroup")
        g = Zarr.zgroup(writer.store; attrs=root_attrs)

        time_attrs = Dict{String, Any}(
            "_ARRAY_DIMENSIONS" => ["time"],
            "units"             => "seconds",
            "long_name"         => "Time",
        )
        _ztrace("zcreate time")
        Zarr.zcreate(Float64, g, "time", 0; chunks=(1,), attrs=time_attrs)
        _ztrace("grid reconstruction")
        write_zarr_grid_reconstruction!(g, serialize_grids)
    else
        g = nothing
    end

    for (name, output) in pairs(writer.outputs)
        _ztrace("define_zarr_output_variable! `$name` start")
        define_zarr_output_variable!(g, writer, output, string(name), model)
        _ztrace("define_zarr_output_variable! `$name` done")
    end
    _ztrace("initialize_zarr_store! return")

    return nothing
end

#####
##### Grid reconstruction
#####

# JSON-friendly conversion for grid constructor args. Uses OrderedDict so positional
# arguments survive the round-trip in their declared order.
convert_for_zarr(dict::AbstractDict) = OrderedDict{String, Any}(string(k) => convert_for_zarr(v) for (k, v) in dict)
convert_for_zarr(x::Number)         = x
convert_for_zarr(x::Bool)           = string(x)
convert_for_zarr(x::NTuple{N, Number}) where N = collect(x)
convert_for_zarr(::CPU)             = "CPU()"
convert_for_zarr(::GPU)             = "GPU()"
# A Distributed arch is not serializable in a portable way; record a placeholder.
# The reader takes `architecture` as a kwarg and substitutes it in via the
# `args_ordered` override in `reconstruct_zarr_grid`.
convert_for_zarr(::Distributed)     = "CPU()"
convert_for_zarr(x)                 = string(x)

# Inverse: parse JSON-encoded values back to Julia. Strings get Meta.parse'd + eval'd.
# Returns OrderedDict to preserve positional argument order on reconstruct.
materialize_from_zarr(dict::AbstractDict) = OrderedDict{Symbol, Any}(Symbol(k) => materialize_from_zarr(v) for (k, v) in dict)
materialize_from_zarr(x::Number)          = x
materialize_from_zarr(x::AbstractArray)   = Tuple(x)
materialize_from_zarr(x::AbstractString)  = @eval $(Meta.parse(x))
materialize_from_zarr(x)                  = x

zarr_grid_type_string(g) = string(typeof(g).name.wrapper)

function zarr_grid_constructor_info(grid)
    args, kwargs = constructor_arguments(grid)
    metadata = Dict(:underlying_grid_type   => zarr_grid_type_string(grid),
                    :immersed_boundary_type => nothing)
    return args, kwargs, metadata
end

function zarr_grid_constructor_info(grid::ImmersedBoundaryGrid)
    underlying_args, underlying_kwargs, _ = constructor_arguments(grid)
    metadata = Dict(:underlying_grid_type   => zarr_grid_type_string(grid.underlying_grid),
                    :immersed_boundary_type => zarr_grid_type_string(grid.immersed_boundary))
    # Immersed boundary fields (mask, bottom_height) need data write; deferred to a
    # follow-on phase. For now the grid still serializes the underlying portion.
    return underlying_args, underlying_kwargs, metadata
end

function write_zarr_grid_reconstruction!(root_group, grids)
    single_grid = length(grids) <= 1
    for (i, grid) in enumerate(grids)
        subgroup_name = single_grid ? "grid" : "grid_$i"
        write_one_grid_reconstruction!(root_group, grid, subgroup_name)
    end
    return nothing
end

function write_one_grid_reconstruction!(root_group, grid, subgroup_name)
    args, kwargs, metadata = zarr_grid_constructor_info(grid)

    # Positional args: stored as a JSON array of [key, value] pairs so order survives
    # the round-trip through JSON (Zarr.jl parses attrs with `dicttype=Dict{String,Any}`
    # which does not preserve insertion order).
    args_json     = [[string(k), convert_for_zarr(v)] for (k, v) in pairs(args)]
    kwargs_json   = convert_for_zarr(kwargs)
    metadata_json = convert_for_zarr(metadata)

    attrs = Dict{String, Any}(
        "underlying_grid_reconstruction_args"   => args_json,
        "underlying_grid_reconstruction_kwargs" => kwargs_json,
        "grid_reconstruction_metadata"          => metadata_json,
    )
    Zarr.zgroup(root_group, subgroup_name; attrs=attrs)
    return nothing
end

"""
    reconstruct_zarr_grid(group; grid_index=1, architecture=nothing)

Read a grid back from a Zarr group written by `ZarrWriter`. Looks for `grid/` (single)
then `grid_<index>/` (multi). Reconstructs `RectilinearGrid`/`LatitudeLongitudeGrid` from
the serialized constructor arguments. Immersed-boundary reconstruction not implemented
in v1.
"""
function reconstruct_zarr_grid(group; grid_index=1, architecture=nothing)
    subgroup_name = "grid" in keys(group.groups) ? "grid" : "grid_$grid_index"
    haskey(group.groups, subgroup_name) ||
        throw(ArgumentError("No grid reconstruction data found in this Zarr group (looked for `$subgroup_name`)."))

    subgroup = group.groups[subgroup_name]
    attrs = subgroup.attrs

    # Positional args: list of [key, value] pairs (preserves order across JSON).
    args_pairs  = attrs["underlying_grid_reconstruction_args"]
    args_ordered = [(Symbol(p[1]), materialize_from_zarr(p[2])) for p in args_pairs]
    if !isnothing(architecture)
        # Override architecture entry with the user-supplied one.
        args_ordered = [(k === :architecture ? :architecture => architecture : k => v)
                        for (k, v) in args_ordered]
    end
    args_values = [v for (_, v) in args_ordered]
    kwargs_dict = materialize_from_zarr(attrs["underlying_grid_reconstruction_kwargs"])
    metadata    = materialize_from_zarr(attrs["grid_reconstruction_metadata"])

    grid_type = metadata[:underlying_grid_type]
    underlying = grid_type(args_values...; kwargs_dict...)

    if isnothing(metadata[:immersed_boundary_type])
        return underlying
    else
        throw(ArgumentError("Immersed-boundary reconstruction not yet implemented for Zarr."))
    end
end


"""
    define_zarr_output_variable!(g, writer, output, name, model)

Create a Zarr array for `output` under group `g`, with shape `(spatial_dims..., 0)` and
chunk `(spatial_chunks..., 1)`. Time grows along the last axis.

Dispatched on output type:
  - `AbstractField`: uses field location + grid to name dims via `trilocation_dim_name`.
  - `WindowedTimeAverage{<:AbstractField}`: delegates to the wrapped field.
  - Anything else (functions, custom callables): requires `writer.dimensions[name]` to
    supply a tuple of dimension names.
"""
function define_zarr_output_variable!(g, writer::ZarrWriter, output::AbstractField, name, model)
    arch = architecture(output.grid)
    distributed = arch isa Distributed

    # On-disk shape and chunk default depend on whether we're distributed.
    if distributed
        _ztrace("about to call global_field_size for $name (collective)")
        spatial_size = global_field_size(output)
        _ztrace("got spatial_size=$spatial_size; about to call distributed_gcd_chunks (collective)")
        gcd_chunks   = distributed_gcd_chunks(output)
        _ztrace("got gcd_chunks=$gcd_chunks")
    else
        sample = fetch_and_convert_output(output, model, writer)
        spatial_size = size(sample)
        gcd_chunks   = spatial_size
    end
    FT = eltype(writer.array_type)

    # Chunk shape: user override > distributed-GCD default > full extent.
    spatial_chunks = if isnothing(writer.chunks)
        gcd_chunks
    else
        length(writer.chunks) == length(spatial_size) ? writer.chunks :
        length(writer.chunks) == length(spatial_size) + 1 ? writer.chunks[1:end-1] :
        throw(ArgumentError("`chunks` length $(length(writer.chunks)) doesn't match field rank $(length(spatial_size))"))
    end

    # Warn (root only) if the GCD-derived chunk gives a pathologically small chunk
    # count for distributed runs — usually indicates accidental coprime decomposition.
    # All ranks call concatenate_local_sizes (it's a collective), but only root emits.
    if distributed && isnothing(writer.chunks)
        per_axis_locals_for_warn = concatenate_local_sizes(size(output), arch)
        if mpi_rank(global_communicator()) == 0
            for d in 1:length(spatial_size)
                n_chunks = spatial_size[d] ÷ max(spatial_chunks[d], 1)
                n_ranks_d = length(per_axis_locals_for_warn[d])
                if n_chunks > 10 * n_ranks_d
                    @warn "ZarrWriter: axis $d would produce $n_chunks chunks/timestep across $n_ranks_d ranks (GCD-default). Consider passing an explicit `chunks` kwarg or rebalancing the decomposition."
                end
            end
        end
    end

    shape  = (spatial_size...,    0)
    chunks = (spatial_chunks..., 1)

    # _ARRAY_DIMENSIONS in C order (reverse of Julia order, then `time` last in Julia
    # but first in C order — i.e., the reversed sequence ends in the slowest-varying axis).
    spatial_dim_names = zarr_field_dimensions(output)
    julia_order_dims  = String[spatial_dim_names..., "time"]
    array_dimensions  = reverse(julia_order_dims)

    # For distributed runs, the field's local `indices` reflect this rank's slab and
    # are not meaningful on disk (the on-disk array is the GLOBAL field). Save the
    # default `(:, :, :)` indices in that case.
    indices_for_attr = distributed ?
        [":", ":", ":"] :
        collect(string.(indices_strings(output)))

    attrs = Dict{String, Any}(
        "_ARRAY_DIMENSIONS" => array_dimensions,
        "location"          => collect(string.(location_strings(output))),
        "indices"           => indices_for_attr,
    )

    # Tag with grid_index when there are multiple grids (1-based; matches NetCDFWriter).
    if length(writer.grids) > 1
        gi = get(writer.output_grid_map, name, nothing)
        isnothing(gi) || (attrs["grid_index"] = gi)
    end

    compressor = isnothing(writer.compressor) ? Zarr.NoCompressor() : writer.compressor

    # Only the root rank persists the array; under distributed writes the other ranks
    # have already participated in the collectives above (via global_field_size etc.)
    # and need not touch the store at this point.
    if !isnothing(g)
        Zarr.zcreate(FT, g, name, shape...;
                     chunks=chunks,
                     compressor=compressor,
                     attrs=attrs)
    end
    return nothing
end

# WindowedTimeAverage over a Field: delegate to operand (matches NetCDFWriter).
define_zarr_output_variable!(g, writer::ZarrWriter, output::WindowedTimeAverage{<:AbstractField}, name, model) =
    define_zarr_output_variable!(g, writer, output.operand, name, model)

# Function / generic custom output: requires `writer.dimensions[name]` to be set.
function define_zarr_output_variable!(g, writer::ZarrWriter, output, name, model)
    if !haskey(writer.dimensions, name)
        throw(ArgumentError(
            "`dimensions[$name]` must be provided when constructing ZarrWriter for " *
            "non-field output $name = $(typeof(output))"))
    end

    sample = fetch_and_convert_output(output, model, writer)
    sample_arr = sample isa AbstractArray ? sample : fill(sample)
    spatial_size = size(sample_arr)
    FT = eltype(writer.array_type)

    user_dims = collect(string.(writer.dimensions[name]))
    if length(user_dims) != length(spatial_size)
        throw(ArgumentError(
            "`dimensions[$name]` has $(length(user_dims)) entries but the output has " *
            "$(length(spatial_size)) dimensions"))
    end

    spatial_chunks = if isnothing(writer.chunks)
        spatial_size
    elseif length(writer.chunks) == length(spatial_size)
        writer.chunks
    elseif length(writer.chunks) == length(spatial_size) + 1
        writer.chunks[1:end-1]
    else
        throw(ArgumentError("`chunks` length doesn't match output `$name` rank"))
    end

    shape  = (spatial_size...,    0)
    chunks = (spatial_chunks..., 1)
    array_dimensions = reverse(String[user_dims..., "time"])

    attrs = Dict{String, Any}("_ARRAY_DIMENSIONS" => array_dimensions)

    compressor = isnothing(writer.compressor) ? Zarr.NoCompressor() : writer.compressor

    if !isnothing(g)
        Zarr.zcreate(FT, g, name, shape...;
                     chunks=chunks,
                     compressor=compressor,
                     attrs=attrs)
    end
    return nothing
end

#####
##### Per-step write
#####

function write_output!(writer::ZarrWriter, model::AbstractModel)
    distributed = is_distributed_arch(model)
    is_root = !distributed || mpi_rank(global_communicator()) == 0

    # File splitting (root-only when distributed; everyone else waits)
    if writer.file_splitting(model)
        if is_root
            start_next_file(model, writer)
        end
        zarr_barrier()
    end
    update_file_splitting_schedule!(writer.file_splitting, writer.filepath)

    if !writer.initialized
        initialize!(writer, model)
    end

    if distributed
        write_output_distributed!(writer, model)
    else
        write_output_serial!(writer, model)
    end

    return nothing
end

function write_output_serial!(writer::ZarrWriter, model)
    g = Zarr.zopen(writer.store, "w")
    Zarr.append!(g["time"], Float64[Float64(model.clock.time)]; dims=1)
    for (name, output) in pairs(writer.outputs)
        data = fetch_and_convert_output(output, model, writer)
        arr = g[string(name)]
        data_arr = data isa AbstractArray ? data : fill(data)
        if eltype(data_arr) === Bool
            data_arr = Int8.(data_arr)
        end
        Zarr.append!(arr, data_arr; dims=ndims(arr))
    end
    return nothing
end

# Distributed write:
#   1. All ranks open the store (read-write).
#   2. Each rank resizes its in-memory handle so the time axis grows by 1.
#      Persisting the new `.zarray` is rank-0 only — all ranks would otherwise race on
#      the same JSON file. (`writemetadata` inside Zarr.resize! happens once per call;
#      we suppress it on non-root ranks by mutating the metadata directly.)
#   3. Each rank writes its local slab to its global slice of the array.
#   4. Rank 0 also writes the time chunk.
#   5. Barrier so all writes land before the next step starts.
function write_output_distributed!(writer::ZarrWriter, model)
    is_root = mpi_rank(global_communicator()) == 0
    g = Zarr.zopen(writer.store, "w")

    # Bump the time axis and write the new time value (root only).
    if is_root
        Zarr.append!(g["time"], Float64[Float64(model.clock.time)]; dims=1)
    end
    zarr_barrier()

    # Re-open after metadata change so all ranks see the new time array length.
    g = Zarr.zopen(writer.store, "w")
    new_time_index = length(g["time"])

    for (name, output) in pairs(writer.outputs)
        arr = g[string(name)]
        data = fetch_and_convert_output(output, model, writer)
        data_arr = data isa AbstractArray ? data : fill(data)
        if eltype(data_arr) === Bool
            data_arr = Int8.(data_arr)
        end

        # Bump the array's time axis by 1. On root, persist; on others, just update
        # the in-memory metadata so subsequent slice writes know the new shape.
        old_shape = size(arr)
        new_shape = ntuple(d -> d == length(old_shape) ? new_time_index : old_shape[d], length(old_shape))
        if is_root
            Zarr.resize!(arr, new_shape)
        else
            arr.metadata.shape[] = new_shape
        end

        # Compute this rank's global slice (0-based offsets) and write the slab.
        offsets = rank_global_offsets(output)
        spatial_slices = ntuple(d -> (offsets[d]+1):(offsets[d]+size(data_arr, d)), length(offsets))
        full_slices = (spatial_slices..., new_time_index:new_time_index)

        # Reshape data to include singleton time dim, matching the global array's rank.
        data_with_time = reshape(data_arr, size(data_arr)..., 1)
        arr[full_slices...] = data_with_time
    end

    zarr_barrier()
    return nothing
end

#####
##### File splitting
#####

function start_next_file(model, writer::ZarrWriter)
    writer.verbose && @info "Splitting Zarr output because $(summary(writer.file_splitting)) is activated."

    if writer.part == 1
        part1_path = replace(writer.filepath, r".zarr$" => "_part1.zarr")
        writer.verbose && @info "Renaming first part: $(writer.filepath) -> $part1_path"
        if isdir(writer.filepath)
            mv(writer.filepath, part1_path; force=writer.overwrite_existing)
        end
        writer.filepath = part1_path
    end

    writer.part += 1
    writer.filepath = replace(writer.filepath, r"part\d+.zarr$" => "part" * string(writer.part) * ".zarr")
    if writer.overwrite_existing && isdir(writer.filepath)
        rm(writer.filepath; recursive=true, force=true)
    end
    writer.store = Zarr.DirectoryStore(writer.filepath)
    writer.verbose && @info "Now writing to: $(writer.filepath)"

    initialize_zarr_store!(writer, model)
    return nothing
end

#####
##### Helpers: field dim names, location strings, indices strings
#####

zarr_field_dimensions(field::AbstractField) = zarr_field_dimensions(field, field.grid)

function zarr_field_dimensions(field::AbstractField, grid::RectilinearGrid)
    LX, LY, LZ = location(field)
    name_x = LX === Nothing ? "" : trilocation_dim_name("x", grid, LX(), nothing, nothing, Val(:x))
    name_y = LY === Nothing ? "" : trilocation_dim_name("y", grid, nothing, LY(), nothing, Val(:y))
    name_z = LZ === Nothing ? "" : trilocation_dim_name("z", grid, nothing, nothing, LZ(), Val(:z))
    return (name_x, name_y, name_z)
end

function zarr_field_dimensions(field::AbstractField, grid::LatitudeLongitudeGrid)
    LΛ, LΦ, LZ = location(field)
    name_λ = LΛ === Nothing ? "" : trilocation_dim_name("λ", grid, LΛ(), nothing, nothing, Val(:x))
    name_φ = LΦ === Nothing ? "" : trilocation_dim_name("φ", grid, nothing, LΦ(), nothing, Val(:y))
    name_z = LZ === Nothing ? "" : trilocation_dim_name("z", grid, nothing, nothing, LZ(), Val(:z))
    return (name_λ, name_φ, name_z)
end

function zarr_field_dimensions(field::AbstractField, grid::OrthogonalSphericalShellGrid)
    LΛ, LΦ, LZ = location(field)
    name_λ = LΛ === Nothing ? "" : trilocation_dim_name("λ", grid, LΛ(), nothing, nothing, Val(:x))
    name_φ = LΦ === Nothing ? "" : trilocation_dim_name("φ", grid, nothing, LΦ(), nothing, Val(:y))
    name_z = LZ === Nothing ? "" : trilocation_dim_name("z", grid, nothing, nothing, LZ(), Val(:z))
    return (name_λ, name_φ, name_z)
end

zarr_field_dimensions(field::AbstractField, grid::ImmersedBoundaryGrid) =
    zarr_field_dimensions(field, grid.underlying_grid)

# Location and indices as JSON-friendly String tuples.
location_strings(field::AbstractField) = map(loc -> loc === Nothing ? "Nothing" : string(loc),
                                             location(field))

indices_strings(field::AbstractField) = map(index_string, indices(field))
index_string(::Colon) = ":"
index_string(r::AbstractUnitRange) = string(first(r), ":", last(r))
index_string(i::Integer) = string(i)



Base.summary(ow::ZarrWriter) =
    string("ZarrWriter writing ", prettykeys(ow.outputs), " to ", ow.filepath, " on ", summary(ow.schedule))

function Base.show(io::IO, ow::ZarrWriter)
    averaging_schedule = output_averaging_schedule(ow)
    Noutputs = length(ow.outputs)

    store_str = isnothing(ow.store) ? "DirectoryStore (deferred)" : string(typeof(ow.store).name.name)
    compressor_str = isnothing(ow.compressor) ? "none" : string(typeof(ow.compressor).name.name)
    chunks_str = isnothing(ow.chunks) ? "auto" : string(ow.chunks)

    print(io, "ZarrWriter scheduled on $(summary(ow.schedule)):", "\n",
              "├── filepath: ", ow.filepath, "\n",
              "├── store: ", store_str, "\n",
              "├── $Noutputs outputs: ", prettykeys(ow.outputs), show_averaging_schedule(averaging_schedule), "\n",
              "├── array_type: ", show_array_type(ow.array_type), "\n",
              "├── chunks: ", chunks_str, "\n",
              "├── compressor: ", compressor_str, "\n",
              "└── file_splitting: ", summary(ow.file_splitting))
end
