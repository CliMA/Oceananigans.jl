using JLD2
using Glob
using Statistics: mean
using GPUArraysCore: @allowscalar

using Oceananigans.Grids: RectilinearGrid, LatitudeLongitudeGrid,
                          cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z,
                          topology, size, halo_size, generate_coordinate,
                          with_precomputed_metrics, metrics_precomputed,
                          Periodic, Bounded, FullyConnected

using Oceananigans.Fields: interior, Field, instantiated_location, FixedTime
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.DistributedComputations: reconstruct_global_topology

#####
##### DistributedPath - wrapper for path that includes rank file info
#####

"""
    DistributedPath(rank_infos)

A wrapper for the `path` field that stores rank file information for combining 
distributed output. This allows dispatching on the path type to enable combining
for both InMemory and OnDisk backends.

The `path` field semantically represents "where to find the data" - for distributed
output, the data lives in multiple rank files.
"""
struct DistributedPath{R}
    rank_infos :: R
end

Base.show(io::IO, dp::DistributedPath) = print(io, "DistributedPath(", length(dp.rank_infos), " ranks)")

# Convenience accessor for the first rank's path (used for metadata)
first_path(dp::DistributedPath) = first(dp.rank_infos).path
first_path(path::String) = path

#####
##### Finding and loading rank files
#####

"""
    find_rank_files(path)

Given a file path, check if distributed rank files exist (e.g., `output_rank0.jld2`, `output_rank1.jld2`).
Returns a sorted vector of rank file paths if they exist, or `nothing` if not.
"""
function find_rank_files(path)
    path = auto_extension(path, ".jld2")
    prefix = path[1:end-5]
    rank_paths = glob(string(prefix, "_rank*.jld2"))
    isempty(rank_paths) && return nothing
    return naturalsort(rank_paths)
end

struct RankFileInfo{G, I}
    grid :: G
    local_index :: I
    ranks :: NTuple{3, Int}
    path :: String
end

"""Load grid and rank information from a distributed output file."""
function load_rank_file_info(path; reader_kw=NamedTuple())
    grid = jldopen(path; reader_kw...) do file
        file["serialized/grid"]
    end
    
    arch = try architecture(grid) catch; return nothing end
    hasproperty(arch, :local_index) && hasproperty(arch, :ranks) || return nothing
    return RankFileInfo(grid, arch.local_index, arch.ranks, path)
end

#####
##### Offline grid reconstruction from rank files
#####
#
# This mirrors the logic in DistributedComputations.reconstruct_global_grid,
# but works offline from JLD2 files rather than using MPI communication.
# The grid building pattern is intentionally similar to maintain consistency.
#

# Compute cumulative offsets for indexing into the global array
function rank_offsets(local_sizes, ranks, dim)
    R = ranks[dim]
    sizes = [local_sizes[ntuple(d -> d == dim ? r : 1, 3)][dim] for r in 1:R-1]
    return cumsum([0; sizes])
end

# Collect and concatenate coordinate data from all ranks along a dimension
function collect_global_coordinates(rank_infos, dim, coord_func)
    R = first(rank_infos).ranks[dim]
    coords = map(1:R) do r
        selector = ntuple(d -> d == dim ? r : 1, 3)
        info = first(filter(i -> i.local_index == selector, rank_infos))
        c = coord_func(info.grid)
        (c isa AbstractVector && r < R) ? c[1:end-1] : c  # Avoid overlap
    end
    
    all(c isa Tuple for c in coords) && return (first(coords[1]), last(coords[end]))
    return vcat(coords...)
end

"""
    reconstruct_global_grid_from_ranks(rank_infos, arch)

Reconstruct a global grid from distributed rank file information.
This is the offline (file-based) equivalent of `DistributedComputations.reconstruct_global_grid`.
"""
function reconstruct_global_grid_from_ranks(rank_infos, arch)
    ranks = first(rank_infos).ranks
    Rx, Ry, Rz = ranks
    Nr = prod(ranks)
    
    length(rank_infos) == Nr || error("Expected $Nr rank files but found $(length(rank_infos))")
    
    grid0 = first(filter(i -> i.local_index == (1, 1, 1), rank_infos)).grid
    local_sizes = Dict(info.local_index => size(info.grid) for info in rank_infos)
    
    # Compute global size (cf. DistributedComputations.global_size)
    Nx = sum(local_sizes[(ri, 1, 1)][1] for ri in 1:Rx)
    Ny = sum(local_sizes[(1, rj, 1)][2] for rj in 1:Ry)
    Nz = local_sizes[(1, 1, 1)][3]
    
    H = halo_size(grid0)
    FT = eltype(grid0)
    
    # Reconstruct topology using the offline utility from DistributedComputations
    all_topos = [[topology(info.grid)[d] for info in rank_infos] for d in 1:3]
    topo = Tuple(reconstruct_global_topology(all_topos[d], ranks[d]) for d in 1:3)
    
    # Collect global coordinates (cf. DistributedComputations.assemble_coordinate)
    xG = collect_global_coordinates(rank_infos, 1, cpu_face_constructor_x)
    yG = collect_global_coordinates(rank_infos, 2, cpu_face_constructor_y)
    zG = cpu_face_constructor_z(grid0)
    
    return build_global_grid(grid0, arch, FT, topo, (Nx, Ny, Nz), H, xG, yG, zG)
end

# Grid building dispatched by grid type
# (Mirrors the pattern in DistributedComputations.reconstruct_global_grid)

function build_global_grid(grid0::RectilinearGrid, arch, FT, topo, N, H, xG, yG, zG)
    TX, TY, TZ = topo
    Nx, Ny, Nz = N
    Hx, Hy, Hz = H
    
    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶠᵃᵃ, Δxᶜᵃᵃ = generate_coordinate(FT, TX(), Nx, Hx, xG, :x, arch)
    Ly, yᵃᶠᵃ, yᵃᶜᵃ, Δyᵃᶠᵃ, Δyᵃᶜᵃ = generate_coordinate(FT, TY(), Ny, Hy, yG, :y, arch)
    Lz, z = generate_coordinate(FT, topo, N, H, zG, :z, 3, arch)
    
    return RectilinearGrid{TX, TY, TZ}(arch, Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz,
                                        Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ,
                                        Δyᵃᶠᵃ, Δyᵃᶜᵃ, yᵃᶠᵃ, yᵃᶜᵃ, z)
end

function build_global_grid(grid0::LatitudeLongitudeGrid, arch, FT, topo, N, H, xG, yG, zG)
    TX, TY, TZ = topo
    Nx, Ny, Nz = N
    Hx, Hy, Hz = H
    
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, TX(), Nx, Hx, xG, :longitude, arch)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, TY(), Ny, Hy, yG, :latitude, arch)
    Lz, z = generate_coordinate(FT, topo, N, H, zG, :z, 3, arch)
    
    grid = LatitudeLongitudeGrid{TX, TY, TZ}(arch, Nx, Ny, Nz, Hx, Hy, Hz, Lλ, Lφ, Lz,
                                              Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                              Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ, z,
                                              (nothing for _ in 1:10)..., grid0.radius)
    
    return metrics_precomputed(grid0) ? with_precomputed_metrics(grid) : grid
end

function build_global_grid(grid0, arch, FT, topo, N, H, xG, yG, zG)
    grid_type = typeof(grid0).name.wrapper
    error("Automatic combining of distributed output is not yet supported for $grid_type. " *
          "Consider using OceanEnsembles.jl or pass a pre-constructed global `grid` to FieldTimeSeries.")
end

#####
##### Loading and combining field data
#####

"""Load and combine field data from rank files into a field."""
function load_combined_field_data!(field, rank_infos, name, iter; reader_kw=NamedTuple())
    ranks = first(rank_infos).ranks
    local_sizes = Dict(info.local_index => size(info.grid) for info in rank_infos)
    x_offsets = rank_offsets(local_sizes, ranks, 1)
    y_offsets = rank_offsets(local_sizes, ranks, 2)
    
    field_data = interior(field)
    
    for info in rank_infos
        ri, rj, _ = info.local_index
        nx, ny, nz = size(info.grid)
        Hx, Hy, Hz = halo_size(info.grid)
        
        raw_data = jldopen(info.path; reader_kw...) do file
            file["timeseries/$name/$iter"]
        end
        
        # Extract interior (remove halos) and copy to global array
        interior_data = @view raw_data[Hx+1:Hx+nx, Hy+1:Hy+ny, Hz+1:Hz+nz]
        field_data[x_offsets[ri]+1:x_offsets[ri]+nx,
                   y_offsets[rj]+1:y_offsets[rj]+ny,
                   1:nz] .= interior_data
    end
    
    return nothing
end

#####
##### Main entry point
#####

"""Create a FieldTimeSeries by combining data from distributed rank files."""
function combined_field_time_series(path, name;
                                     backend = InMemory(),
                                     architecture = nothing,
                                     grid = nothing,
                                     location = nothing,
                                     boundary_conditions = UnspecifiedBoundaryConditions(),
                                     time_indexing = Linear(),
                                     iterations = nothing,
                                     times = nothing,
                                     reader_kw = NamedTuple())
    
    rank_paths = find_rank_files(path)
    isnothing(rank_paths) && error("No rank files found for path: $path")
    
    rank_infos = filter(!isnothing, [load_rank_file_info(p; reader_kw) for p in rank_paths])
    isempty(rank_infos) && error("Could not load distributed grid info from rank files.")
    
    sort!(rank_infos, by = i -> (i.local_index[3], i.local_index[2], i.local_index[1]))
    
    architecture = something(architecture, isnothing(grid) ? CPU() : Architectures.architecture(grid))
    isnothing(grid) && (grid = reconstruct_global_grid_from_ranks(rank_infos, architecture))
    
    # Read metadata from first rank file
    metadata_path = first(rank_paths)
    file = jldopen(metadata_path; reader_kw...)
    
    indices = try file["timeseries/$name/serialized/indices"] catch; (:, :, :) end
    isnothing(location) && (location = file["timeseries/$name/serialized/location"])
    LX, LY, LZ = location
    loc = (LX(), LY(), LZ())
    
    # Note: We intentionally do NOT load boundary conditions from distributed files.
    # The distributed files contain DistributedCommunicationBoundaryCondition which are 
    # not valid for the reconstructed global grid. Default BCs will be used instead.
    if boundary_conditions isa UnspecifiedBoundaryConditions
        boundary_conditions = nothing
    end
    
    isnothing(iterations) && (iterations = parse.(Int, keys(file["timeseries/t"])))
    isnothing(times) && (times = [file["timeseries/t/$i"] for i in iterations])
    close(file)
    
    # Use DistributedPath to store rank_infos - this enables dispatch for both backends
    distributed_path = DistributedPath(rank_infos)
    
    # Create FieldTimeSeries
    Nt = time_indices_length(backend, times)
    @apply_regionally data = new_data(eltype(grid), grid, loc, indices, Nt)
    
    fts = FieldTimeSeries{LX, LY, LZ}(data, grid, backend, boundary_conditions, indices,
                                       times, distributed_path, name, time_indexing, reader_kw)
    
    # For InMemory, load data now
    backend isa AbstractInMemoryBackend && set!(fts)
    
    return fts
end

#####
##### InMemory support - set! dispatches on DistributedPath
#####

"""Set FieldTimeSeries data by loading and combining from distributed rank files."""
function set!(fts::FieldTimeSeries{LX, LY, LZ, TI, K, I, D, G, ET, B, χ, <:DistributedPath}
             ) where {LX, LY, LZ, TI, K <: AbstractInMemoryBackend, I, D, G, ET, B, χ}
    
    rank_infos = fts.path.rank_infos
    metadata_path = first_path(fts.path)
    
    file = jldopen(metadata_path; fts.reader_kw...)
    file_iterations = iterations_from_file(file)
    file_times = [file["timeseries/t/$i"] for i in file_iterations]
    close(file)
    
    cpu_times = on_architecture(CPU(), fts.times)
    Δt = mean(diff(file_times))
    
    for n in time_indices(fts)
        file_index = find_time_index(cpu_times[n], file_times, Δt)
        if isnothing(file_index)
            @warn "No data found for time $(cpu_times[n]) and time index $n"
        else
            load_combined_field_data!(fts[n], rank_infos, fts.name, file_iterations[file_index];
                                       reader_kw=fts.reader_kw)
        end
    end
    
    return nothing
end

#####
##### OnDisk support - getindex dispatches on DistributedPath
#####

"""
    getindex(fts, n::Int)

Load and combine field data from distributed rank files at time index `n`.
This method dispatches when `fts.path isa DistributedPath` and `fts.backend isa OnDisk`.
"""
function Base.getindex(fts::FieldTimeSeries{LX, LY, LZ, TI, <:OnDisk, I, D, G, ET, B, χ, <:DistributedPath},
                       n::Int) where {LX, LY, LZ, TI, I, D, G, ET, B, χ}
    rank_infos = fts.path.rank_infos
    metadata_path = first_path(fts.path)
    
    # Get iteration key from first rank file
    file = jldopen(metadata_path; fts.reader_kw...)
    iter = keys(file["timeseries/t"])[n]
    close(file)
    
    # Create an empty field on the global grid
    loc = instantiated_location(fts)
    field = Field(loc, fts.grid;
                  indices = fts.indices,
                  boundary_conditions = fts.boundary_conditions)
    
    # Load and combine data from all rank files
    load_combined_field_data!(field, rank_infos, fts.name, iter; reader_kw=fts.reader_kw)
    
    # Set the field time status
    status = @allowscalar FixedTime(fts.times[n])
    field_with_time = Field(loc, fts.grid;
                            indices = fts.indices,
                            boundary_conditions = fts.boundary_conditions,
                            status,
                            data = field.data)
    
    fill_halo_regions!(field_with_time)
    
    return field_with_time
end
