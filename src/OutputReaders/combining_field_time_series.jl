using JLD2
using Glob
using Statistics: mean
using GPUArraysCore: @allowscalar

using Oceananigans.Grids: RectilinearGrid, LatitudeLongitudeGrid, OrthogonalSphericalShellGrid,
                          cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z,
                          topology, size, halo_size, generate_coordinate,
                          with_precomputed_metrics, metrics_precomputed

using Oceananigans.Fields: interior, Field, instantiated_location, FixedTime
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.DistributedComputations: reconstruct_global_topology
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid, Tripolar

#####
##### DistributedPaths - wrapper for path that includes rank file info
#####

"""
    DistributedPaths(ranks)

A wrapper for the `path` field that stores information about each rank's output file.
This enables dispatching for combining distributed output with both InMemory and OnDisk backends.

The `path` field semantically represents "where to find the data" - for distributed
output, the data lives across multiple rank files.
"""
struct DistributedPaths{R}
    ranks :: R  # Vector of RankOutputData, one per MPI rank
end

Base.show(io::IO, dp::DistributedPaths) = print(io, "DistributedPaths(", length(dp.ranks), " ranks)")

# Convenience accessor for the first rank's path (used for metadata)
first_path(dp::DistributedPaths) = first(dp.ranks).path
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

"""
    RankOutputData{G, I}

Data about a single MPI rank's output file, used for combining distributed output.

Fields:
- `local_grid`: The local grid for this rank (used for sizes, halos, coordinates)
- `local_index`: The rank's position in the decomposition, e.g., `(2, 1, 1)`
- `partition`: Total number of ranks in each dimension, e.g., `(2, 2, 1)`
- `path`: Path to this rank's JLD2 output file
"""
struct RankOutputData{G, I}
    local_grid :: G
    local_index :: I
    partition :: NTuple{3, Int}
    path :: String
end

"""Load grid and rank information from a distributed output file."""
function load_rank_data(path; reader_kw=NamedTuple())
    local_grid = jldopen(path; reader_kw...) do file
        file["serialized/grid"]
    end

    arch = try architecture(local_grid) catch; return nothing end
    hasproperty(arch, :local_index) && hasproperty(arch, :ranks) || return nothing
    return RankOutputData(local_grid, arch.local_index, arch.ranks, path)
end

#####
##### Offline grid reconstruction from rank files
#####
#
# Reconstructs a global grid from distributed JLD2 output files.
# This is the offline equivalent of DistributedComputations.reconstruct_global_grid.
#

"""
    get_rank(all_ranks, i, j, k=1)

Find the RankOutputData with local_index == (i, j, k).
"""
function get_rank(all_ranks, i, j, k=1)
    for rank in all_ranks
        rank.local_index == (i, j, k) && return rank
    end
    error("No rank found with local_index = ($i, $j, $k)")
end

"""
    compute_global_size(all_ranks)

Compute the global grid size by summing local sizes along each partitioned dimension.
"""
function compute_global_size(all_ranks)
    Rx, Ry, Rz = first(all_ranks).partition

    # Sum local sizes along x (first column of ranks)
    Nx = sum(size(get_rank(all_ranks, i, 1, 1).local_grid, 1) for i in 1:Rx)

    # Sum local sizes along y (first row of ranks)
    Ny = sum(size(get_rank(all_ranks, 1, j, 1).local_grid, 2) for j in 1:Ry)

    # z is not partitioned
    Nz = size(get_rank(all_ranks, 1, 1, 1).local_grid, 3)

    return (Nx, Ny, Nz)
end

"""
    compute_partition_offsets(all_ranks)

Compute cumulative offsets for placing each rank's data in the global array.
Returns (x_offsets, y_offsets) where offset[i] gives the starting index for rank i.
"""
function compute_partition_offsets(all_ranks)
    Rx, Ry, _ = first(all_ranks).partition

    # Compute x offsets: cumulative sum of local Nx values
    x_sizes = [size(get_rank(all_ranks, i, 1, 1).local_grid, 1) for i in 1:Rx]
    x_offsets = cumsum([0; x_sizes[1:end-1]])

    # Compute y offsets: cumulative sum of local Ny values
    y_sizes = [size(get_rank(all_ranks, 1, j, 1).local_grid, 2) for j in 1:Ry]
    y_offsets = cumsum([0; y_sizes[1:end-1]])

    return x_offsets, y_offsets
end

# Note: This is intentionally separate from DistributedComputations.assemble_coordinate,
# which uses MPI for live distributed coordination. This function works offline with
# pre-loaded file data and does not require MPI.
"""
    collect_global_coordinates_from_files(all_ranks, dim, coord_func)

Concatenate coordinate data from all ranks along dimension `dim`.
`coord_func` extracts coordinates from a grid (e.g., `cpu_face_constructor_x`).
"""
function collect_global_coordinates_from_files(all_ranks, dim, coord_func)
    Rx, Ry, _ = first(all_ranks).partition
    R = dim == 1 ? Rx : Ry

    coords = map(1:R) do r
        # Get the rank along this dimension (others fixed at 1)
        rank = dim == 1 ? get_rank(all_ranks, r, 1, 1) : get_rank(all_ranks, 1, r, 1)
        c = coord_func(rank.local_grid)
        # Drop last point to avoid overlap (except for last rank)
        (c isa AbstractVector && r < R) ? c[1:end-1] : c
    end

    # If all coordinates are tuples (start, end), return (first_start, last_end)
    if all(c isa Tuple for c in coords)
        return (first(coords[1]), last(coords[end]))
    end

    return vcat(coords...)
end

"""
    reconstruct_global_grid_from_files(all_ranks, arch)

Reconstruct a global grid from distributed rank output data.
This is the offline (file-based) equivalent of `DistributedComputations.reconstruct_global_grid`.
"""
function reconstruct_global_grid_from_files(all_ranks, arch)
    partition = first(all_ranks).partition
    Rx, Ry, Rz = partition
    Nr = prod(partition)

    length(all_ranks) == Nr || error("Expected $Nr rank files but found $(length(all_ranks))")

    # Get reference grid from rank (1,1,1)
    grid0 = get_rank(all_ranks, 1, 1, 1).local_grid

    # Compute global size
    Nx, Ny, Nz = compute_global_size(all_ranks)

    H = halo_size(grid0)
    FT = eltype(grid0)

    # Reconstruct global topology
    all_topos = [[topology(rd.local_grid)[d] for rd in all_ranks] for d in 1:3]
    topo = Tuple(reconstruct_global_topology(all_topos[d], partition[d]) for d in 1:3)

    # Collect global coordinates
    xG = collect_global_coordinates_from_files(all_ranks, 1, cpu_face_constructor_x)
    yG = collect_global_coordinates_from_files(all_ranks, 2, cpu_face_constructor_y)
    zG = cpu_face_constructor_z(grid0)

    return build_global_grid_from_files(grid0, arch, FT, topo, (Nx, Ny, Nz), H, xG, yG, zG)
end

# Grid building dispatched by grid type
# (Mirrors the pattern in DistributedComputations.reconstruct_global_grid)

function build_global_grid_from_files(grid0::RectilinearGrid, arch, FT, topo, N, H, xG, yG, zG)
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

function build_global_grid_from_files(grid0::LatitudeLongitudeGrid, arch, FT, topo, N, H, xG, yG, zG)
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

function build_global_grid_from_files(grid0::OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Tripolar},
                                      arch, FT, topo, N, H, xG, yG, zG)
    Nx, Ny, Nz = N
    Hx, Hy, Hz = H

    # Extract TripolarGrid-specific parameters from the reference grid
    conformal_mapping = grid0.conformal_mapping
    north_poles_latitude = conformal_mapping.north_poles_latitude
    first_pole_longitude = conformal_mapping.first_pole_longitude
    southernmost_latitude = conformal_mapping.southernmost_latitude

    return TripolarGrid(arch, FT;
                        size = (Nx, Ny, Nz),
                        halo = (Hx, Hy, Hz),
                        z = zG,
                        north_poles_latitude,
                        first_pole_longitude,
                        southernmost_latitude)
end

function build_global_grid_from_files(grid0, arch, FT, topo, N, H, xG, yG, zG)
    grid_type = typeof(grid0).name.wrapper
    error("Automatic combining of distributed output is not yet supported for $grid_type. " *
          "Consider using OceanEnsembles.jl or pass a pre-constructed global `grid` to FieldTimeSeries.")
end

#####
##### Loading and combining field data
#####

"""Load and combine field data from rank files into a field."""
function load_combined_field_data!(field, all_ranks, name, iter; reader_kw=NamedTuple())
    x_offsets, y_offsets = compute_partition_offsets(all_ranks)
    field_data = interior(field)

    for rank in all_ranks
        ri, rj, _ = rank.local_index
        nx, ny, nz = size(rank.local_grid)
        Hx, Hy, Hz = halo_size(rank.local_grid)

        raw_data = jldopen(rank.path; reader_kw...) do file
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

    all_ranks = filter(!isnothing, [load_rank_data(p; reader_kw) for p in rank_paths])
    isempty(all_ranks) && error("Could not load distributed grid info from rank files.")

    sort!(all_ranks, by = rd -> (rd.local_index[3], rd.local_index[2], rd.local_index[1]))

    architecture = something(architecture, isnothing(grid) ? CPU() : Architectures.architecture(grid))
    isnothing(grid) && (grid = reconstruct_global_grid_from_files(all_ranks, architecture))

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

    # Use DistributedPaths to store rank data - enables dispatch for both backends
    distributed_path = DistributedPaths(all_ranks)

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
##### InMemory support - set! dispatches on DistributedPaths
#####

"""Set FieldTimeSeries data by loading and combining from distributed rank files."""
function set!(fts::FieldTimeSeries{LX, LY, LZ, TI, K, I, D, G, ET, B, χ, <:DistributedPaths}
              ) where {LX, LY, LZ, TI, K <: AbstractInMemoryBackend, I, D, G, ET, B, χ}

    all_ranks = fts.path.ranks
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
            load_combined_field_data!(fts[n], all_ranks, fts.name, file_iterations[file_index];
                                      reader_kw=fts.reader_kw)
        end
    end

    return nothing
end

#####
##### OnDisk support - getindex dispatches on DistributedPaths
#####

"""
    getindex(fts, n::Int)

Load and combine field data from distributed rank files at time index `n`.
This method dispatches when `fts.path isa DistributedPaths` and `fts.backend isa OnDisk`.
"""
function Base.getindex(fts::FieldTimeSeries{LX, LY, LZ, TI, <:OnDisk, I, D, G, ET, B, χ, <:DistributedPaths},
                       n::Int) where {LX, LY, LZ, TI, I, D, G, ET, B, χ}
    all_ranks = fts.path.ranks
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
    load_combined_field_data!(field, all_ranks, fts.name, iter; reader_kw=fts.reader_kw)

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
