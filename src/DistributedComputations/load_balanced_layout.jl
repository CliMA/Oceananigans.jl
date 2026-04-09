using Oceananigans.Grids: AbstractGrid
using Oceananigans.ImmersedBoundaries: AbstractImmersedBoundary, ImmersedBoundaryGrid, immersed_cell

#####
##### Core 1D balanced partition
#####

"""
    balanced_1d_partition(c, R)

Partition the index range `1:N` into `R` contiguous bins minimizing the maximum
bin cost. `c` can be:

- `AbstractVector{<:Integer}` of length `N`: a single cost profile. The cost of
  a bin is `Σ c[i]` over the bin.
- `AbstractMatrix{<:Integer}` of size `N × K`: `K` simultaneous cost profiles.
  The cost of a bin is `max over k of (Σ c[i, k])`. This "multi-profile" variant
  is used by the alternating-minimization 2D optimizer, where each column is the
  load profile of a fixed cross-axis bin.

Returns a `Vector{Int}` of bin sizes summing to `N`.

Algorithm: binary search on the maximum bin cost with a greedy feasibility check.
"""
function balanced_1d_partition(c::AbstractVector{<:Integer}, R::Integer)
    N = length(c)
    R >= 1 || throw(ArgumentError("R must be ≥ 1, got R=$R"))
    R <= N || throw(ArgumentError("R=$R exceeds number of cells N=$N"))

    R == 1 && return [N]
    R == N && return fill(1, N)

    lo = max(maximum(c), 1)
    hi = max(sum(c), 1)

    function feasible(cap)
        bins  = 1
        accum = zero(eltype(c))
        for ci in c
            if accum + ci > cap
                bins += 1
                accum = ci
                bins > R && return false
            else
                accum += ci
            end
        end
        return true
    end

    while lo < hi
        mid = (lo + hi) ÷ 2
        feasible(mid) ? (hi = mid) : (lo = mid + 1)
    end

    return _reconstruct_partition(c, R, lo)
end

function balanced_1d_partition(c::AbstractMatrix{<:Integer}, R::Integer)
    N, K = size(c)
    R >= 1 || throw(ArgumentError("R must be ≥ 1, got R=$R"))
    R <= N || throw(ArgumentError("R=$R exceeds number of cells N=$N"))

    R == 1 && return [N]
    R == N && return fill(1, N)

    lo = max(maximum(c), 1)
    hi = max(maximum(sum(c, dims=1)), 1)

    function feasible(cap)
        bins  = 1
        accum = zeros(eltype(c), K)
        for i in 1:N
            overflow = false
            @inbounds for k in 1:K
                if accum[k] + c[i, k] > cap
                    overflow = true
                    break
                end
            end
            if overflow
                bins += 1
                bins > R && return false
                @inbounds for k in 1:K; accum[k] = c[i, k]; end
            else
                @inbounds for k in 1:K; accum[k] += c[i, k]; end
            end
        end
        return true
    end

    while lo < hi
        mid = (lo + hi) ÷ 2
        feasible(mid) ? (hi = mid) : (lo = mid + 1)
    end

    return _reconstruct_partition(c, R, lo)
end

# Shared reconstruction: greedily assign cells to bins at the optimal cap `L`.
function _reconstruct_partition(c::AbstractVecOrMat{<:Integer}, R::Integer, L::Integer)
    N = size(c, 1)
    K = ndims(c) == 1 ? 1 : size(c, 2)
    sizes   = Int[]
    accum   = zeros(eltype(c), K)
    count   = 0

    for idx in 1:N
        remaining = N - idx + 1
        bins_left = R - length(sizes)
        must_split = bins_left > remaining

        overflow = false
        if K == 1
            cv = ndims(c) == 1 ? c[idx] : c[idx, 1]
            overflow = accum[1] + cv > L
        else
            @inbounds for k in 1:K
                if accum[k] + c[idx, k] > L
                    overflow = true
                    break
                end
            end
        end

        can_split = length(sizes) < R - 1 && count > 0
        if (overflow || must_split) && can_split
            push!(sizes, count)
            if K == 1
                accum[1] = ndims(c) == 1 ? c[idx] : c[idx, 1]
            else
                @inbounds for k in 1:K; accum[k] = c[idx, k]; end
            end
            count = 1
        else
            if K == 1
                accum[1] += ndims(c) == 1 ? c[idx] : c[idx, 1]
            else
                @inbounds for k in 1:K; accum[k] += c[idx, k]; end
            end
            count += 1
        end
    end
    push!(sizes, count)

    @assert length(sizes) == R
    @assert sum(sizes) == N
    return sizes
end

#####
##### Column loads: the fundamental 2D data from one 3D-mask walk
#####

"""
    compute_column_loads(grid, immersed_boundary)

Walk the 3D immersed mask once and return a `Matrix{Int}` of size `(Nx, Ny)` where
entry `(i, j)` is the number of active cells in column `(i, j)` summed over `k`.
This is the single 3D-mask pass from which all partitioning and tile-load
computations are derived.
"""
function compute_column_loads(grid::AbstractGrid, immersed_boundary::AbstractImmersedBoundary)
    Nx, Ny, Nz = size(grid)
    col_load = zeros(Int, Nx, Ny)
    ibg = ImmersedBoundaryGrid(grid, immersed_boundary)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if !immersed_cell(i, j, k, ibg.underlying_grid, ibg.immersed_boundary)
            col_load[i, j] += 1
        end
    end

    return col_load
end

#####
##### Tile loads + partition optimization
#####

function _cell_to_tile(sizes::AbstractVector{<:Integer})
    N = sum(sizes)
    out = Vector{Int}(undef, N)
    cell = 1
    for (tile, n) in enumerate(sizes), _ in 1:n
        out[cell] = tile
        cell += 1
    end
    return out
end

function _tile_loads(col_load::Matrix{Int}, sizes_x, sizes_y)
    Rx, Ry = length(sizes_x), length(sizes_y)
    tile_x = _cell_to_tile(sizes_x)
    tile_y = _cell_to_tile(sizes_y)
    loads = zeros(Int, Rx, Ry)
    Nx, Ny = size(col_load)
    @inbounds for j in 1:Ny, i in 1:Nx
        loads[tile_x[i], tile_y[j]] += col_load[i, j]
    end
    return loads
end

# Aggregate col_load along one axis for fixed bins on the other axis.
# Returns an N × R_other matrix of load profiles.
function _aggregate(col_load::Matrix{Int}, sizes_other, dim::Int)
    Nx, Ny = size(col_load)
    R = length(sizes_other)
    tile_map = _cell_to_tile(sizes_other)
    if dim == 2  # aggregate y → profiles are Nx × Ry
        result = zeros(Int, Nx, R)
        @inbounds for j in 1:Ny, i in 1:Nx
            result[i, tile_map[j]] += col_load[i, j]
        end
    else          # aggregate x → profiles are Ny × Rx
        result = zeros(Int, Ny, R)
        @inbounds for j in 1:Ny, i in 1:Nx
            result[j, tile_map[i]] += col_load[i, j]
        end
    end
    return result
end

_active_imbalance(tl) = (a = filter(>(0), vec(tl)); isempty(a) ? 0.0 : maximum(a) / (sum(a) / length(a)))

"""
    _optimize_partition(col_load, Rx, Ry; max_iterations=20)

Find `(sizes_x, sizes_y)` that minimize the maximum tile load over the `Rx × Ry`
Cartesian partition, using alternating minimization:

1. Initialize from independent 1D projections.
2. Fix y-cuts → optimize x-cuts against the Ry simultaneous load profiles.
3. Fix x-cuts → optimize y-cuts against the Rx simultaneous load profiles.
4. Repeat until convergence.

Also tries the uniform initialization and keeps whichever is better.
"""
function _optimize_partition(col_load::Matrix{Int}, Rx::Int, Ry::Int;
                             max_iterations::Int = 20)
    Nx, Ny = size(col_load)

    # --- Run 1: initialize from 1D projections ---
    cx = vec(sum(col_load, dims=2))
    cy = vec(sum(col_load, dims=1))
    sx1 = balanced_1d_partition(cx, Rx)
    sy1 = balanced_1d_partition(cy, Ry)
    sx1, sy1 = _alternating_minimization(col_load, sx1, sy1, Rx, Ry, max_iterations)

    # --- Run 2: initialize from uniform partition ---
    sx2 = _split_evenly(Nx, Rx)
    sy2 = _split_evenly(Ny, Ry)
    sx2, sy2 = _alternating_minimization(col_load, sx2, sy2, Rx, Ry, max_iterations)

    # Pick the better result: lower max wins; ties broken by lower imbalance ratio
    tl1 = _tile_loads(col_load, sx1, sy1)
    tl2 = _tile_loads(col_load, sx2, sy2)
    score1 = (maximum(tl1), _active_imbalance(tl1))
    score2 = (maximum(tl2), _active_imbalance(tl2))
    return score2 < score1 ? (sx2, sy2) : (sx1, sy1)
end

function _split_evenly(N::Int, R::Int)
    base = N ÷ R
    rem  = N - base * R
    return [i <= rem ? base + 1 : base for i in 1:R]
end

function _alternating_minimization(col_load, sizes_x, sizes_y, Rx, Ry, max_iterations)
    for _ in 1:max_iterations
        old_x, old_y = copy(sizes_x), copy(sizes_y)

        # Fix y → optimize x across Ry load profiles
        profiles_x = _aggregate(col_load, sizes_y, 2)   # Nx × Ry
        sizes_x = balanced_1d_partition(profiles_x, Rx)

        # Fix x → optimize y across Rx load profiles
        profiles_y = _aggregate(col_load, sizes_x, 1)   # Ny × Rx
        sizes_y = balanced_1d_partition(profiles_y, Ry)

        sizes_x == old_x && sizes_y == old_y && break
    end
    return sizes_x, sizes_y
end

#####
##### Layout assembly
#####

"""
    build_rank_layout(sizes_x, sizes_y, tile_loads)

Construct a `RankLayout` from partition sizes and per-tile active-cell counts.
Tiles with zero load get rank id `-1`. Active tiles are numbered `0, 1, 2, …`
walking `iy` fastest (matching `index2rank` from `distributed_architectures.jl`).
"""
function build_rank_layout(sizes_x::AbstractVector{<:Integer},
                           sizes_y::AbstractVector{<:Integer},
                           tile_loads::AbstractMatrix{<:Integer})
    Rx, Ry = length(sizes_x), length(sizes_y)
    size(tile_loads) == (Rx, Ry) ||
        throw(ArgumentError("tile_loads has size $(size(tile_loads)) but expected ($Rx, $Ry)"))

    tile_to_rank = fill(-1, Rx, Ry)
    rank_to_tile = Tuple{Int, Int}[]
    next_rank = 0
    for ix in 1:Rx, iy in 1:Ry
        if tile_loads[ix, iy] > 0
            tile_to_rank[ix, iy] = next_rank
            push!(rank_to_tile, (ix, iy))
            next_rank += 1
        end
    end

    partition = Partition(Sizes(sizes_x...), Sizes(sizes_y...), 1)
    return RankLayout(partition, tile_to_rank, rank_to_tile)
end

#####
##### Public API
#####

"""
    load_balanced_layout(grid, immersed_boundary, (Rx, Ry); weight=:active_cells, report=true)

Compute a load-balanced `RankLayout` for a distributed immersed grid.

The algorithm finds Cartesian `(sizes_x, sizes_y)` cuts that minimize the maximum
tile load via alternating minimization: fix y-cuts and optimize x-cuts against
all y-profiles simultaneously (multi-profile balanced partition), then vice-versa,
repeating until convergence. Two initializations are tried (1D projections and
uniform) and the better result is kept.

Tiles with zero active cells are excluded from the layout; the recommended MPI
rank count equals the number of non-empty tiles.

Tripolar grids are handled by a separate method in `OrthogonalSphericalShellGrids`.
"""
function load_balanced_layout(grid::AbstractGrid,
                              immersed_boundary::AbstractImmersedBoundary,
                              tile_shape_target::Tuple{Integer, Integer};
                              weight::Symbol = :active_cells,
                              report::Bool = true)

    weight === :active_cells || throw(ArgumentError("weight=$weight not supported (only :active_cells for now)"))

    grid isa DistributedGrid &&
        throw(ArgumentError("load_balanced_layout requires a serial grid; received a DistributedGrid."))

    Nx, Ny, _ = size(grid)
    Rx, Ry = tile_shape_target
    Rx <= Nx || throw(ArgumentError("Rx=$Rx exceeds Nx=$Nx; tile_shape_target[1] must be ≤ Nx"))
    Ry <= Ny || throw(ArgumentError("Ry=$Ry exceeds Ny=$Ny; tile_shape_target[2] must be ≤ Ny"))

    col_load = compute_column_loads(grid, immersed_boundary)
    total = sum(col_load)
    total > 0 || throw(ArgumentError(
        "all cells are immersed — nothing to load-balance. " *
        "Check that your immersed_boundary does not mask the entire domain " *
        "(e.g. sign error in GridFittedBottom height)."))

    sizes_x, sizes_y = _optimize_partition(col_load, Rx, Ry)
    tile_loads = _tile_loads(col_load, sizes_x, sizes_y)
    layout = build_rank_layout(sizes_x, sizes_y, tile_loads)

    report && _print_layout_banner(layout, tile_loads, total)
    return layout
end

function _print_layout_banner(layout::RankLayout, tile_loads, total)
    Rx, Ry  = tile_shape(layout)
    n_total = Rx * Ry
    n_active = n_active_tiles(layout)
    n_empty  = n_total - n_active
    active_loads = filter(>(0), vec(tile_loads))
    max_load = isempty(active_loads) ? 0 : maximum(active_loads)
    mean_load = isempty(active_loads) ? 0.0 : sum(active_loads) / length(active_loads)
    ratio = mean_load == 0 ? 0.0 : max_load / mean_load

    println("┌─ load_balanced_layout ─────────────────────────────────────")
    println("│  tile shape           : $(Rx) × $(Ry)  ($(n_total) total)")
    println("│  active tiles         : $(n_active)")
    println("│  empty tiles          : $(n_empty)")
    println("│  total active cells   : $(total)")
    println("│  max / mean load      : $(round(ratio, digits=3))")
    println("│  recommended launch   : mpiexec -n $(n_active) julia ...")
    println("└────────────────────────────────────────────────────────────")
end

"""
    inspect_tile_occupancy(grid, immersed_boundary, (Rx, Ry); weight=:active_cells)

Return a diagnostic `String` showing how a load-balanced partition would populate
`(Rx, Ry)` tiles, without allocating a `RankLayout`. Useful for quickly comparing
factorizations.
"""
function inspect_tile_occupancy(grid::AbstractGrid,
                                immersed_boundary::AbstractImmersedBoundary,
                                tile_shape_target::Tuple{Integer, Integer};
                                weight::Symbol = :active_cells)

    weight === :active_cells || throw(ArgumentError("weight=$weight not supported (only :active_cells for now)"))

    Rx, Ry = tile_shape_target
    col_load = compute_column_loads(grid, immersed_boundary)
    sizes_x, sizes_y = _optimize_partition(col_load, Rx, Ry)
    tile_loads = _tile_loads(col_load, sizes_x, sizes_y)

    n_total  = Rx * Ry
    n_empty  = count(==(0), tile_loads)
    n_active = n_total - n_empty

    io = IOBuffer()
    println(io, "tile shape    : $(Rx) × $(Ry)  ($(n_total) total)")
    println(io, "active tiles  : $(n_active)")
    println(io, "empty tiles   : $(n_empty)")
    println(io, "active cells  : $(sum(col_load))")
    println(io, "tile loads:")
    for iy in Ry:-1:1   # print north on top so the matrix reads as a map view
        print(io, "  ")
        for ix in 1:Rx
            print(io, lpad(tile_loads[ix, iy], 8))
        end
        println(io)
    end
    return String(take!(io))
end

#####
##### Backward-compatible convenience wrappers used by tests
#####

"""Convenience: derive 1D projections from column loads."""
function compute_active_cell_weights(grid::AbstractGrid, immersed_boundary::AbstractImmersedBoundary)
    col_load = compute_column_loads(grid, immersed_boundary)
    cx = vec(sum(col_load, dims=2))
    cy = vec(sum(col_load, dims=1))
    return cx, cy, sum(col_load)
end

"""Convenience: compute tile loads from grid + immersed boundary."""
function compute_tile_loads(grid::AbstractGrid, immersed_boundary::AbstractImmersedBoundary,
                            sizes_x::AbstractVector{<:Integer},
                            sizes_y::AbstractVector{<:Integer})
    Nx, Ny, _ = size(grid)
    sum(sizes_x) == Nx || throw(ArgumentError("sum(sizes_x)=$(sum(sizes_x)) must equal Nx=$Nx"))
    sum(sizes_y) == Ny || throw(ArgumentError("sum(sizes_y)=$(sum(sizes_y)) must equal Ny=$Ny"))
    return _tile_loads(compute_column_loads(grid, immersed_boundary), sizes_x, sizes_y)
end
