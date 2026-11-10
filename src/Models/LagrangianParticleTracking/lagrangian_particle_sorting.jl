using Oceananigans.Utils: KernelParameters

#####
##### Sorting algorithms
#####

"""
    CountingSort()

Sort `LagrangianParticles` by the grid cell they occupy using a counting sort.

Particles are counted per cell with atomic increments, cell offsets are computed with a
cumulative sum, and particles are scattered into their cells with atomic increments.
A final pass orders the particles within each cell by their position before the sort,
so that the result is deterministic and identical to [`ComparisonSort`](@ref).
The cost is linear in the number of particles plus the number of cells.
"""
struct CountingSort end

"""
    ComparisonSort()

Sort `LagrangianParticles` by the grid cell they occupy using the general-purpose `sort!`
for the particle array type (for example, the CUDA sort for `CuArray`s).

Sort keys combine the cell index with the position of each particle before the sort,
so that the result is deterministic and identical to [`CountingSort`](@ref).
The cost is ``O(N \\log N)`` for ``N`` particles.
"""
struct ComparisonSort end

const SortingAlgorithm = Union{CountingSort, ComparisonSort}

Base.summary(::CountingSort) = "CountingSort()"
Base.summary(::ComparisonSort) = "ComparisonSort()"

#####
##### Sorting state
#####

"""
    CellSorting{A, I, P, O, C, B}

The `sorting` property of `LagrangianParticles` whose particles are sorted by grid cell.

After [`sort_lagrangian_particles!`](@ref), particle properties are stored in order of increasing
linear cell index `c = i + Nx * (j - 1 + Ny * (k - 1))`, and

- `cell_indices[p]` is the linear cell index of particle `p`;
- the particles in cell `c` are `cell_offsets[c] + 1 : cell_offsets[c + 1]`;
- `permutation[p]` is the index that particle `p` had before the most recent sort.

Within each cell, particles keep the order they had before the sort.
"""
struct CellSorting{A, I, P, O, C, B}
       algorithm :: A
    cell_indices :: I
     permutation :: P
    cell_offsets :: O
     cell_counts :: C
         buffers :: B
end

function CellSorting(algorithm::SortingAlgorithm, properties::StructArray)
    x = properties.x
    cell_indices = similar(x, Int, 0)
    permutation  = similar(x, Int, 0)
    cell_offsets = similar(x, Int, 0)
    cell_counts  = similar(x, Int, 0)
    buffers = map(property -> similar(property, 0), components(properties))
    return CellSorting(algorithm, cell_indices, permutation, cell_offsets, cell_counts, buffers)
end

materialize_sorting(::Nothing, properties) = nothing
materialize_sorting(algorithm::SortingAlgorithm, properties) = CellSorting(algorithm, properties)
materialize_sorting(sorting::CellSorting, properties) = sorting

Base.summary(sorting::CellSorting) = string("CellSorting(", summary(sorting.algorithm), ")")

"""
$(TYPEDSIGNATURES)

Return the linear index `i + Nx * (j - 1 + Ny * (k - 1))` of cell `(i, j, k)` of `grid`.
"""
@inline function linear_cell_index(i, j, k, grid)
    Nx, Ny, _ = size(grid)
    return i + Nx * (j - 1 + Ny * (k - 1))
end

@inline function cell_particle_range(c, cell_offsets)
    @inbounds return cell_offsets[c] + 1 : cell_offsets[c + 1]
end

"""
$(TYPEDSIGNATURES)

Return the range of indices of the particles in cell `(i, j, k)` of `grid`,
valid after the most recent [`sort_lagrangian_particles!`](@ref).
"""
@inline cell_particle_indices(i, j, k, grid, sorting::CellSorting) =
    cell_particle_range(linear_cell_index(i, j, k, grid), sorting.cell_offsets)

#####
##### Cell index of a particle
#####

"""
$(TYPEDSIGNATURES)

Return the linear index `i + Nx * (j - 1 + Ny * (k - 1))` of the cell of `grid`
that contains the position `(x, y, z)`. Positions outside the grid are assigned to the
nearest boundary cell.
"""
@inline function particle_cell_index((x, y, z), grid)
    X = flattened_node((x, y, z), grid)
    fi = FractionalIndices(X, grid, f, f, f)

    i, _, _ = interpolator(fi.i)
    j, _, _ = interpolator(fi.j)
    k, _, _ = interpolator(fi.k)

    Nx, Ny, Nz = size(grid)
    i = clamp(i, 1, Nx)
    j = clamp(j, 1, Ny)
    k = clamp(k, 1, Nz)

    return linear_cell_index(i, j, k, grid)
end

@kernel function _compute_cell_indices!(cell_indices, properties, grid)
    p = @index(Global)
    @inbounds begin
        x = properties.x[p]
        y = properties.y[p]
        z = properties.z[p]
        cell_indices[p] = particle_cell_index((x, y, z), grid)
    end
end

@kernel function _count_particles_per_cell!(cell_counts, cell_indices)
    p = @index(Global)
    @inbounds c = cell_indices[p]
    # cell_counts[c + 1] so that a cumulative sum yields the offsets of each cell
    @atomic cell_counts[c + 1] += 1
end

#####
##### Algorithm-specific permutations
#####

# Scatter particles into their cells. `cursors[c]` starts at `cell_offsets[c]`,
# so the incremented value is the sorted position of the particle.
@kernel function _scatter_particles!(permutation, cursors, cell_indices)
    p = @index(Global)
    @inbounds c = cell_indices[p]
    s = @atomic cursors[c] += 1
    @inbounds permutation[s] = p
end

function compute_permutation!(sorting::CellSorting{<:CountingSort}, grid, Np)
    arch = architecture(grid)
    cursors = sorting.cell_counts
    copyto!(cursors, sorting.cell_offsets)
    launch!(arch, grid, KernelParameters(1:Np), _scatter_particles!,
            sorting.permutation, cursors, sorting.cell_indices)
    return nothing
end

# Sort keys (c - 1) * Np + (p - 1) are unique, so any sort algorithm
# yields the same permutation, ordered by cell and then by pre-sort position.
@kernel function _compute_sort_keys!(keys, cell_indices, Np)
    p = @index(Global)
    @inbounds keys[p] = (cell_indices[p] - 1) * Np + (p - 1)
end

@kernel function _decode_sort_keys!(keys, Np)
    s = @index(Global)
    @inbounds keys[s] = mod(keys[s], Np) + 1
end

function compute_permutation!(sorting::CellSorting{<:ComparisonSort}, grid, Np)
    arch = architecture(grid)
    keys = sorting.permutation
    launch!(arch, grid, KernelParameters(1:Np), _compute_sort_keys!, keys, sorting.cell_indices, Np)
    sort!(keys)
    launch!(arch, grid, KernelParameters(1:Np), _decode_sort_keys!, keys, Np)
    return nothing
end

#####
##### Per-cell finalization: deterministic order within cells and sorted cell indices
#####

# Insertion sort of the pre-sort indices within a cell, which makes the counting sort stable.
@inline function order_within_cell!(::CountingSort, permutation, start, stop)
    for s in start+1:stop
        @inbounds q = permutation[s]
        t = s - 1
        while t ≥ start && @inbounds(permutation[t] > q)
            @inbounds permutation[t+1] = permutation[t]
            t -= 1
        end
        @inbounds permutation[t+1] = q
    end
    return nothing
end

# Unique sort keys already order particles within cells.
@inline order_within_cell!(::ComparisonSort, permutation, start, stop) = nothing

@inline function finalize_cell!(i, j, k, grid, permutation, cell_indices, cell_offsets, algorithm)
    c = linear_cell_index(i, j, k, grid)
    particles_in_cell = cell_particle_range(c, cell_offsets)

    order_within_cell!(algorithm, permutation, first(particles_in_cell), last(particles_in_cell))

    for s in particles_in_cell
        @inbounds cell_indices[s] = c
    end

    return nothing
end

@kernel function _finalize_cells!(grid, permutation, cell_indices, cell_offsets, algorithm)
    i, j, k = @index(Global, NTuple)
    finalize_cell!(i, j, k, grid, permutation, cell_indices, cell_offsets, algorithm)
end

#####
##### Reordering particle properties
#####

@kernel function _gather!(buffer, property, permutation)
    s = @index(Global)
    @inbounds buffer[s] = property[permutation[s]]
end

function permute_properties!(properties, sorting, grid, Np)
    arch = architecture(grid)
    for (property, buffer) in zip(components(properties), sorting.buffers)
        launch!(arch, grid, KernelParameters(1:Np), _gather!, buffer, property, sorting.permutation)
        copyto!(property, buffer)
    end
    return nothing
end

#####
##### Sorting Lagrangian particles
#####

sort_lagrangian_particles!(particles, grid) = nothing
sort_lagrangian_particles!(::LagrangianParticles{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing}, grid) = nothing

"""
$(TYPEDSIGNATURES)

Sort the properties of `particles` by the cell of `grid` that each particle occupies,
and update `particles.sorting`. Does nothing if `particles.sorting` is `nothing`.

After sorting, the particles in cell `(i, j, k)` are indexed by
`cell_particle_indices(i, j, k, grid, particles.sorting)`. Note that sorting changes the
index of each particle, so particle identity should be tracked with a custom property.
"""
function sort_lagrangian_particles!(particles::LagrangianParticles, grid)
    sorting = particles.sorting
    properties = particles.properties
    Np = length(properties)
    Nc = prod(size(grid))
    arch = architecture(grid)

    resize!(sorting.cell_offsets, Nc + 1)
    resize!(sorting.cell_counts, Nc + 1)
    resize!(sorting.cell_indices, Np)
    resize!(sorting.permutation, Np)
    foreach(buffer -> resize!(buffer, Np), sorting.buffers)

    fill!(sorting.cell_offsets, 0)
    Np == 0 && return nothing

    launch!(arch, grid, KernelParameters(1:Np), _compute_cell_indices!, sorting.cell_indices, properties, grid)

    fill!(sorting.cell_counts, 0)
    launch!(arch, grid, KernelParameters(1:Np), _count_particles_per_cell!, sorting.cell_counts, sorting.cell_indices)
    cumsum!(sorting.cell_offsets, sorting.cell_counts)

    compute_permutation!(sorting, grid, Np)

    launch!(arch, grid, :xyz, _finalize_cells!,
            grid, sorting.permutation, sorting.cell_indices, sorting.cell_offsets, sorting.algorithm)

    permute_properties!(properties, sorting, grid, Np)

    return nothing
end
