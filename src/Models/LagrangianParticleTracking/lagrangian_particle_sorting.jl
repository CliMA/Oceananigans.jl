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

After [`sort_lagrangian_particles!`](@ref), the particles of each species are stored in order of
increasing linear cell index `c = i + Nx * (j - 1 + Ny * (k - 1))`. For a single species,

- `cell_indices[p]` is the linear cell index of particle `p`;
- the particles in cell `c` are `cell_offsets[c] + 1 : cell_offsets[c + 1]`;
- `permutation[p]` is the index that particle `p` had before the most recent sort.

With multiple species, `cell_indices` and `permutation` are `NamedTuple`s with one vector per
species, and `cell_offsets` stores the offsets of all species in a vector of length `(Nc + 1) * S`
for `Nc` cells and `S` species, which is the column-major layout of an `(Nc + 1) × S` matrix.
The particles of species `s` in cell `c` are `cell_offsets[o + c] + 1 : cell_offsets[o + c + 1]`,
where `o = (s - 1) * (Nc + 1)`, so that threads that visit consecutive cells read consecutive
offsets for every species. Use [`cell_particle_indices`](@ref) to obtain these ranges.

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

function CellSorting(algorithm::SortingAlgorithm, properties,
                     buffers = map_species(species -> similar(species, 0), properties))
    x = first(particle_species(properties)).x
    cell_indices = map_species(species -> similar(species.x, Int, 0), properties)
    permutation  = map_species(species -> similar(species.x, Int, 0), properties)
    cell_offsets = similar(x, Int, 0)
    cell_counts  = similar(x, Int, 0)
    return CellSorting(algorithm, cell_indices, permutation, cell_offsets, cell_counts, buffers)
end

materialize_sorting(::Nothing, properties, buffers) = nothing
materialize_sorting(algorithm::SortingAlgorithm, properties, buffers) = CellSorting(algorithm, properties, buffers)
materialize_sorting(sorting::CellSorting, properties, buffers) = sorting

Base.summary(sorting::CellSorting) = string("CellSorting(", summary(sorting.algorithm), ")")

"""
$(TYPEDSIGNATURES)

Return the linear index `i + Nx * (j - 1 + Ny * (k - 1))` of cell `(i, j, k)` of `grid`.
"""
@inline function linear_cell_index(i, j, k, grid)
    Nx, Ny, _ = size(grid)
    return i + Nx * (j - 1 + Ny * (k - 1))
end

@inline number_of_cells(grid) = prod(size(grid))

# Index of the offset of cell `c` of species `s` in `cell_offsets`, for `Nc` cells
@inline species_cell_index(c, s, Nc) = c + (s - 1) * (Nc + 1)

@inline function cell_particle_range(c, cell_offsets)
    @inbounds return cell_offsets[c] + 1 : cell_offsets[c + 1]
end

"""
$(TYPEDSIGNATURES)

Return the range of indices of the particles of species `s` in cell `(i, j, k)` of `grid`,
valid after the most recent [`sort_lagrangian_particles!`](@ref). The indices refer to
the `StructArray` of species `s`, that is, `particle_species(particles)[s]`.
"""
@inline function cell_particle_indices(i, j, k, grid, sorting::CellSorting, s)
    c = linear_cell_index(i, j, k, grid)
    return cell_particle_range(species_cell_index(c, s, number_of_cells(grid)), sorting.cell_offsets)
end

"""
$(TYPEDSIGNATURES)

Return the range of indices of the particles in cell `(i, j, k)` of `grid`,
valid after the most recent [`sort_lagrangian_particles!`](@ref).
For multiple species, this is the range of the first species.
"""
@inline cell_particle_indices(i, j, k, grid, sorting::CellSorting) = cell_particle_indices(i, j, k, grid, sorting, 1)

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

# Compute the cell index of particle `p` of species `s`, and count the particle in its cell.
# `cell_counts[o + c + 1]` is incremented, where `o = (s - 1) * (Nc + 1)`, so that
# a cumulative sum over the cells of each species yields the offsets of each cell.
@inline function index_and_count_particle!((properties, cell_indices, s), p, (grid, cell_counts))
    @inbounds begin
        x = properties.x[p]
        y = properties.y[p]
        z = properties.z[p]
    end

    c = particle_cell_index((x, y, z), grid)
    @inbounds cell_indices[p] = c

    n = species_cell_index(c, s, number_of_cells(grid)) + 1
    @atomic cell_counts[n] += 1

    return nothing
end

#####
##### Algorithm-specific permutations
#####

# Scatter particles into their cells. `cursors[n]` starts at `cell_offsets[n]`,
# so the incremented value is the sorted position of the particle.
@inline function scatter_particle!((cell_indices, permutation, s), p, (cursors, Nc))
    @inbounds c = cell_indices[p]
    n = species_cell_index(c, s, Nc)
    t = @atomic cursors[n] += 1
    @inbounds permutation[t] = p
    return nothing
end

function compute_permutation!(sorting::CellSorting{<:CountingSort}, arch, grid, counts, cell_indices, permutations)
    cursors = sorting.cell_counts
    copyto!(cursors, sorting.cell_offsets)
    species_args = map(tuple, cell_indices, permutations, species_numbers(counts))
    launch_species!(arch, scatter_particle!, counts, species_args, (cursors, number_of_cells(grid)))
    return nothing
end

species_numbers(counts::Tuple) = ntuple(identity, length(counts))

# Sort keys (c - 1) * Np + (p - 1) are unique, so any sort algorithm
# yields the same permutation, ordered by cell and then by pre-sort position.
@inline function compute_sort_key!((keys, cell_indices, Np), p, shared_args)
    @inbounds keys[p] = (cell_indices[p] - 1) * Np + (p - 1)
    return nothing
end

@inline function decode_sort_key!((keys, Np), q, shared_args)
    @inbounds keys[q] = mod(keys[q], Np) + 1
    return nothing
end

function compute_permutation!(sorting::CellSorting{<:ComparisonSort}, arch, grid, counts, cell_indices, permutations)
    launch_species!(arch, compute_sort_key!, counts, map(tuple, permutations, cell_indices, counts), nothing)
    foreach(keys -> isempty(keys) || sort!(keys), permutations) # CUDA's sort! fails for empty arrays
    launch_species!(arch, decode_sort_key!, counts, map(tuple, permutations, counts), nothing)
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

@inline function finalize_cell!(c, n, permutation, cell_indices, cell_offsets, algorithm)
    particles_in_cell = cell_particle_range(n, cell_offsets)

    order_within_cell!(algorithm, permutation, first(particles_in_cell), last(particles_in_cell))

    for q in particles_in_cell
        @inbounds cell_indices[q] = c
    end

    return nothing
end

# Finalize cell `c` for every species
@inline function finalize_species_cells!(c, Nc, s, species_args::Tuple, cell_offsets, algorithm)
    permutation, cell_indices = first(species_args)
    finalize_cell!(c, species_cell_index(c, s, Nc), permutation, cell_indices, cell_offsets, algorithm)
    finalize_species_cells!(c, Nc, s + 1, Base.tail(species_args), cell_offsets, algorithm)
    return nothing
end

@inline finalize_species_cells!(c, Nc, s, ::Tuple{}, cell_offsets, algorithm) = nothing

@kernel function _finalize_cells!(grid, species_args, cell_offsets, algorithm)
    i, j, k = @index(Global, NTuple)
    c = linear_cell_index(i, j, k, grid)
    finalize_species_cells!(c, number_of_cells(grid), 1, species_args, cell_offsets, algorithm)
end

#####
##### Reordering particle properties
#####

"""
$(TYPEDSIGNATURES)

Reorder the properties and transfer flags of each species of `particles`
according to `permutations`, so that particle `p` of species `s` becomes particle `q`, where
`permutations[s][q] = p`. All attributes of all species are gathered in a single kernel, and
copied back in a second kernel. For permutations that move a small fraction of particles,
as when sorted particles are sorted again, this is faster than gathering each attribute separately.
"""
function permute_particles!(particles, arch, permutations)
    properties = particles.properties
    species = particle_species(properties)
    counts = map(length, species)

    attributes = species_attributes(particles)
    buffers = resize_species_buffers!(particles, particles.sorting.buffers, counts)

    launch_species!(arch, gather_particle!, counts, map(tuple, permutations, buffers, attributes), nothing)
    launch_species!(arch, copy_particle!, counts, map(tuple, attributes, buffers), nothing)

    return nothing
end

# The attributes of each species; see `particle_attributes`
function species_attributes(particles)
    properties = particles.properties
    flags = species_tuple(properties, transfer_flags(particles.transfers))
    return map(particle_attributes, particle_species(properties), flags)
end

# Resize the buffers of each species to `counts` and return their attributes
function resize_species_buffers!(particles, buffers, counts)
    properties = particles.properties
    buffers = species_tuple(properties, buffers)
    flag_buffers = species_tuple(properties, transfer_flag_buffers(particles.transfers))
    attributes = map(particle_attributes, buffers, flag_buffers)
    foreach(attributes, counts) do arrays, N
        foreach(array -> resize!(array, N), arrays)
    end
    return attributes
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
The transfer flags in `particles.transfers` are permuted together with the properties.
Each species is sorted separately, but the cell indices of all species are computed and counted
in a single kernel, and the particles of all species are scattered into their cells in a single kernel.

After sorting, the particles of species `s` in cell `(i, j, k)` are indexed by
`cell_particle_indices(i, j, k, grid, particles.sorting, s)`. Note that sorting changes the
index of each particle, so particle identity should be tracked with a custom property.
"""
function sort_lagrangian_particles!(particles::LagrangianParticles, grid)
    sorting = particles.sorting
    properties = particles.properties
    species = particle_species(properties)
    counts = map(length, species)
    S = length(species)
    Nc = number_of_cells(grid)
    arch = architecture(grid)

    cell_indices = species_tuple(properties, sorting.cell_indices)
    permutations = species_tuple(properties, sorting.permutation)

    resize!(sorting.cell_offsets, S * (Nc + 1))
    resize!(sorting.cell_counts, S * (Nc + 1))
    foreach(resize!, cell_indices, counts)
    foreach(resize!, permutations, counts)

    fill!(sorting.cell_offsets, 0)
    sum(counts) == 0 && return nothing

    # Joint histogram over (cell, species)
    fill!(sorting.cell_counts, 0)
    species_args = map(tuple, species, cell_indices, species_numbers(counts))
    launch_species!(arch, index_and_count_particle!, counts, species_args, (grid, sorting.cell_counts))

    # Offsets of each cell within each species
    for s in 1:S
        n = (s - 1) * (Nc + 1) + 1 : s * (Nc + 1)
        cumsum!(view(sorting.cell_offsets, n), view(sorting.cell_counts, n))
    end

    compute_permutation!(sorting, arch, grid, counts, cell_indices, permutations)

    launch!(arch, grid, :xyz, _finalize_cells!,
            grid, map(tuple, permutations, cell_indices), sorting.cell_offsets, sorting.algorithm)

    permute_particles!(particles, arch, permutations)

    return nothing
end
