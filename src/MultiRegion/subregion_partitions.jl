#####
##### Partitions of the regions of a parent partition into pieces: logically rectangular ranges of the interior of a
##### region, each the interior of one region of the grid (in time, of one rank). Every region may be split differently,
##### into pieces of different sizes. The halo nodes of a piece that lie inside the interior of its region are copies
##### of the interior of a sibling; the others are filled by the connectivity of the parent partition, split by donor
##### piece. Everything below works on the list of pieces, whatever its order or shape.
#####

# The parent `region`, the index `offsets` `(ox, oy)` (region index = piece index + offset) and the interior `size`
# `(nx, ny)` of every piece; pieces of the same region tile its interior
struct SubregionPartition{P, L} <: AbstractPartition
    partition :: P
    layout :: L # Vector of (; region, offsets, size)
end

Base.length(p::SubregionPartition) = length(p.layout)

Base.summary(p::SubregionPartition) = length(p) == length(p.partition) ? summary(p.partition) :
    string(summary(p.partition), " in ", length(p), " pieces")

# The first index and size of each of `n` nearly equal parts of 1:N
function split_range(N, n)
    sizes = [N ÷ n + (m ≤ N % n) for m in 1:n]
    offsets = cumsum([0; sizes[1:end-1]])
    return offsets, sizes
end

# Split region q of `partition`, whose regions have `Nx × Ny` interior cells, into `splits[q] = (px, py)` nearly equal
# pieces, numbered region by region and then as (i, j) ↦ (j - 1) px + i
function SubregionPartition(partition, splits, (Nx, Ny))
    layout = NamedTuple{(:region, :offsets, :size), Tuple{Int, NTuple{2, Int}, NTuple{2, Int}}}[]
    for (region, (px, py)) in enumerate(splits)
        x_offsets, x_sizes = split_range(Nx, px)
        y_offsets, y_sizes = split_range(Ny, py)
        for j in 1:py, i in 1:px
            push!(layout, (; region, offsets = (x_offsets[i], y_offsets[j]), size = (x_sizes[i], y_sizes[j])))
        end
    end
    return SubregionPartition(partition, layout)
end

"""
$(TYPEDSIGNATURES)

The partition of each region of a multi-region grid by its own `Partition`, e.g.
`MultiRegionPartition(Partition(1), Partition(2))` puts the first region in one piece and splits the second in
two along `x`. A grid takes a single `Partition` as the same partition of every region.

```jldoctest
julia> using Oceananigans

julia> using Oceananigans.MultiRegion: MultiRegionPartition

julia> length(MultiRegionPartition(Partition(1), Partition(2, 2)))
5
```
"""
struct MultiRegionPartition{P}
    partitions :: P
    MultiRegionPartition(partitions::Partition...) = new{typeof(partitions)}(partitions)
end

Base.length(p::MultiRegionPartition) = sum(prod, map(ranks, p.partitions))

function region_splits(p::MultiRegionPartition)
    for partition in p.partitions
        ranks(partition)[3] == 1 || throw(ArgumentError("Regions cannot be partitioned in z; got $(summary(partition))."))
    end
    return map(partition -> ranks(partition)[1:2], p.partitions)
end

# The subregion of `region` whose interior holds the region index (I, J)
function owning_subregion(layout, region, (I, J))
    for (r, subregion) in enumerate(layout)
        (ox, oy), (nx, ny) = subregion.offsets, subregion.size
        subregion.region == region && ox < I ≤ ox + nx && oy < J ≤ oy + ny && return r
    end
    return nothing
end

# A stencil of width s starting at the subregion index i₀ is held, halos included, by a subregion of interior size n
stencil_held(i₀, s, n, H) = 1 - H ≤ i₀ && i₀ + s - 1 ≤ n + H

#####
##### Copies between sibling subregions
#####

# The receiver and donor indices of the halo nodes of one subregion copied from one sibling
struct SubregionCopy{I}
    receivers :: I # (; i, j)
    donors :: I    # (; i, j)
end

Base.length(copy::SubregionCopy) = length(copy.receivers.i)

Adapt.adapt_structure(to, copy::SubregionCopy) = SubregionCopy(adapt(to, copy.receivers), adapt(to, copy.donors))
Grids.on_architecture(arch, copy::SubregionCopy) =
    SubregionCopy(on_architecture(arch, copy.receivers), on_architecture(arch, copy.donors))

function SubregionCopy(pairs)
    indices(nodes) = (; i = first.(nodes), j = last.(nodes))
    return SubregionCopy(indices(first.(pairs)), indices(last.(pairs)))
end

"""
    subregion_halos(layout, r, (Nx, Ny), (Hx, Hy))

The copies that fill the halo nodes of subregion `r` inside the `Nx × Ny` interior of its region, as a `Vector` of
`(donor, copy::SubregionCopy)`, and the region indices `(I, J)` of its other halo nodes, in the order of
`halo_nodes`.
"""
function subregion_halos(layout, r, (Nx, Ny), (Hx, Hy))
    (ox, oy), (nx, ny) = layout[r].offsets, layout[r].size

    copies = Dict{Int, Vector{Tuple{NTuple{2, Int}, NTuple{2, Int}}}}()
    seam_nodes = NTuple{2, Int}[]

    for (i, j) in halo_nodes((nx, ny), (Hx, Hy))
        I, J = i + ox, j + oy
        if 1 ≤ I ≤ Nx && 1 ≤ J ≤ Ny
            d = owning_subregion(layout, layout[r].region, (I, J))
            dox, doy = layout[d].offsets
            push!(get!(copies, d, []), ((i, j), (I - dox, J - doy)))
        else
            push!(seam_nodes, (I, J))
        end
    end

    # A Vector, not a Tuple: the number of donors varies by subregion and must not enter the type of the grid
    return [(d, SubregionCopy(pairs)) for (d, pairs) in sort(collect(copies))], seam_nodes
end

# Copy the halos of `data` on subregion `r` from its siblings
function copy_subregion_halos!(field, r, copies)
    data = getregion(field.data, r)
    ks = axes(data, 3)
    for (donor, copy) in copies
        kernel_parameters = KernelParameters((length(copy), length(ks)), (0, first(ks) - 1))
        launch!(architecture(field.grid), getregion(field.grid, donor), kernel_parameters,
                _copy_subregion_halo!, data, getregion(field.data, donor), copy)
    end
    return nothing
end

@kernel function _copy_subregion_halo!(c, source, copy)
    n, k = @index(Global, NTuple)
    @inbounds c[copy.receivers.i[n], copy.receivers.j[n], k] = source[copy.donors.i[n], copy.donors.j[n], k]
end

