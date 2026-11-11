#####
##### Partitions that subdivide every region of a parent partition into px × py subregions, each of which may live on
##### its own device (see `MultiRegionPartition`). Subregion r = (q - 1) px py + (j - 1) px + i is the subregion (i, j)
##### of region q. The halo nodes of a subregion that lie inside the interior of its region are copies of the interior
##### of a sibling; the others are filled by the connectivity of the parent partition, split by donor subregion.
#####

"""
$(TYPEDFIELDS)

Subdivide each region of `partition`, whose regions have logically rectangular interiors of equal size,
into `subregions = (px, py)` subregions.

```jldoctest
julia> using Oceananigans.MultiRegion: SubregionPartition, YinYangPartition

julia> summary(SubregionPartition(YinYangPartition(), (2, 2)))
"YinYangPartition with (1: Yin, 2: Yang) with 2×2 subregions per region"
```
"""
struct SubregionPartition{P} <: AbstractPartition
    partition :: P
    subregions :: NTuple{2, Int}
end

Base.length(p::SubregionPartition) = length(p.partition) * prod(p.subregions)

Base.summary(p::SubregionPartition) = p.subregions == (1, 1) ? summary(p.partition) :
    string(summary(p.partition), " with ", p.subregions[1], "×", p.subregions[2], " subregions per region")

# The first index and size of each of `n` nearly equal parts of 1:N
function split_range(N, n)
    sizes = [N ÷ n + (m ≤ N % n) for m in 1:n]
    offsets = cumsum([0; sizes[1:end-1]])
    return offsets, sizes
end

"""
    subregion_layout(partition::SubregionPartition, (Nx, Ny))

The parent `region`, the index offsets `(ox, oy)` (region index = subregion index + offset) and the interior
`size` `(nx, ny)` of every subregion of `partition`, whose regions have `Nx × Ny` interior cells.
"""
function subregion_layout(p::SubregionPartition, (Nx, Ny))
    px, py = p.subregions
    x_offsets, x_sizes = split_range(Nx, px)
    y_offsets, y_sizes = split_range(Ny, py)
    return [(; region, offsets = (x_offsets[i], y_offsets[j]), size = (x_sizes[i], y_sizes[j]))
            for region in 1:length(p.partition) for j in 1:py for i in 1:px]
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

# Copy the halos of `data` on subregion `r` from its siblings, each on the device of its donor
function copy_subregion_halos!(field, r, copies)
    data = getregion(field.data, r)
    ks = axes(data, 3)
    for (donor, copy) in copies
        activate_region_device!(field.grid, donor)
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

#####
##### The device of each region
#####

"""
$(TYPEDFIELDS)

The partition of a `MultiRegionGrid` into the regions of `layout` (e.g. `XPartition(2)` or
`SubregionPartition(YinYangPartition(), (2, 2))`), with region `r` on device `devices[r]` (0-based),
or every region on the current device if `devices = nothing`.

```jldoctest
julia> using Oceananigans.MultiRegion: MultiRegionPartition, XPartition

julia> summary(MultiRegionPartition(XPartition(2); devices = (0, 1)))
"Equal partitioning in X with (2 regions) on devices (0, 1)"
```
"""
struct MultiRegionPartition{L, D}
    layout :: L
    devices :: D

    function MultiRegionPartition(layout; devices = nothing)
        isnothing(devices) || length(devices) == length(layout) ||
            throw(ArgumentError("$(summary(layout)) needs $(length(layout)) devices; got $devices."))
        return new{typeof(layout), typeof(devices)}(layout, devices)
    end
end

MultiRegionPartition(p::MultiRegionPartition) = p

Base.summary(p::MultiRegionPartition) = isnothing(p.devices) ? summary(p.layout) :
    string(summary(p.layout), " on devices ", p.devices)

# The architecture of every region of `partition`, a `MultiRegionObject`
function region_architectures(arch, p::MultiRegionPartition)
    isnothing(p.devices) && return MultiRegionObject(Tuple(arch for _ in 1:length(p.layout)))
    enable_region_peer_access!(arch, unique(p.devices))
    return MultiRegionObject(Tuple(on_device(arch, d) for d in p.devices))
end

# The devices of the regions of `grid`, or `nothing` if they are all on the current device
function region_devices(grid)
    archs = [regional_architecture(getregion(grid, r)) for r in regions(grid)]
    return any(isnothing, archs) ? nothing : Tuple(arch.ordinal for arch in archs)
end

Utils.regional_architecture(grid::AbstractGrid) = regional_architecture(architecture(grid))
Utils.regional_architecture(field::Field) = regional_architecture(field.grid)
