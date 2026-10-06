#####
##### Utilities for launching kernels
#####

# Δt for kernel arguments: Metal cannot load Float64 ones. Reactant needs it unconverted,
# and overrides this in OceananigansReactantExt.
@inline kernel_time_step(arch, grid, Δt) = convert(eltype(grid), Δt)

using Adapt: Adapt
using Base: @pure
using KernelAbstractions: Kernel,
                          KernelAbstractions as KA,
                          ndrange, workgroupsize,
                          CompilerMetadata
using KernelAbstractions.NDIteration: NDIteration, NDRange, blocks, workitems, _Size
using Oceananigans.Architectures: Architectures

import Oceananigans
import KernelAbstractions: get, expand, StaticSize

struct KernelParameters{S, O} end

"""
$(TYPEDSIGNATURES)

Return parameters for kernel launching and execution that define (i) a tuple that
defines the `size` of the kernel being launched and (ii) a tuple of `offsets` that
offset loop indices. For example, `offsets = (0, 0, 0)` with `size = (N, N, N)` means
all indices loop from `1:N`. If `offsets = (1, 1, 1)`, then all indices loop from
`2:N+1`. And so on.

Example
=======

```julia
size = (8, 6, 4)
offsets = (0, 1, 2)
kp = KernelParameters(size, offsets)

# Launch a kernel with indices that range from i=1:8, j=2:7, k=3:6,
# where i, j, k are the first, second, and third index, respectively:

launch!(arch, grid, kp, kernel!, kernel_args...)
```

See [`launch!`](@ref).
"""
@inline KernelParameters(size, offsets) = KernelParameters{size, offsets}()

# If `size` and `offsets` are numbers, we convert them to tuples
KernelParameters(s::Number, o::Number) = KernelParameters(tuple(s), tuple(o))

"""
$(TYPEDSIGNATURES)

Return parameters for launching a kernel of up to three dimensions, where the
indices spanned by the kernel in each dimension are given by (range1, range2, range3).

Example
=======

```julia
kp = KernelParameters(1:4, 0:10)

# Launch a kernel with indices that range from i=1:4, j=0:10,
# where i, j are the first and second index, respectively.
launch!(arch, grid, kp, kernel!, kernel_args...)
```

See the documentation for [`launch!`](@ref).
"""
function KernelParameters(r::AbstractUnitRange)
    size = length(r)
    offset = first(r) - 1
    return KernelParameters(tuple(size), tuple(offset))
end

function KernelParameters(r1::AbstractUnitRange, r2::AbstractUnitRange)
    size = (length(r1), length(r2))
    offsets = (first(r1) - 1, first(r2) - 1)
    return KernelParameters(size, offsets)
end

function KernelParameters(r1::AbstractUnitRange, r2::AbstractUnitRange, r3::AbstractUnitRange)
    size = (length(r1), length(r2), length(r3))
    offsets = (first(r1) - 1, first(r2) - 1, first(r3) - 1)
    return KernelParameters(size, offsets)
end

# Convenience `Tuple`d constructor
KernelParameters(args::Tuple) = KernelParameters(args...)

contiguousrange(range::StaticSize{S}, offset) where S = contiguousrange(S, offset)
contiguousrange(::KernelParameters{S, O}) where {S, O} = contiguousrange(S, O)
contiguousrange(range::NTuple{N, Int}, offset::NTuple{N, Int}) where N = Tuple(1+o:r+o for (r, o) in zip(range, offset))

# Heuristic for 1-tuple, 2-tuple and 3-tuple of integers
contiguousrange(range::NTuple{1, Int}, offset::NTuple{1, Int}) = @inbounds (1+offset[1]:range[1]+offset[1], )
contiguousrange(range::NTuple{2, Int}, offset::NTuple{2, Int}) = @inbounds (1+offset[1]:range[1]+offset[1], 1+offset[2]:range[2]+offset[2])
contiguousrange(range::NTuple{3, Int}, offset::NTuple{3, Int}) = @inbounds (1+offset[1]:range[1]+offset[1], 1+offset[2]:range[2]+offset[2], 1+offset[3]:range[3]+offset[3])

flatten_reduced_dimensions(worksize, dims) = Tuple(d ∈ dims ? 1 : worksize[d] for d = 1:3)

# Heuristic for a 3-tuple of integers (our main case)
flatten_reduced_dimensions(worksize::Tuple{Int, Int, Int}, dims) =
    (1 ∈ dims ? 1 : worksize[1],
     2 ∈ dims ? 1 : worksize[2],
     3 ∈ dims ? 1 : worksize[3])

# Support for 1D
heuristic_workgroup(Wx) = min(Wx, 256)

# This supports 2D, 3D and 4D work sizes (but the 3rd and 4th dimension are discarded)
function heuristic_workgroup(Wx::Int, Wy::Int, Wz=nothing, Wt=nothing)
    if Wx == 1 && Wy == 1            # One-dimensional column models
        return (1, 1)
    elseif Wx == 1                   # Two-dimensional y-z slice models
        return (1, min(256, Wy))
    elseif Wy == 1                   # Two-dimensional x-z slice models
        return (min(256, Wx), 1)
    else                             # Three-dimensional models
        return (16, 16)
    end
end

"""
$(TYPEDSIGNATURES)

Return the workgroup for a kernel launched over `launch_size` on the device `dev`.
`grid_size` is the size of the grid (with reduced dimensions flattened), which
the GPU heuristic uses regardless of the dimensions the kernel spans.
"""
@inline workgroup_layout(dev, grid_size, launch_size) = heuristic_workgroup(grid_size...)

# The KernelAbstractions CPU backend runs each block as a loop over `CartesianIndices(workgroup)`,
# so a workgroup spanning the first dimension of `launch_size` yields a single contiguous inner loop.
@inline workgroup_layout(::KA.CPU, grid_size, launch_size) = cpu_workgroup(launch_size...)

@inline cpu_workgroup(W1::Int) = W1
@inline cpu_workgroup(W1::Int, W2::Int, Wz...) = W1 == 1 ? (1, W2) : (W1, 1)

# To be extended in the `Grids` modules for non-trivial peripheries,
# for all other cases, `periphery_offset` is zero.
periphery_offset(loc, grid, side) = 0

# Returns the default worksize given a particular grid
# defaults to the size of the grid.
@inline worksize(grid) = size(grid)

"""
$(TYPEDSIGNATURES)

Returns the `workgroup` and `worksize` for launching a kernel over `dims`
on `grid` that excludes peripheral nodes (determined from `location`).
The `workgroup` is a tuple specifying the threads per block in each
dimension. The `worksize` specifies the range of the loop in each dimension.

For more information, see: https://github.com/CliMA/Oceananigans.jl/pull/308
"""
@inline select_dims(::Val{:xyz}, x, y, z) = (x, y, z)
@inline select_dims(::Val{:xy},  x, y, z) = (x, y)
@inline select_dims(::Val{:xz},  x, y, z) = (x, z)
@inline select_dims(::Val{:yz},  x, y, z) = (y, z)

@inline function interior_work_layout(dev, grid, workdims::Val, (ℓx, ℓy, ℓz))
    Fx, Fy, Fz = worksize(grid)

    ox = periphery_offset(ℓx, grid, Val(1))
    oy = periphery_offset(ℓy, grid, Val(2))
    oz = periphery_offset(ℓz, grid, Val(3))

    Wx, Wy, Wz = (Fx-ox, Fy-oy, Fz-oz)
    launch_size = select_dims(workdims, Wx, Wy, Wz)
    workgroup = StaticSize(workgroup_layout(dev, (Wx, Wy, Wz), launch_size))

    range = contiguousrange(launch_size, select_dims(workdims, ox, oy, oz))

    return workgroup, OffsetStaticSize(range)
end

"""
$(TYPEDSIGNATURES)

Returns the `workgroup` and `worksize` for launching a kernel over `dims`
on `grid`. The `workgroup` is a tuple specifying the threads per block in each
dimension. The `worksize` specifies the range of the loop in each dimension.

For more information, see: https://github.com/CliMA/Oceananigans.jl/pull/308
"""
@inline function work_layout(dev, grid, workdims::Val, reduced_dimensions)
    Fx, Fy, Fz = worksize(grid)
    Wx, Wy, Wz = flatten_reduced_dimensions((Fx, Fy, Fz), reduced_dimensions) # this seems to be for halo filling
    launch_size = select_dims(workdims, Wx, Wy, Wz)
    workgroup = workgroup_layout(dev, (Wx, Wy, Wz), launch_size)
    return StaticSize(workgroup), StaticSize(launch_size)
end

@inline work_layout(dev, grid, workdims::Symbol, reduced_dimensions) = work_layout(dev, grid, Val(workdims), reduced_dimensions)
@inline interior_work_layout(dev, grid, workdims::Symbol, location) = interior_work_layout(dev, grid, Val(workdims), location)

@inline function work_layout(dev, grid, worksize::NTuple{N, Int}, reduced_dimensions) where N
    workgroup = workgroup_layout(dev, worksize, worksize)
    return StaticSize(workgroup), StaticSize(worksize)
end

@inline function offset_work_layout(dev, grid, ::KernelParameters{spec, offsets}, reduced_dimensions) where {spec, offsets}
    workgroup, worksize = work_layout(dev, grid, spec, reduced_dimensions)
    range = contiguousrange(worksize, offsets)
    return  workgroup, OffsetStaticSize(range)
end

"""
    configure_kernel(arch, grid, workspec, kernel!;
                     exclude_periphery = false,
                     reduced_dimensions = (),
                     location = nothing)

Configure `kernel!` to launch over the `dims` of `grid` on
the architecture `arch`.

Arguments
=========

- `arch`: The architecture on which the kernel will be launched.
- `grid`: The grid on which the kernel will be executed.
- `workspec`: The workspec that defines the work distribution: a `Symbol` such as `:xyz`, a tuple of sizes,
              `KernelParameters`, or an active cells map, i.e. an array of `(i, j[, k])` indices. A kernel launched
              over an active cells map is a linear kernel with one work item per index in the map, which has to be
              launched with `ndrange` equal to the returned `IndexMap`.
- `kernel!`: The kernel function to be executed.

Keyword Arguments
=================

- `reduced_dimensions`: A tuple specifying the dimensions to be reduced in the work distribution. Default is an empty tuple.
- `location`: The location of the kernel execution, used when `exclude_periphery = true`. Default is `nothing`.
- `exclude_periphery`: A boolean indicating whether to exclude the periphery, used only for interior kernels.
"""
@inline configure_kernel(arch, grid, workspec::Symbol, kernel!; kwargs...) =
    configure_kernel(arch, grid, Val(workspec), kernel!; kwargs...)

@inline function configure_kernel(arch, grid, workspec, kernel!;
                                  exclude_periphery = false,
                                  reduced_dimensions = (),
                                  location = nothing)

    # Transform keyword arguments into arguments to be able to dispatch correctly
    return configure_kernel(arch, grid, workspec, kernel!, Val(exclude_periphery);
                            reduced_dimensions,
                            location)
end

@inline function configure_kernel(arch, grid, workspec, kernel!, ::Val;
                                  reduced_dimensions = (),
                                  location = nothing)

    dev  = Architectures.device(arch)
    workgroup, worksize = work_layout(dev, grid, workspec, reduced_dimensions)
    loop = kernel!(dev, workgroup, worksize)

    return loop, worksize::StaticSize
end

# With a "true" exclude_periphery, we use the `interior_work_layout` function
@inline function configure_kernel(arch, grid, workspec::Val, kernel!, ::Val{true};
                                  reduced_dimensions = (),
                                  location = nothing)

    dev  = Architectures.device(arch)
    workgroup, worksize = interior_work_layout(dev, grid, workspec, location)
    loop = kernel!(dev, workgroup, worksize)

    return loop, worksize::OffsetStaticSize
end

# When there are KernelParameters, we use the `offset_work_layout` function
@inline function configure_kernel(arch, grid, workspec::KernelParameters, kernel!, ::Val;
                                  reduced_dimensions = (), kwargs...)

    dev  = Architectures.device(arch)
    workgroup, worksize = offset_work_layout(dev, grid, workspec, reduced_dimensions)
    loop = kernel!(dev, workgroup, worksize)

    return loop, worksize::OffsetStaticSize
end

# An active cells map launches a linear kernel with one work item per index it holds (see `IndexMap`)
@inline function configure_kernel(arch, grid, active_cells_map::AbstractArray, kernel!, ::Val; kwargs...)
    dev  = Architectures.device(arch)
    loop = kernel!(dev, StaticSize((256,)), NDIteration.DynamicSize())
    return loop, IndexMap(active_cells_map)
end

"""
$(TYPEDSIGNATURES)

Launches `kernel!` with arguments `kernel_args`
over the `dims` of `grid` on the architecture `arch`.
Kernels run on the default stream.

See [configure_kernel](@ref) for more information and also a list of the
keyword arguments `kw`.
"""
@inline launch!(arch, grid, workspec, kernel!, kernel_args::Vararg{Any, N}; kwargs...) where N = _launch!(arch, grid, workspec, kernel!, kernel_args...; kwargs...)

@inline launch!(arch, grid, workspec::NTuple{M, Int}, kernel!, kernel_args::Vararg{Any, N}; kwargs...) where {M, N} =
    _launch!(arch, grid, workspec, kernel!, kernel_args...; kwargs...)

@inline function launch!(arch, grid, workspec_tuple::Tuple, kernel!, kernel_args::Vararg{Any, N}; kwargs...) where N
    _launch!(arch, grid, first(workspec_tuple), kernel!, kernel_args...; kwargs...)
    launch!(arch, grid, Base.tail(workspec_tuple), kernel!, kernel_args...; kwargs...)
    return nothing
end

@inline launch!(arch, grid, ::Tuple{}, kernel!, kernel_args...; kwargs...) = nothing

@inline launch!(arch, grid, workspec::Symbol, kernel!, kernel_args::Vararg{Any, N}; kw...) where N = _launch!(arch, grid, Val(workspec), kernel!, kernel_args...; kw...)
@inline launch!(arch, grid, workspec::Val, kernel!, kernel_args::Vararg{Any, N}; kw...) where N = _launch!(arch, grid, workspec, kernel!, kernel_args...; kw...)

# Inner interface
@inline function _launch!(arch, grid, workspec, kernel!, kernel_args::Vararg{Any, N};
                          exclude_periphery = false,
                          reduced_dimensions = ()) where N

    workspec = possibly_load_active_cells_map(grid, workspec, exclude_periphery)

    location = Oceananigans.instantiated_location(first(kernel_args))

    loop!, worksize = configure_kernel(arch, grid, workspec, kernel!, Val(exclude_periphery);
                                       location,
                                       reduced_dimensions)

    # Don't launch kernels with no size
    if length(worksize) > 0
        loop!(Architectures.convert_to_device(arch, kernel_args)...; ndrange = launch_ndrange(worksize))
    end

    return nothing
end

# `:xyz` and `:xy` launch over the corresponding active cells map of `grid`, if it has one
@inline possibly_load_active_cells_map(grid, workspec, exclude_periphery) = workspec

@inline function possibly_load_active_cells_map(grid, workspec::Union{Val{:xyz}, Val{:xy}}, exclude_periphery)
    exclude_periphery && return workspec
    return something(get_active_cells_map(grid, workspec), workspec)
end

#####
##### Extension to KA for offset indices: to remove when implemented in KA
##### Allows to use `launch!` with offsets, e.g.:
##### `launch!(arch, grid, KernelParameters(size, offsets), kernel!; kernel_args...)`
##### where offsets is a tuple containing the offset to pass to @index
##### Note that this syntax is only usable in conjunction with the `launch!` function and
##### will have no effect if the kernel is launched with `kernel!` directly.
##### To achieve the same result with kernel launching, the correct syntax is:
##### `kernel!(arch, StaticSize(size), OffsetStaticSize(contiguousrange(size, offset)))`
##### Using offsets is (at the moment) incompatible with dynamic workgroup sizes: in case of offset dynamic kernels
##### offsets will have to be passed manually.
#####

# TODO: when offsets are implemented in KA so that we can call `kernel(dev, group, size, offsets)`, remove all of this
import KernelAbstractions: partition
import KernelAbstractions: __ndrange, __groupsize

struct OffsetStaticSize{S} <: _Size
    function OffsetStaticSize{S}() where S
        new{S::Tuple{Vararg}}()
    end
end

@pure OffsetStaticSize(s::Tuple{}) = OffsetStaticSize{s}()
@pure OffsetStaticSize(s::Tuple{Vararg{Int}}) = OffsetStaticSize{s}()
@pure OffsetStaticSize(s::Int...) = OffsetStaticSize{s}()
@pure OffsetStaticSize(s::Type{<:Tuple}) = OffsetStaticSize{tuple(s.parameters...)}()
@pure OffsetStaticSize(s::Tuple{Vararg{UnitRange{Int}}}) = OffsetStaticSize{s}()

# Some @pure convenience functions for `OffsetStaticSize` (following `StaticSize` in KA)
@pure get(::Type{OffsetStaticSize{S}}) where {S} = S
@pure get(::OffsetStaticSize{S}) where {S} = S
@pure Base.getindex(::OffsetStaticSize{S}, i::Int) where {S} = i <= length(S) ? S[i] : 1
@pure Base.ndims(::OffsetStaticSize{S}) where {S}  = length(S)
@pure Base.length(::OffsetStaticSize{S}) where {S} = prod(map(worksize, S))

@inline getrange(::OffsetStaticSize{S}) where {S} = worksize(S), offsets(S)
@inline getrange(::Type{OffsetStaticSize{S}}) where {S} = worksize(S), offsets(S)

# Makes sense to explicitly define the offsets for up to 3 dimensions,
# since Oceananigans typically runs kernels with up to 3 dimensions.
@inline offsets(ranges::NTuple{1, UnitRange}) = @inbounds (ranges[1].start - 1, )
@inline offsets(ranges::NTuple{2, UnitRange}) = @inbounds (ranges[1].start - 1, ranges[2].start - 1)
@inline offsets(ranges::NTuple{3, UnitRange}) = @inbounds (ranges[1].start - 1, ranges[2].start - 1, ranges[3].start - 1)

# Generic case for any number of dimensions
@inline offsets(ranges::NTuple{N, UnitRange}) where N = @inbounds Tuple(ranges[t].start - 1 for t in 1:N)

@inline worksize(t::Tuple) = map(worksize, t)
@inline worksize(sz::Int) = sz
@inline worksize(r::AbstractUnitRange) = length(r)

const OffsetNDRange{N, S} = NDRange{N, <:StaticSize, <:StaticSize, <:Any, <:OffsetStaticSize{S}} where {N, S}

# NDRange has been modified to have offsets in place of workitems: Remember, dynamic offset kernels are not possible with this extension!!
# TODO: maybe don't do this
@inline function expand(ndrange::OffsetNDRange{N, S}, groupidx::CartesianIndex{N}, idx::CartesianIndex{N}) where {N, S}
    nI = ntuple(Val(N)) do I
        Base.@_inline_meta
        offsets = workitems(ndrange)
        stride = size(offsets, I)
        gidx = groupidx.I[I]
        (gidx - 1) * stride + idx.I[I] + S[I]
    end
    return CartesianIndex(nI)
end

@inline __ndrange(::CompilerMetadata{NDRange}) where {NDRange<:OffsetStaticSize}  = CartesianIndices(get(NDRange))
@inline __groupsize(cm::CompilerMetadata{NDRange}) where {NDRange<:OffsetStaticSize} = size(__ndrange(cm))

# Kernel{<:Any, <:StaticSize, <:StaticSize} and Kernel{<:Any, <:StaticSize, <:OffsetStaticSize} are the only kernels used by Oceananigans
const OffsetKernel = Kernel{<:Any, <:StaticSize, <:OffsetStaticSize}

# Extending the partition function to include offsets in NDRange: note that in this case the
# offsets take the place of the DynamicWorkitems which we assume is not needed in static kernels
function partition(kernel::OffsetKernel, inrange, ingroupsize)
    static_ndrange = ndrange(kernel)
    static_workgroupsize = workgroupsize(kernel)

    if inrange !== nothing && inrange != get(static_ndrange)
        error("Static NDRange ($static_ndrange) and launch NDRange ($inrange) differ")
    end

    range, offsets = getrange(static_ndrange)

    if static_workgroupsize <: StaticSize
        if ingroupsize !== nothing && ingroupsize != get(static_workgroupsize)
            error("Static WorkgroupSize ($static_workgroupsize) and launch WorkgroupSize $(ingroupsize) differ")
        end
        groupsize = get(static_workgroupsize)
    end

    @assert groupsize !== nothing
    @assert range !== nothing
    blocks, groupsize, dynamic = NDIteration.partition(range, groupsize)

    static_blocks = StaticSize{blocks}
    static_workgroupsize = StaticSize{groupsize} # we might have padded workgroupsize

    iterspace = NDRange{length(range), static_blocks, static_workgroupsize}(blocks, OffsetStaticSize(offsets))

    return iterspace, dynamic
end

#####
##### Index maps: kernels running one work item per listed index
#####
##### `launch!` passes an `IndexMap` as the `ndrange` of a kernel with a static one-dimensional
##### workgroup. `partition` builds the blocked iteration space with the map as `NDRange` mapping,
##### `expand` looks the index of a work item up in the map, and the `ndrange` of the context is a
##### `MappedIndices`, in which every index but `invalid_index` is contained, so KernelAbstractions'
##### `expand(iterspace, group, item) in ndrange` validity check for custom mappings works unchanged.
#####

"""
$(TYPEDSIGNATURES)

Iteration space given by the indices listed in `map`, whose elements are `CartesianIndex{N}`
or `NTuple{N, Integer}`. Work item `p` handles the index `map[p]`.
"""
struct IndexMap{N, A <: AbstractVector}
    map :: A
    IndexMap{N}(map::AbstractVector) where N = new{N, typeof(map)}(map)
end

IndexMap(map::AbstractVector) = IndexMap{mapdims(eltype(map))}(map)

mapdims(::Type{CartesianIndex{N}}) where N = N
mapdims(::Type{<:NTuple{N, Integer}}) where N = N

Base.length(m::IndexMap) = length(m.map)
Base.@propagate_inbounds Base.getindex(m::IndexMap{N}, p::Integer) where N = CartesianIndex{N}(m.map[p])

Adapt.adapt_structure(to, m::IndexMap{N}) where N = IndexMap{N}(Adapt.adapt(to, m.map))

# `ndrange` to launch a configured kernel with: `nothing` for a static worksize, the index map for an active cells map
@inline launch_ndrange(worksize) = nothing
@inline launch_ndrange(map::IndexMap) = map

# `CartesianIndex` returned for a work item past the end of the map
@inline invalid_index(::Val{N}) where N = CartesianIndex(ntuple(_ -> typemin(Int), Val(N)))

@inline mapped_index(m::IndexMap{N}, p::Integer) where N = p <= length(m) ? (@inbounds m[p]) : invalid_index(Val(N))

"""
$(TYPEDSIGNATURES)

The `ndrange` of a launch over an `IndexMap` with `N`-dimensional indices: every
`CartesianIndex{N}` but the `invalid_index` is contained in it.
"""
struct MappedIndices{N}
    length :: Int
end

MappedIndices(m::IndexMap{N}) where N = MappedIndices{N}(length(m))

Base.length(r::MappedIndices) = r.length
Base.size(r::MappedIndices) = (r.length,)
@inline Base.in(I::CartesianIndex{N}, ::MappedIndices{N}) where N = I != invalid_index(Val(N))

KA.cartesian(m::IndexMap) = MappedIndices(m)
KA.cartesian(r::MappedIndices) = r

const MappedNDRange = NDRange{1, <:Any, <:Any, <:Any, <:Any, <:IndexMap}

function partition(kernel::Kernel{<:Any, <:StaticSize, <:NDIteration.DynamicSize}, map::IndexMap, ingroupsize)
    static_workgroupsize = workgroupsize(kernel)
    items = NDIteration.get(static_workgroupsize)
    length(items) == 1 || throw(ArgumentError("A kernel launched over an index map needs a one-dimensional workgroup, got $items"))
    blocks, _, dynamic = NDIteration.partition((length(map),), items)
    iterspace = NDRange{1, NDIteration.DynamicSize, static_workgroupsize}(CartesianIndices(blocks), nothing, map)
    return iterspace, dynamic
end

# Position in the map of work item `idx` of workgroup `groupidx`
@inline mapped_position(ndrange::MappedNDRange, groupidx::Integer, idx::Integer) = (groupidx - 1) * length(workitems(ndrange)) + idx
@inline mapped_position(ndrange::MappedNDRange, groupidx::CartesianIndex{1}, idx::CartesianIndex{1}) = mapped_position(ndrange, groupidx.I[1], idx.I[1])
@inline mapped_position(ndrange::MappedNDRange, groupidx::CartesianIndex{1}, idx::Integer) = mapped_position(ndrange, groupidx.I[1], idx)
@inline mapped_position(ndrange::MappedNDRange, groupidx::Integer, idx::CartesianIndex{1}) = mapped_position(ndrange, groupidx, idx.I[1])

@inline expand(ndrange::MappedNDRange, groupidx::Integer, idx::Integer) = mapped_index(ndrange.mapping, mapped_position(ndrange, groupidx, idx))
@inline expand(ndrange::MappedNDRange, groupidx::CartesianIndex{1}, idx::CartesianIndex{1}) = mapped_index(ndrange.mapping, mapped_position(ndrange, groupidx, idx))
@inline expand(ndrange::MappedNDRange, groupidx::CartesianIndex{1}, idx::Integer) = mapped_index(ndrange.mapping, mapped_position(ndrange, groupidx, idx))
@inline expand(ndrange::MappedNDRange, groupidx::Integer, idx::CartesianIndex{1}) = mapped_index(ndrange.mapping, mapped_position(ndrange, groupidx, idx))

# The linear index of a mapped work item is its position in the map
@inline NDIteration.linear_index(ndrange::MappedNDRange, ::MappedIndices, groupidx::CartesianIndex{1}, idx::CartesianIndex{1}) =
    mapped_position(ndrange, groupidx, idx)
