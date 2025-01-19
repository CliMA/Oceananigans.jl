#####
##### Utilities for launching kernels
#####

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Grids: AbstractGrid
using Base: @pure

import Oceananigans
import KernelAbstractions: get, expand

struct KernelParameters{S, O} end

"""
    KernelParameters(size, offsets)

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
KernelParameters(size, offsets) = KernelParameters{size, offsets}()

"""
    KernelParameters(range1, [range2, range3])

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
function KernelParameters(r::UnitRange)
    size = length(r)
    offset = first(r) - 1
    return KernelParameters(tuple(size), tuple(offset))
end

function KernelParameters(r1::UnitRange, r2::UnitRange)
    size = (length(r1), length(r2))
    offsets = (first(r1) - 1, first(r2) - 1)
    return KernelParameters(size, offsets)
end

function KernelParameters(r1::UnitRange, r2::UnitRange, r3::UnitRange)
    size = (length(r1), length(r2), length(r3))
    offsets = (first(r1) - 1, first(r2) - 1, first(r3) - 1)
    return KernelParameters(size, offsets)
end

contiguousrange(range::NTuple{N, Int}, offset::NTuple{N, Int}) where N = Tuple(1+o:r+o for (r, o) in zip(range, offset))
flatten_reduced_dimensions(worksize, dims) = Tuple(d ∈ dims ? 1 : worksize[d] for d = 1:3)

# This supports 2D, 3D and 4D work sizes (but the 3rd and 4th dimension are discarded)
function heuristic_workgroup(Wx, Wy, Wz=nothing, Wt=nothing)

    workgroup = Wx == 1 && Wy == 1 ?

                    # One-dimensional column models:
                    (1, 1) :

                Wx == 1 ?

                    # Two-dimensional y-z slice models:
                    (1, min(256, Wy)) :

                Wy == 1 ?

                    # Two-dimensional x-z slice models:
                    (min(256, Wx), 1) :

                    # Three-dimensional models
                    (16, 16)

    return workgroup
end

periphery_offset(loc, topo, N) = 0
periphery_offset(::Face, ::Bounded, N) = ifelse(N > 1, 1, 0)

drop_omitted_dims(::Val{:xyz}, xyz) = xyz
drop_omitted_dims(::Val{:xy}, (x, y, z)) = (x, y)
drop_omitted_dims(::Val{:xz}, (x, y, z)) = (x, z)
drop_omitted_dims(::Val{:yz}, (x, y, z)) = (y, z)
drop_omitted_dims(workdims, xyz) = throw(ArgumentError("Unsupported launch configuration: $workdims"))
    
"""
    interior_work_layout(grid, dims, location)

Returns the `workgroup` and `worksize` for launching a kernel over `dims`
on `grid` that excludes peripheral nodes.
The `workgroup` is a tuple specifying the threads per block in each
dimension. The `worksize` specifies the range of the loop in each dimension.

Specifying `include_right_boundaries=true` will ensure the work layout includes the
right face end points along bounded dimensions. This requires the field `location`
to be specified.

For more information, see: https://github.com/CliMA/Oceananigans.jl/pull/308
"""
@inline function interior_work_layout(grid, workdims::Symbol, location)
    valdims = Val(workdims)
    Nx, Ny, Nz = size(grid)

    # just an example for :xyz
    ℓx, ℓy, ℓz = map(instantiate, location)
    tx, ty, tz = map(instantiate, topology(grid))

    # Offsets
    ox = periphery_offset(ℓx, tx, Nx)
    oy = periphery_offset(ℓy, ty, Ny)
    oz = periphery_offset(ℓz, tz, Nz)

    # Worksize
    Wx, Wy, Wz = (Nx-ox, Ny-oy, Nz-oz)
    workgroup = heuristic_workgroup(Wx, Wy, Wz)
    workgroup = StaticSize(workgroup)

    # Adapt to workdims
    worksize = drop_omitted_dims(valdims, (Wx, Wy, Wz))
    offsets = drop_omitted_dims(valdims, (ox, oy, oz))
    range = contiguousrange(worksize, offsets)
    worksize = OffsetStaticSize(range)

    return workgroup, worksize
end

"""
    work_layout(grid, dims, location)

Returns the `workgroup` and `worksize` for launching a kernel over `dims`
on `grid`. The `workgroup` is a tuple specifying the threads per block in each
dimension. The `worksize` specifies the range of the loop in each dimension.

Specifying `include_right_boundaries=true` will ensure the work layout includes the
right face end points along bounded dimensions. This requires the field `location`
to be specified.

For more information, see: https://github.com/CliMA/Oceananigans.jl/pull/308
"""
@inline function work_layout(grid, workdims::Symbol, reduced_dimensions)
    valdims = Val(workdims)
    Nx, Ny, Nz = size(grid)
    Wx, Wy, Wz = flatten_reduced_dimensions((Nx, Ny, Nz), reduced_dimensions) # this seems to be for halo filling
    workgroup = heuristic_workgroup(Wx, Wy, Wz)
    worksize = drop_omitted_dims(valdims, (Wx, Wy, Wz))
    return workgroup, worksize
end

function work_layout(grid, worksize::NTuple{N, Int}, reduced_dimensions) where N
    workgroup = heuristic_workgroup(worksize...)
    return workgroup, worksize
end

function work_layout(grid, ::KernelParameters{spec, offsets}, reduced_dimensions) where {spec, offsets}
    workgroup, worksize = work_layout(grid, spec, reduced_dimensions)
    static_workgroup = StaticSize(workgroup)
    range = contiguousrange(worksize, offsets)
    offset_worksize = OffsetStaticSize(range)
    return static_workgroup, offset_worksize
end

"""
    configure_kernel(arch, grid, workspec, kernel!;
                     exclude_periphery = false,
                     reduced_dimensions = (),
                     location = nothing,
                     active_cells_map = nothing,
                     only_local_halos = false,
                     async = false)

Configure `kernel!` to launch over the `dims` of `grid` on
the architecture `arch`.

# Arguments
============

- `arch`: The architecture on which the kernel will be launched.
- `grid`: The grid on which the kernel will be executed.
- `workspec`: The workspec that defines the work distribution.
- `kernel!`: The kernel function to be executed.

# Keyword Arguments
====================

- `include_right_boundaries`: A boolean indicating whether to include right boundaries `(N + 1)`. Default is `false`.
- `reduced_dimensions`: A tuple specifying the dimensions to be reduced in the work distribution. Default is an empty tuple.
- `location`: The location of the kernel execution, needed for `include_right_boundaries`. Default is `nothing`.
- `active_cells_map`: A map indicating the active cells in the grid. If the map is not a nothing, the workspec will be disregarded and 
                      the kernel is configured as a linear kernel with a worksize equal to the length of the active cell map. Default is `nothing`.
"""
@inline function configure_kernel(arch, grid, workspec, kernel!;
                                  exclude_periphery = false,
                                  reduced_dimensions = (),
                                  location = nothing,
                                  active_cells_map = nothing)

    if !isnothing(active_cells_map) # everything else is irrelevant
        workgroup = min(length(active_cells_map), 256)
        worksize = length(active_cells_map)
    elseif exclude_periphery && !(workspec isa KernelParameters) # TODO: support KernelParameters
        workgroup, worksize = interior_work_layout(grid, workspec, location)
    else
        workgroup, worksize = work_layout(grid, workspec, reduced_dimensions)
    end

    dev = Architectures.device(arch)
    loop = kernel!(dev, workgroup, worksize)
    return loop, worksize
end

       
"""
    launch!(arch, grid, workspec, kernel!, kernel_args...; kw...)

Launches `kernel!` with arguments `kernel_args`
over the `dims` of `grid` on the architecture `arch`.
Kernels run on the default stream.

See [configure_kernel](@ref) for more information and also a list of the
keyword arguments `kw`.
"""
@inline launch!(args...; kwargs...) = _launch!(args...; kwargs...)

@inline launch!(arch, grid, workspec::NTuple{N, Int}, args...; kwargs...) where N =
    _launch!(arch, grid, workspec, args...; kwargs...)
 
@inline function launch!(arch, grid, workspec_tuple::Tuple, args...; kwargs...)
    for workspec in workspec_tuple
        _launch!(arch, grid, workspec, args...; kwargs...)
    end
    return nothing
end
 
# When dims::Val
@inline launch!(arch, grid, ::Val{workspec}, args...; kw...) where workspec =
    _launch!(arch, grid, workspec, args...; kw...)

# Inner interface
@inline function _launch!(arch, grid, workspec, kernel!, first_kernel_arg, other_kernel_args...;
                          exclude_periphery = false,
                          reduced_dimensions = (),
                          active_cells_map = nothing,
                          # TODO: these two kwargs do nothing:
                          only_local_halos = false,
                          async = false)

    location = Oceananigans.Grids.location(first_kernel_arg)

    loop!, worksize = configure_kernel(arch, grid, workspec, kernel!;
                                       location,
                                       exclude_periphery,
                                       reduced_dimensions,
                                       active_cells_map)
                                       
    # Don't launch kernels with no size
    if worksize != 0
        loop!(first_kernel_arg, other_kernel_args...)
    end

    return nothing
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

using KernelAbstractions: Kernel
using KernelAbstractions.NDIteration: _Size, StaticSize
using KernelAbstractions.NDIteration: NDRange

struct OffsetStaticSize{S} <: _Size
    function OffsetStaticSize{S}() where S
        new{S::Tuple{Vararg}}()
    end
end

@pure OffsetStaticSize(s::Tuple{Vararg{Int}}) = OffsetStaticSize{s}() 
@pure OffsetStaticSize(s::Int...) = OffsetStaticSize{s}() 
@pure OffsetStaticSize(s::Type{<:Tuple}) = OffsetStaticSize{tuple(s.parameters...)}()
@pure OffsetStaticSize(s::Tuple{Vararg{UnitRange{Int}}}) = OffsetStaticSize{s}()

# Some @pure convenience functions for `OffsetStaticSize` (following `StaticSize` in KA)
@pure get(::Type{OffsetStaticSize{S}}) where {S} = S
@pure get(::OffsetStaticSize{S}) where {S} = S
@pure Base.getindex(::OffsetStaticSize{S}, i::Int) where {S} = i <= length(S) ? S[i] : 1
@pure Base.ndims(::OffsetStaticSize{S}) where {S}  = length(S)
@pure Base.length(::OffsetStaticSize{S}) where {S} = prod(worksize.(S))

@inline getrange(::OffsetStaticSize{S}) where {S} = worksize(S), offsets(S)
@inline getrange(::Type{OffsetStaticSize{S}}) where {S} = worksize(S), offsets(S)

@inline offsets(ranges::Tuple{Vararg{UnitRange}}) = Tuple(r.start - 1 for r in ranges)

@inline worksize(i::Tuple) = worksize.(i)
@inline worksize(i::Int) = i
@inline worksize(i::UnitRange) = length(i)

"""a type used to store offsets in `NDRange` types"""
struct KernelOffsets{O}
    offsets :: O
end

Base.getindex(o::KernelOffsets, args...) = getindex(o.offsets, args...)

const OffsetNDRange{N} = NDRange{N, <:StaticSize, <:StaticSize, <:Any, <:KernelOffsets} where N

# NDRange has been modified to have offsets in place of workitems: Remember, dynamic offset kernels are not possible with this extension!!
# TODO: maybe don't do this
@inline function expand(ndrange::OffsetNDRange{N}, groupidx::CartesianIndex{N}, idx::CartesianIndex{N}) where {N}
    nI = ntuple(Val(N)) do I
        Base.@_inline_meta
        offsets = workitems(ndrange)
        stride = size(offsets, I)
        gidx = groupidx.I[I]
        (gidx - 1) * stride + idx.I[I] + ndrange.workitems[I]
    end
    return CartesianIndex(nI)
end

using KernelAbstractions.NDIteration
using KernelAbstractions: ndrange, workgroupsize
import KernelAbstractions: partition

using KernelAbstractions: CompilerMetadata
import KernelAbstractions: __ndrange, __groupsize

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
    
    iterspace = NDRange{length(range), static_blocks, static_workgroupsize}(blocks, KernelOffsets(offsets))

    return iterspace, dynamic
end

