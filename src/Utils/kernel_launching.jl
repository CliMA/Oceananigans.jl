#####
##### Utilities for launching kernels
#####

using Oceananigans.Architectures
using Oceananigans.Utils
using Oceananigans.Grids

using Oceananigans.Grids: AbstractGrid
import Base

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
launch!(arch, grid, kp, kernel!; kernel_args...)
```

See the documentation for [`launch!`](@ref).
"""
KernelParameters(size, offsets) = KernelParameters{size, offsets}()

offsets(::KernelParameters{S, O}) where {S, O} = O
offsets(workspec)  = nothing

contiguousrange(range::NTuple{N, Int}, offset::NTuple{N, Int}) where N = Tuple(1+o:r+o for (r, o) in zip(range, offset))

flatten_reduced_dimensions(worksize, dims) = Tuple(i ∈ dims ? 1 : worksize[i] for i = 1:3)

function heuristic_workgroup(Wx, Wy, Wz=nothing)

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

work_layout(grid, ::KernelParameters{worksize}; kw...) where worksize =
    work_layout(grid, worksize; kw...)

function work_layout(grid, worksize::Tuple; kw...)
    workgroup = heuristic_workgroup(worksize...)
    return workgroup, worksize
end

"""
    work_layout(grid, dims; include_right_boundaries=false, location=nothing)

Returns the `workgroup` and `worksize` for launching a kernel over `dims`
on `grid`. The `workgroup` is a tuple specifying the threads per block in each
dimension. The `worksize` specifies the range of the loop in each dimension.

Specifying `include_right_boundaries=true` will ensure the work layout includes the
right face end points along bounded dimensions. This requires the field `location`
to be specified.

For more information, see: https://github.com/CliMA/Oceananigans.jl/pull/308
"""
function work_layout(grid, workdims::Symbol; include_right_boundaries=false, location=nothing, reduced_dimensions=())

    Nx′, Ny′, Nz′ = include_right_boundaries ? size(location, grid) : size(grid)
    Nx′, Ny′, Nz′ = flatten_reduced_dimensions((Nx′, Ny′, Nz′), reduced_dimensions)

    workgroup = heuristic_workgroup(Nx′, Ny′, Nz′)

    # Drop omitted dimemsions
    worksize = workdims == :xyz ? (Nx′, Ny′, Nz′) :
               workdims == :xy  ? (Nx′, Ny′) :
               workdims == :xz  ? (Nx′, Nz′) :
               workdims == :yz  ? (Ny′, Nz′) : throw(ArgumentError("Unsupported launch configuration: $workdims"))

    return workgroup, worksize
end

@inline active_cells_work_layout(workgroup, worksize, only_active_cells, grid) = workgroup, worksize

"""
    launch!(arch, grid, layout, kernel!, args...; kwargs...)

Launches `kernel!`, with arguments `args` and keyword arguments `kwargs`,
over the `dims` of `grid` on the architecture `arch`. kernels run on the default stream
"""
function launch!(arch, grid, workspec, kernel!, kernel_args...;
                 include_right_boundaries = false,
                 reduced_dimensions = (),
                 location = nothing,
                 only_active_cells = nothing,
                 kwargs...)


    loop! = configured_kernel(arch, grid, workspec, kernel!;
                              include_right_boundaries,
                              reduced_dimensions,
                              location,
                              only_active_cells,
                              kwargs...)
    
    
    !isnothing(loop!) && loop!(kernel_args...)
    
    return nothing
end

function configured_kernel(arch, grid, workspec, kernel!;
                           include_right_boundaries = false,
                           reduced_dimensions = (),
                           location = nothing,
                           only_active_cells = nothing,
                           kwargs...)

    workgroup, worksize = work_layout(grid, workspec;
                                    include_right_boundaries,
                                    reduced_dimensions,
                                    location)


    offset = offsets(workspec)

    if !isnothing(only_active_cells) 
        workgroup, worksize = active_cells_work_layout(workgroup, worksize, only_active_cells, grid) 
        offset = nothing
    end

    if worksize == 0
        return nothing
    end
    
    # We can only launch offset kernels with Static sizes!!!!
    loop! = isnothing(offset) ? kernel!(Architectures.device(arch), workgroup, worksize) : 
                                kernel!(Architectures.device(arch), StaticSize(workgroup), OffsetStaticSize(contiguousrange(worksize, offset))) 

    return loop!
end
        
# When dims::Val
@inline launch!(arch, grid, ::Val{workspec}, args...; kwargs...) where workspec =
    launch!(arch, grid, workspec, args...; kwargs...)

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

import Base
import Base: @pure
import KernelAbstractions: get, expand

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
@inline function expand(ndrange::OffsetNDRange{N}, groupidx::CartesianIndex{N}, idx::CartesianIndex{N}) where {N}
    nI = ntuple(Val(N)) do I
        Base.@_inline_meta
        stride = size(workitems(ndrange), I)
        gidx = groupidx.I[I]
        (gidx-1)*stride + idx.I[I] + ndrange.workitems[I]
    end
    CartesianIndex(nI)
end

using KernelAbstractions.NDIteration
using KernelAbstractions: ndrange, workgroupsize
import KernelAbstractions: partition

using KernelAbstractions: CompilerMetadata
import KernelAbstractions: __ndrange

@inline __ndrange(::CompilerMetadata{NDRange}) where {NDRange<:OffsetStaticSize}  = CartesianIndices(get(NDRange))

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
