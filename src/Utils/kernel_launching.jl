#####
##### Utilities for launching kernels
#####

using KernelAbstractions
using Oceananigans.Architectures
using Oceananigans.Grids

flatten_reduced_dimensions(worksize, dims) = Tuple(i ∈ dims ? 1 : worksize[i] for i = 1:3)

const MAX_GPU_THREADS_PER_BLOCK = 256

function heuristic_workgroup(arch, Nx, Ny, Nz)
    # One-dimensional column models:
    Nx == 1 && Ny == 1 && return (1, 1)

    # Two-dimensional y-z slice models:
    Nx == 1 && return (1, min(MAX_GPU_THREADS_PER_BLOCK, Ny))

    # Two-dimensional x-z slice models:
    Ny == 1 && return (1, min(MAX_GPU_THREADS_PER_BLOCK, Nx))

    # Three-dimensional GPU models
    arch isa GPU && return (Int(√MAX_GPU_THREADS_PER_BLOCK), Int(√MAX_GPU_THREADS_PER_BLOCK))

    # Three-dimensional CPU models
    return (Nx, Ny)
end

"""
    work_layout(arch, grid, dims; include_right_boundaries=false, location=nothing)

Returns the `workgroup` and `worksize` for launching a kernel over `dims`
on `grid` and `arch`itecture. The `workgroup` is a tuple specifying the threads per block in each
dimension. The `worksize` specifies the range of the loop in each dimension.

Specifying `include_right_boundaries=true` will ensure the work layout includes the
right face end points along bounded dimensions. This requires the field `location`
to be specified.

For more information, see: https://github.com/CliMA/Oceananigans.jl/pull/308
"""
function work_layout(arch, grid, dims; include_right_boundaries=false, location=nothing, reduced_dimensions=())

    Nx, Ny, Nz = size(grid)

    workgroup = heuristic_workgroup(arch, Nx, Ny, Nz)
   
    Nx′, Ny′, Nz′ = include_right_boundaries ? size(location, grid) : (Nx, Ny, Nz)

    Nx′, Ny′, Nz′ = flatten_reduced_dimensions((Nx′, Ny′, Nz′), reduced_dimensions)

    worksize = dims == :xyz ? (Nx′, Ny′, Nz′) :
               dims == :xy  ? (Nx′, Ny′) :
               dims == :xz  ? (Nx′, Nz′) :
               dims == :yz  ? (Ny′, Nz′) : throw(ArgumentError("Unsupported launch configuration: $dims"))

    return workgroup, worksize
end

"""
    launch!(arch, grid, layout, kernel!, args...; dependencies=nothing, kwargs...)

Launches `kernel!`, with arguments `args` and keyword arguments `kwargs`,
over the `dims` of `grid` on the architecture `arch`.

Returns an `event` token associated with the `kernel!` launch.

The keyword argument `dependencies` is an `Event` or `MultiEvent` specifying prior kernels
that must complete before `kernel!` is launched.
"""
function launch!(arch, grid, dims, kernel!, args...;
                 dependencies = nothing,
                 include_right_boundaries = false,
                 reduced_dimensions = (),
                 location = nothing,
                 kwargs...)

    workgroup, worksize = work_layout(arch, grid, dims,
                                      include_right_boundaries = include_right_boundaries,
                                      reduced_dimensions = reduced_dimensions,
                                      location = location)

    loop! = kernel!(Architectures.device(arch), workgroup, worksize)

    @debug "Launching kernel $kernel! with worksize $worksize"

    event = loop!(args...; dependencies=dependencies, kwargs...)

    return event
end

# When dims::Val
@inline launch!(arch, grid, ::Val{dims}, args...; kwargs...) where dims = launch!(arch, grid, dims, args...; kwargs...)
